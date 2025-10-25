# -*- coding: utf-8 -*-
"""
Real CVD Calculator - Task 1.2.6
真实CVD计算器（基于成交流）

功能：
- 基于主动买卖成交计算累积成交量差 (Cumulative Volume Delta)
- Z-score标准化（"上一窗口"基线 + std_zero标记）
- EMA平滑
- Tick Rule买卖方向判定（可选）
- 纯计算，无I/O操作

核心实现要点：
1. CVD累积：
   - 买入成交：CVD += qty
   - 卖出成交：CVD -= qty
   
2. 方向判定：
   - 优先使用 is_buy 字段（来自数据源）
   - 回退到 Tick Rule（与上一成交价比较）
   
3. Z-score（优化版）：
   - 基线="上一窗口"（不包含当前cvd），避免当前值稀释
   - warmup_threshold = max(5, z_window//5)，不足返回 z_cvd=None
   - std <= 1e-9 则 z_cvd=0.0 且 meta.std_zero=True

4. EMA：
   - ema_alpha可配，首次用当前cvd初始化，其后标准递推

5. 状态与边界：
   - reset()/get_state() 可观测
   - 负量/NaN/Inf → 计入 bad_points

作者: V13 OFI+CVD+AI System
创建时间: 2025-10-17
最后优化: 2025-10-17 (Task 1.2.6)
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional, Iterable, Tuple, Dict, Any
import math
import logging

# 模块级logger
logger = logging.getLogger(__name__)
logger.propagate = True

@dataclass
class CVDConfig:
    """CVD计算器配置类"""
    z_window: int = 150           # 300 → 150 (缩短Z-score窗口，快速点火)
    ema_alpha: float = 0.2        # EMA平滑系数
    use_tick_rule: bool = True    # 无 is_buy 时回退到 Tick Rule
    warmup_min: int = 3           # 5 → 3 (降低暖启动阈值)
    auto_flip_enabled: bool = False  # 是否启用自动翻转
    auto_flip_threshold: float = 0.04  # AUC提升阈值，超过此值自动翻转
    
    # P1.1 Delta-Z配置
    z_mode: str = "level"         # Z-score模式: "level"(旧版) | "delta"(新版)
    half_life_trades: int = 300   # Delta-Z半衰期（笔数）
    winsor_limit: float = 8.0     # Z-score截断阈值
    freeze_min: int = 25          # Z-score最小样本数（降低冻结门槛）
    stale_threshold_ms: int = 5000 # Stale冻结阈值（毫秒）
    
    # 空窗后冻结配置（事件时间间隔）
    soft_freeze_ms: int = 4000    # 软冻结阈值（4-5s，首1笔冻结）
    hard_freeze_ms: int = 5000    # 硬冻结阈值（>5s，首2笔冻结）
    
    # Step 1 稳健尺度地板配置
    scale_mode: str = "ewma"      # 尺度模式: "ewma" | "hybrid"
    ewma_fast_hl: int = 80        # 快EWMA半衰期（笔数）
    mad_window_trades: int = 300  # MAD窗口大小（笔数）
    mad_scale_factor: float = 1.4826 # MAD还原为σ的一致性系数
    
    # Step 1 微调配置
    scale_fast_weight: float = 0.30  # 快EWMA权重 (fast:slow = 0.30:0.70)
    scale_slow_weight: float = 0.70  # 慢EWMA权重
    mad_multiplier: float = 1.30     # MAD地板安全系数
    post_stale_freeze: int = 2       # 空窗后首N笔冻结

class RealCVDCalculator:
    """
    真实CVD计算器（基于成交流）
    
    核心功能:
    1. 基于主动买卖成交计算CVD
    2. Z-score标准化（滚动窗口300）
    3. EMA平滑（alpha=0.2）
    4. Tick Rule方向判定（可选）
    
    计算公式:
    - CVD = Σ(买入qty - 卖出qty)
    - z_cvd = (CVD - mean(CVD_hist)) / std(CVD_hist)
    - ema_cvd = alpha * CVD + (1-alpha) * ema_cvd_prev
    
    使用示例:
        >>> config = CVDConfig(z_window=300, use_tick_rule=True)
        >>> calc = RealCVDCalculator("ETHUSDT", config)
        >>> # 买入成交
        >>> result = calc.update_with_trade(price=3245.5, qty=10.5, is_buy=True)
        >>> print(f"CVD={result['cvd']:.4f}, Z-score={result['z_cvd']}")
    """
    
    __slots__ = (
        "symbol", "cfg", "cvd", "ema_cvd", "_hist", 
        "bad_points", "_last_price", "_last_event_time_ms", "_last_side",
        # P1.1 Delta-Z状态
        "_ewma_abs_delta", "_trades_count", "_alpha", "_last_delta",
        # Step 1 稳健尺度地板状态
        "_ewma_abs_fast", "_alpha_fast", "_mad_buf",
        # Step 1 微调状态
        "_post_stale_remaining", "_prev_event_time_ms",
        # 新增：上一笔Z与尺度诊断缓存（避免再估计口径错位）
        "_last_z_raw", "_last_z_post", "_last_is_warmup", "_last_is_flat",
        "_last_scale_diag",
        # P0 时间衰减EWMA配置
        "_tau_fast_sec", "_tau_slow_sec", "_last_event_ts",
        # P1 活动度自适应配置
        "_recv_rate_window", "_r_ref", "_gamma", "_beta",
        # 自动翻转状态
        "_is_flipped", "_flip_reason", "_tick_rule_count"
    )
    
    def __init__(self, symbol: str, cfg: Optional[CVDConfig] = None, config_loader=None, 
                 mad_multiplier: Optional[float] = None, scale_fast_weight: Optional[float] = None,
                 z_hi: Optional[float] = None, z_mid: Optional[float] = None) -> None:
        """
        初始化CVD计算器
        
        参数:
            symbol: 交易对符号（如"ETHUSDT"）
            cfg: CVD配置对象，默认None使用默认配置
            config_loader: 配置加载器实例，用于从统一配置系统加载参数
            mad_multiplier: MAD乘数（Fix Pack v2: 强制参数）
            scale_fast_weight: 快速权重（Fix Pack v2: 强制参数）
            z_hi: Z高阈值（Fix Pack v2: 强制参数）
            z_mid: Z中阈值（Fix Pack v2: 强制参数）
        """
        self.symbol = (symbol or "").upper()
        
        if config_loader:
            # 从统一配置系统加载参数
            self.cfg = self._load_from_config_loader(config_loader, symbol)
        else:
            self.cfg = cfg or CVDConfig()
        
        # Fix Pack v2: 强制参数覆盖
        if mad_multiplier is not None:
            self.cfg.mad_multiplier = mad_multiplier
        if scale_fast_weight is not None:
            self.cfg.scale_fast_weight = scale_fast_weight
        # 提示：CVDConfig 未定义 z_hi/z_mid；如需使用，请在配置类中显式加入或删除这两项。
        self.cvd: float = 0.0
        self.ema_cvd: Optional[float] = None
        self._hist: deque[float] = deque(maxlen=self.cfg.z_window)
        self.bad_points: int = 0
        self._last_price: Optional[float] = None
        self._last_event_time_ms: Optional[int] = None
        self._last_side: Optional[bool] = None  # 用于 Tick Rule price==last_price 情况
        
        # P1.1 Delta-Z状态初始化
        self._ewma_abs_delta: float = 0.0
        self._trades_count: int = 0
        self._alpha: float = 1 - math.exp(math.log(0.5) / max(1, self.cfg.half_life_trades))
        self._last_delta: Optional[float] = None
        
        # Step 1 稳健尺度地板状态初始化
        self._ewma_abs_fast: float = 0.0
        self._alpha_fast: float = 1 - math.exp(math.log(0.5) / max(1, self.cfg.ewma_fast_hl))
        self._mad_buf: deque[float] = deque(maxlen=self.cfg.mad_window_trades)
        
        # P0 时间衰减EWMA配置
        self._tau_fast_sec: float = 60.0    # 快速EWMA时间半衰期：1分钟
        self._tau_slow_sec: float = 600.0   # 慢速EWMA时间半衰期：10分钟
        self._last_event_ts: Optional[float] = None  # 上一笔事件时间戳（秒）
        
        # P1 活动度自适应配置
        # 用时间窗口维护TPS（按秒）——不限制个数，按时间裁剪
        self._recv_rate_window: deque[float] = deque()  # 存事件秒级时间戳
        self._r_ref: float = 1.5  # 基准活动度（tps）
        self._gamma: float = 0.7  # 地板抬升系数
        self._beta: float = 1.0   # 权重削弱系数
        
        # Step 1 微调状态初始化
        self._post_stale_remaining: int = 0
        self._prev_event_time_ms: Optional[int] = None
        
        # Z缓存与尺度诊断
        self._last_z_raw: Optional[float] = None
        self._last_z_post: Optional[float] = None
        self._last_is_warmup: bool = True
        self._last_is_flat: bool = False
        self._last_scale_diag: Dict[str, float] = {}
        
        # 自动翻转状态
        self._is_flipped: bool = False  # 当前是否已翻转
        self._flip_reason: Optional[str] = None  # 翻转原因
        self._tick_rule_count: int = 0  # tick-rule传播计数
        
        # 配置验证和诊断日志
        self._print_effective_config()
    
    def _load_from_config_loader(self, config_loader, symbol: str) -> CVDConfig:
        """
        从统一配置系统加载CVD参数
        
        参数:
            config_loader: 配置加载器实例
            symbol: 交易对符号
            
        返回:
            CVD配置对象
        """
        try:
            # 获取CVD配置
            cvd_config = config_loader.get('components.cvd', {})
            
            # 提取配置参数
            z_window = cvd_config.get('z_window', 300)
            ema_alpha = cvd_config.get('ema_alpha', 0.2)
            use_tick_rule = cvd_config.get('use_tick_rule', True)
            warmup_min = cvd_config.get('warmup_min', 5)
            
            # P1.1 Delta-Z配置
            z_mode = cvd_config.get('z_mode', 'level')
            half_life_trades = cvd_config.get('half_life_trades', 300)
            winsor_limit = cvd_config.get('winsor_limit', 8.0)
            freeze_min = cvd_config.get('freeze_min', 50)
            stale_threshold_ms = cvd_config.get('stale_threshold_ms', 5000)
            
            # 空窗后冻结配置
            soft_freeze_ms = cvd_config.get('soft_freeze_ms', 4000)
            hard_freeze_ms = cvd_config.get('hard_freeze_ms', 5000)
            
            # Step 1 稳健尺度地板配置
            scale_mode = cvd_config.get('scale_mode', 'ewma')
            ewma_fast_hl = cvd_config.get('ewma_fast_hl', 80)
            mad_window_trades = cvd_config.get('mad_window_trades', 300)
            mad_scale_factor = cvd_config.get('mad_scale_factor', 1.4826)
            
            # Step 1 微调配置
            scale_fast_weight = cvd_config.get('scale_fast_weight', 0.30)
            scale_slow_weight = cvd_config.get('scale_slow_weight', 0.70)
            mad_multiplier = cvd_config.get('mad_multiplier', 1.30)
            post_stale_freeze = cvd_config.get('post_stale_freeze', 2)
            
            # 创建配置对象
            return CVDConfig(
                z_window=z_window,
                ema_alpha=ema_alpha,
                use_tick_rule=use_tick_rule,
                warmup_min=warmup_min,
                z_mode=z_mode,
                half_life_trades=half_life_trades,
                winsor_limit=winsor_limit,
                freeze_min=freeze_min,
                stale_threshold_ms=stale_threshold_ms,
                soft_freeze_ms=soft_freeze_ms,
                hard_freeze_ms=hard_freeze_ms,
                scale_mode=scale_mode,
                ewma_fast_hl=ewma_fast_hl,
                mad_window_trades=mad_window_trades,
                mad_scale_factor=mad_scale_factor,
                scale_fast_weight=scale_fast_weight,
                scale_slow_weight=scale_slow_weight,
                mad_multiplier=mad_multiplier,
                post_stale_freeze=post_stale_freeze
            )
            
        except Exception as e:
            # 如果配置加载失败，使用默认配置并记录警告
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load CVD config from config_loader: {e}. Using default config.")
            return CVDConfig()

    def _print_effective_config(self) -> None:
        """打印有效配置，用于验证Step 1.6是否正确加载"""
        print(f"[CVD] Effective config for {self.symbol}:")
        print(f"  Z_MODE={self.cfg.z_mode}")  # 防止误配置
        print(f"  HALF_LIFE_TRADES={self.cfg.half_life_trades}")
        print(f"  WINSOR_LIMIT={self.cfg.winsor_limit}")
        print(f"  STALE_THRESHOLD_MS={self.cfg.stale_threshold_ms}")
        print(f"  FREEZE_MIN={self.cfg.freeze_min}")
        print(f"  SOFT_FREEZE_MS={self.cfg.soft_freeze_ms}")  # 软冻结阈值
        print(f"  HARD_FREEZE_MS={self.cfg.hard_freeze_ms}")  # 硬冻结阈值
        print(f"  SCALE_MODE={self.cfg.scale_mode}")
        print(f"  EWMA_FAST_HL={self.cfg.ewma_fast_hl}")
        # 打印归一化后的权重
        w_fast = max(0.0, min(1.0, self.cfg.scale_fast_weight))
        w_slow = max(0.0, min(1.0, self.cfg.scale_slow_weight))
        w_sum = w_fast + w_slow
        if w_sum > 1e-9:
            w_fast_norm, w_slow_norm = w_fast / w_sum, w_slow / w_sum
            print(f"  SCALE_FAST_WEIGHT={self.cfg.scale_fast_weight} → {w_fast_norm:.3f} (归一化后)")
            print(f"  SCALE_SLOW_WEIGHT={self.cfg.scale_slow_weight} → {w_slow_norm:.3f} (归一化后)")
        else:
            print(f"  SCALE_FAST_WEIGHT={self.cfg.scale_fast_weight} (slow={self.cfg.scale_slow_weight})")
        print(f"  MAD_WINDOW_TRADES={self.cfg.mad_window_trades}")
        print(f"  MAD_SCALE_FACTOR={self.cfg.mad_scale_factor}")
        print(f"  MAD_MULTIPLIER={self.cfg.mad_multiplier}")

    # 状态管理
    def reset(self) -> None:
        """
        重置计算器状态，清空所有历史数据
        """
        self.cvd = 0.0
        self.ema_cvd = None
        self._hist.clear()
        self.bad_points = 0
        self._last_price = None
        self._last_event_time_ms = None
        self._last_side = None
        
        # P1.1 Delta-Z状态重置
        self._ewma_abs_delta = 0.0
        self._trades_count = 0
        self._last_delta = None
        
        # Step 1 稳健尺度地板状态重置
        self._ewma_abs_fast = 0.0
        self._mad_buf.clear()
        
        # P0 时间衰减EWMA重置
        self._last_event_ts = None
        
        # P1 活动度自适应重置
        self._recv_rate_window.clear()
        
        # Step 1 微调状态重置
        self._post_stale_remaining = 0
        self._prev_event_time_ms = None
        
        # Z缓存重置
        self._last_z_raw = None
        self._last_z_post = None
        self._last_is_warmup = True
        self._last_is_flat = False
        self._last_scale_diag = {}

    def get_state(self) -> Dict[str, Any]:
        """
        获取计算器当前状态
        
        返回:
            Dict: 包含symbol, cvd, z_cvd, ema_cvd等状态信息
        """
        if self.cfg.z_mode == "delta":
            warmup, std_zero, z_val = self._peek_delta_z()
        else:
            warmup, std_zero, z_val = self._peek_z()
            
        return {
            "symbol": self.symbol,
            "cvd": self.cvd,
            "ema_cvd": self.ema_cvd,
            "z_cvd": z_val,
            "meta": {
                "bad_points": self.bad_points,
                "warmup": warmup,
                "std_zero": std_zero,
                "last_price": self._last_price,
                "event_time_ms": self._last_event_time_ms,
                "z_mode": self.cfg.z_mode,
                "delta": self._last_delta,
                "ewma_abs_delta": self._ewma_abs_delta,
                "trades_count": self._trades_count,
                "is_flipped": self._is_flipped,
                "flip_reason": self._flip_reason,
            },
        }
    
    def set_flip_state(self, is_flipped: bool, reason: Optional[str] = None) -> None:
        """
        设置翻转状态
        
        参数:
            is_flipped: 是否已翻转
            reason: 翻转原因
        """
        self._is_flipped = is_flipped
        self._flip_reason = reason
    
    def get_flip_state(self) -> Tuple[bool, Optional[str]]:
        """
        获取翻转状态
        
        返回:
            Tuple[bool, Optional[str]]: (是否已翻转, 翻转原因)
        """
        return self._is_flipped, self._flip_reason
    
    def get_last_zscores(self) -> Dict[str, Any]:
        """返回上一笔计算时缓存的 Z（raw=截断前，cvd=截断后）与标志，不再二次估计。"""
        return {
            "z_raw": self._last_z_raw,
            "z_cvd": self._last_z_post,
            "is_warmup": self._last_is_warmup,
            "is_flat": self._last_is_flat,
        }
    
    @property
    def last_price(self) -> Optional[float]:
        """最后成交价（用于外部访问）"""
        return self._last_price

    # 主入口：单笔成交
    def update_with_trade(
        self, *, price: Optional[float] = None, qty: float,
        is_buy: Optional[bool] = None, event_time_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        基于单笔成交更新CVD
        
        参数:
            price: 成交价格（用于Tick Rule，可选）
            qty: 成交数量（必需）
            is_buy: 是否买入（True=买入，False=卖出，None=使用Tick Rule）
            event_time_ms: 事件时间戳（毫秒，可选）
        
        返回:
            Dict: {
                "symbol": 交易对,
                "cvd": CVD值,
                "z_cvd": Z-score标准化后的CVD (warmup期间为None),
                "ema_cvd": EMA平滑后的CVD,
                "meta": {
                    "bad_points": 坏数据点计数,
                    "warmup": 是否在warmup期,
                    "std_zero": 标准差是否为0,
                    "last_price": 最后成交价,
                    "event_time_ms": 事件时间戳
                }
            }
        """
        # 数据清洗：数量必须为有限非负
        if qty is None or not isinstance(qty, (int, float)) or not math.isfinite(qty) or qty < 0:
            self.bad_points += 1
            return self._result(None, warmup=None, std_zero=None, event_time_ms=event_time_ms)

        # 判定方向：优先 is_buy；否则 Tick Rule；仍无法判定则忽略并计数
        side = is_buy
        if side is None and self.cfg.use_tick_rule and price is not None:
            if self._last_price is not None:
                if price > self._last_price:
                    side = True  # 买入
                elif price < self._last_price:
                    side = False  # 卖出
                else:  # price == last_price，限制tick-rule回退传播
                    # 检查是否超过传播限制
                    if hasattr(self, '_tick_rule_count'):
                        self._tick_rule_count += 1
                    else:
                        self._tick_rule_count = 1
                    
                    # 限制传播长度：最多连续5笔或超过2秒
                    max_consecutive = 5
                    max_time_ms = 2000
                    
                    time_exceeded = False
                    if event_time_ms is not None and self._last_event_time_ms is not None:
                        time_exceeded = (event_time_ms - self._last_event_time_ms) > max_time_ms
                    
                    if self._tick_rule_count <= max_consecutive and not time_exceeded:
                        side = self._last_side  # 沿用上一笔方向
                    else:
                        side = None  # 超过限制，不累计
                        self._tick_rule_count = 0  # 重置计数
            else:
                # 价格变化，重置tick-rule计数
                if hasattr(self, '_tick_rule_count'):
                    self._tick_rule_count = 0
        
        if side is None:
            self.bad_points += 1
            return self._result(None, warmup=None, std_zero=None, event_time_ms=event_time_ms)

        # 更新累计
        delta = float(qty) if side else -float(qty)
        self.cvd += delta
        self._last_delta = delta
        self._trades_count += 1

        # EMA
        if self.ema_cvd is None:
            self.ema_cvd = self.cvd
        else:
            a = float(self.cfg.ema_alpha)
            self.ema_cvd = a * self.cvd + (1.0 - a) * self.ema_cvd

        # 维护 last
        if price is not None and math.isfinite(price):
            self._last_price = float(price)
        if event_time_ms is not None:
            # Step 1.1: 保存前一个event_time_ms用于软冻结计算
            self._prev_event_time_ms = self._last_event_time_ms
            self._last_event_time_ms = int(event_time_ms)
        self._last_side = side  # 记录方向用于下次 Tick Rule

        # Z-score计算：根据模式选择
        if self.cfg.z_mode == "delta":
            # P1.1 Delta-Z模式
            self._hist.append(self.cvd)  # 保持历史记录用于兼容
            z_val, warmup, std_zero = self._z_delta()
        else:
            # 原有Level-Z模式
            self._hist.append(self.cvd)
            z_val, warmup, std_zero = self._z_last_excl()
            # level模式下 raw/post 相同
            self._last_z_raw = z_val
            self._last_z_post = z_val
            self._last_is_warmup = bool(warmup)
            self._last_is_flat = bool(std_zero)
            
        return self._result(z_val, warmup, std_zero, event_time_ms=event_time_ms)

    # 适配交易所消息格式
    def update_with_agg_trade(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        适配Binance aggTrade消息格式
        
        参数:
            msg: Binance aggTrade消息，包含字段：
                - 'p': 价格 (price)
                - 'q': 数量 (quantity)
                - 'm': 是否买方maker (isBuyerMaker, True=卖出, False=买入)
                - 'E': 事件时间 (event time, 毫秒)
        
        返回:
            Dict: update_with_trade() 的返回值
        
        注意:
            Binance的'm'字段含义是"买方是maker"，因此:
            - m=True → 卖方是taker → 主动卖出 → is_buy=False
            - m=False → 买方是taker → 主动买入 → is_buy=True
        """
        try:
            price = float(msg.get('p', 0))
            qty = float(msg.get('q', 0))
            m = msg.get('m', None)
            # Binance: m=True表示买方是maker，即卖方是taker（主动卖出）
            is_buy = not m if m is not None else None
            event_time_ms = int(msg.get('E', 0)) if 'E' in msg else None
            
            return self.update_with_trade(
                price=price,
                qty=qty,
                is_buy=is_buy,
                event_time_ms=event_time_ms
            )
        except (ValueError, TypeError, KeyError) as e:
            # 解析失败，计入坏数据点
            self.bad_points += 1
            return self._result(None, warmup=None, std_zero=None, event_time_ms=None)
    
    # 批量接口
    def update_with_trades(
        self, trades: Iterable[Tuple[Optional[float], float, Optional[bool], Optional[int]]]
    ) -> Dict[str, Any]:
        """
        批量成交更新（聚合更高效）
        
        参数:
            trades: 成交列表，每个元素为 (price, qty, is_buy, event_time_ms)
        
        返回:
            Dict: 最后一笔成交的 update_with_trade() 返回值
        """
        ret: Dict[str, Any] = {}
        for price, qty, is_buy, ts in trades:
            ret = self.update_with_trade(price=price, qty=qty, is_buy=is_buy, event_time_ms=ts)
        return ret

    # ——内部实现——
    def _z_last_excl(self) -> Tuple[Optional[float], bool, bool]:
        # 使用上一窗口（不含当前值）做基线
        if not self._hist:
            return None, True, False
        arr = list(self._hist)[:-1]
        warmup_threshold = max(int(self.cfg.z_window // 5), int(self.cfg.warmup_min))
        if len(arr) < max(1, warmup_threshold):
            return None, True, False
        mean, std = self._mean_std(arr)
        if std <= 1e-9:
            return 0.0, False, True
        z = (self.cvd - mean) / std
        return z, False, False

    def _peek_z(self) -> Tuple[bool, bool, Optional[float]]:
        """只读当前 z 的估计（不改变窗口），用于 get_state。"""
        if not self._hist:
            return True, False, None
        arr = list(self._hist)
        arr = arr[:-1] if len(arr) > 1 else []
        warmup_threshold = max(int(self.cfg.z_window // 5), int(self.cfg.warmup_min))
        if len(arr) < max(1, warmup_threshold):
            return True, False, None
        mean, std = self._mean_std(arr)
        if std <= 1e-9:
            return False, True, 0.0
        return False, False, (self.cvd - mean) / std

    @staticmethod
    def _mean_std(arr: Iterable[float]) -> Tuple[float, float]:
        n = 0
        s = 0.0
        ss = 0.0
        for v in arr:
            n += 1
            s += v
            ss += v * v
        if n == 0:
            return 0.0, 0.0
        mean = s / n
        var = ss / n - mean * mean
        if var < 0:
            var = 0.0
        return mean, math.sqrt(var)
    
    def _robust_mad_sigma(self) -> float:
        """
        Step 1: 计算稳健MAD尺度地板
        
        返回:
            float: MAD还原为σ的稳健估计，样本不足时返回0.0
        """
        if len(self._mad_buf) < max(50, self.cfg.mad_window_trades // 5):
            return 0.0
        
        # 计算中位数
        mad_values = list(self._mad_buf)
        mad_values.sort()
        n = len(mad_values)
        if n % 2 == 0:
            med = (mad_values[n//2-1] + mad_values[n//2]) / 2
        else:
            med = mad_values[n//2]
        
        # 计算MAD
        abs_deviations = [abs(x - med) for x in mad_values]
        abs_deviations.sort()
        if len(abs_deviations) % 2 == 0:
            mad = (abs_deviations[len(abs_deviations)//2-1] + abs_deviations[len(abs_deviations)//2]) / 2
        else:
            mad = abs_deviations[len(abs_deviations)//2]
        
        return self.cfg.mad_scale_factor * mad

    def _result(
        self, z_val: Optional[float], warmup: Optional[bool],
        std_zero: Optional[bool], *, event_time_ms: Optional[int]
    ) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "cvd": self.cvd,
            "z_cvd": z_val,
            "ema_cvd": self.ema_cvd,
            "meta": {
                "bad_points": self.bad_points,
                "warmup": bool(warmup) if warmup is not None else True,
                "std_zero": bool(std_zero) if std_zero is not None else False,
                "last_price": self._last_price,
                "event_time_ms": event_time_ms if event_time_ms is not None else self._last_event_time_ms,
            },
        }

    # P1.1 Delta-Z核心计算方法
    def _z_delta(self) -> Tuple[Optional[float], bool, bool]:
        """
        Delta-Z计算：z = ΔCVD / 稳健尺度 + winsor + 暖启动/空窗冻结
        
        Step 1增强：支持混合尺度地板（双EWMA + MAD地板）
        
        返回:
            (z_val, warmup, std_zero)
        """
        if self._last_delta is None:
            return None, True, False
            
        # P0 时间衰减EWMA更新
        abs_delta = abs(self._last_delta)
        current_event_ts = self._last_event_time_ms / 1000.0 if self._last_event_time_ms else None
        
        if self._trades_count == 1:
            # 第一笔交易，直接初始化
            self._ewma_abs_delta = abs_delta
            self._ewma_abs_fast = abs_delta
            if current_event_ts:
                self._last_event_ts = current_event_ts
        else:
            # 计算时间差（秒）
            if current_event_ts and self._last_event_ts:
                dt = current_event_ts - self._last_event_ts
                dt = max(0.001, dt)  # 最小1ms，避免除零
                
                # 时间衰减alpha计算
                alpha_fast_time = 1.0 - math.exp(-math.log(2) * dt / self._tau_fast_sec)
                alpha_slow_time = 1.0 - math.exp(-math.log(2) * dt / self._tau_slow_sec)
                
                # 时间衰减EWMA更新
                self._ewma_abs_fast = (1 - alpha_fast_time) * self._ewma_abs_fast + alpha_fast_time * abs_delta
                self._ewma_abs_delta = (1 - alpha_slow_time) * self._ewma_abs_delta + alpha_slow_time * abs_delta
                
                # 更新事件时间戳
                self._last_event_ts = current_event_ts
            else:
                # 回退到按笔更新（兼容性）
                self._ewma_abs_delta = self._alpha * abs_delta + (1 - self._alpha) * self._ewma_abs_delta
                self._ewma_abs_fast = self._alpha_fast * abs_delta + (1 - self._alpha_fast) * self._ewma_abs_fast
        
        # 更新MAD缓冲区
        self._mad_buf.append(self._last_delta)
        
        # P1 活动度自适应：更新"最近60秒"的时间窗口并计算TPS
        if current_event_ts is not None:
            self._recv_rate_window.append(current_event_ts)
            # 裁剪掉60秒以外的时间戳
            while self._recv_rate_window and (current_event_ts - self._recv_rate_window[0]) > 60.0:
                self._recv_rate_window.popleft()
        
        # 暖启动检查
        if self._trades_count < self.cfg.freeze_min:
            return None, True, False
            
        # 计算稳健尺度
        if self.cfg.scale_mode == "hybrid":
            # 混合尺度：双EWMA + MAD地板（Step 1微调）
            # 权重归一化：防止配置错误
            w_fast = max(0.0, min(1.0, self.cfg.scale_fast_weight))
            w_slow = max(0.0, min(1.0, self.cfg.scale_slow_weight))
            w_sum = w_fast + w_slow
            if w_sum <= 1e-9:
                w_fast, w_slow = 0.5, 0.5
            else:
                w_fast, w_slow = w_fast / w_sum, w_slow / w_sum
            
            # P1 活动度自适应：计算当前tps和自适应参数
            # 用固定60秒窗口估计TPS（更稳）
            current_tps = 0.0
            window_span = 0.0
            if len(self._recv_rate_window) >= 2:
                window_span = self._recv_rate_window[-1] - self._recv_rate_window[0]
                current_tps = len(self._recv_rate_window) / max(60.0, window_span)
            
            # P1 活动度自适应权重削弱
            w_fast_eff = w_fast * min(1.0, (current_tps / self._r_ref) ** self._beta)
            w_slow_eff = 1.0 - w_fast_eff
            
            ewma_mix = (w_fast_eff * self._ewma_abs_fast + 
                       w_slow_eff * self._ewma_abs_delta)
            
            # MAD→σ：_robust_mad_sigma() 已含 mad_scale_factor，这里直接乘安全系数
            sigma_floor_base = self._robust_mad_sigma() * self.cfg.mad_multiplier
            
            # P1 活动度自适应地板抬升
            boost = max(1.0, (self._r_ref / max(current_tps, 0.2)) ** self._gamma)
            sigma_floor = sigma_floor_base * boost
            
            # P0 修正floor_hit_rate统计口径（避免浮点比较误差）
            floor_used = (sigma_floor >= ewma_mix * (1 - 1e-6))  # 允许微小数值误差
            
            scale = max(ewma_mix, sigma_floor, 1e-9)
            
            # 诊断日志：检查反相/归一化问题（每300笔记录一次，避免阻塞）
            if self._trades_count % 1000 == 0:  # 每1000笔打印一次
                print(f"[DIAGNOSTIC] [count={self._trades_count}]:")
                print(f"  ewma_fast={self._ewma_abs_fast:.6f}")
                print(f"  ewma_slow={self._ewma_abs_delta:.6f}")
                print(f"  w_fast={self.cfg.scale_fast_weight}, w_slow={self.cfg.scale_slow_weight}")
                print(f"  w_fast+w_slow={self.cfg.scale_fast_weight + self.cfg.scale_slow_weight}")
                print(f"  ewma_mix={ewma_mix:.6f}")
                print(f"  mad_raw={self._robust_mad_sigma() / self.cfg.mad_scale_factor:.6f}")
                print(f"  sigma_floor={sigma_floor:.6f}")
                print(f"  scale={scale:.6f}")
                print(f"  delta={self._last_delta:.6f}")
                print(f"  z_raw={self._last_delta/scale:.6f}")
        else:
            # 原始EWMA尺度
            scale = max(self._ewma_abs_delta, 1e-9)
            
        # 尺度零检查
        if scale <= 1e-9:
            return None, False, True
            
        # Stale冻结检查：与上笔event_time_ms间隔 > stale_threshold_ms
        # 注意：这里应该检查当前event_time_ms与上一笔的间隔，而不是与自己的间隔
        if (self._last_event_time_ms is not None and 
            self._trades_count > 1 and 
            hasattr(self, '_prev_event_time_ms') and 
            self._prev_event_time_ms is not None and
            self._last_event_time_ms - self._prev_event_time_ms > self.cfg.stale_threshold_ms):
            # 设置空窗后首N笔冻结
            self._post_stale_remaining = self.cfg.post_stale_freeze
            return None, False, False
            
        # Step 1.1: 事件时间(E)分段冻结 - 基于重排后的事件时间E的相邻间隔
        if (self._last_event_time_ms is not None and 
            self._trades_count > 1 and
            hasattr(self, '_prev_event_time_ms') and 
            self._prev_event_time_ms is not None):
            interarrival_ms = self._last_event_time_ms - self._prev_event_time_ms
            if interarrival_ms > self.cfg.hard_freeze_ms:
                # 硬冻结：E间隔 > hard_freeze_ms → 首 2 笔 z=None
                self._post_stale_remaining = 2
                return None, False, False
            elif interarrival_ms > self.cfg.soft_freeze_ms:
                # 软冻结：soft_freeze_ms < E间隔 ≤ hard_freeze_ms → 首 1 笔 z=None
                self._post_stale_remaining = 1
                return None, False, False
            
        # 空窗后首N笔冻结检查
        if self._post_stale_remaining > 0:
            self._post_stale_remaining -= 1
            return None, False, False
            
        # 计算ΔZ（raw 在 winsor 之前；post 为截断后）
        z_raw = self._last_delta / scale
        z_post = max(min(z_raw, self.cfg.winsor_limit), -self.cfg.winsor_limit)
        # 缓存"上一笔"结果与尺度诊断
        self._last_z_raw = z_raw
        self._last_z_post = z_post
        self._last_is_warmup = False
        self._last_is_flat = False
        self._last_scale_diag = {
            "ewma_fast": self._ewma_abs_fast,
            "ewma_slow": self._ewma_abs_delta,
            "w_fast": self.cfg.scale_fast_weight,
            "w_slow": self.cfg.scale_slow_weight,
            "w_fast_eff": w_fast_eff if 'w_fast_eff' in locals() else self.cfg.scale_fast_weight,
            "w_slow_eff": w_slow_eff if 'w_slow_eff' in locals() else self.cfg.scale_slow_weight,
            # 记录真实使用的 ewma_mix
            "ewma_mix": ewma_mix,
            "sigma_floor_base": self._robust_mad_sigma() * self.cfg.mad_multiplier if self.cfg.scale_mode=="hybrid" else 0.0,
            "sigma_floor": sigma_floor if 'sigma_floor' in locals() else self._robust_mad_sigma() * self.cfg.mad_multiplier,
            "boost": boost if 'boost' in locals() else 1.0,
            "current_tps": current_tps if 'current_tps' in locals() else 0.0,
            "window_span_sec": window_span if 'window_span' in locals() else 0.0,
            "scale": scale,
            "floor_used": floor_used if 'floor_used' in locals() else False,  # P0 修正floor命中率统计
        }
        return z_post, False, False

    def _peek_delta_z(self) -> Tuple[bool, bool, Optional[float]]:
        """只读当前 Delta-Z 的估计（不改变状态），口径与 _z_delta 保持一致。"""
        if self._last_delta is None:
            return True, False, None
            
        # 暖启动检查
        if self._trades_count < self.cfg.freeze_min:
            return True, False, None
    
    def update_params(self, *, z_mode: str = None, freeze_min: int = None, 
                     z_window: int = None, ema_alpha: float = None):
        """
        更新CVD计算器参数
        
        Args:
            z_mode: Z-score模式 ("level" 或 "delta")
            freeze_min: 最小冻结样本数
            z_window: Z-score窗口大小
            ema_alpha: EMA平滑系数
        """
        if z_mode in {"level", "delta"}:
            self.cfg.z_mode = z_mode
            logger.info(f"Updated z_mode to {self.cfg.z_mode}")
            
        if isinstance(freeze_min, int) and freeze_min > 0:
            self.cfg.freeze_min = freeze_min
            logger.info(f"Updated freeze_min to {self.cfg.freeze_min}")
            
        if z_window and z_window != self.cfg.z_window:
            self.cfg.z_window = int(z_window)
            # 重建历史窗口
            from collections import deque
            self._hist = deque(list(self._hist)[-self.cfg.z_window:], maxlen=self.cfg.z_window)
            logger.info(f"Updated z_window to {self.cfg.z_window}")
            
        if ema_alpha is not None:
            self.cfg.ema_alpha = float(ema_alpha)
            logger.info(f"Updated ema_alpha to {self.cfg.ema_alpha}")
            
        # 计算稳健尺度（与 _z_delta 同口径：时间衰减 + 活动度自适应）
        if self.cfg.scale_mode == "hybrid":
            # 权重归一化
            w_fast = max(0.0, min(1.0, self.cfg.scale_fast_weight))
            w_slow = max(0.0, min(1.0, self.cfg.scale_slow_weight))
            w_sum = w_fast + w_slow
            if w_sum <= 1e-9: 
                w_fast, w_slow = 0.5, 0.5
            else:             
                w_fast, w_slow = w_fast / w_sum, w_slow / w_sum

            # 60秒窗口 tps
            current_tps = 0.0
            if len(self._recv_rate_window) >= 2:
                win_span = self._recv_rate_window[-1] - self._recv_rate_window[0]
                current_tps = len(self._recv_rate_window) / max(60.0, win_span)

            # 自适应权重 & 地板
            w_fast_eff = w_fast * min(1.0, (current_tps / self._r_ref) ** self._beta)
            w_slow_eff = 1.0 - w_fast_eff
            ewma_mix = w_fast_eff * self._ewma_abs_fast + w_slow_eff * self._ewma_abs_delta
            sigma_floor_base = self._robust_mad_sigma() * self.cfg.mad_multiplier
            boost = max(1.0, (self._r_ref / max(current_tps, 0.2)) ** self._gamma)
            sigma_floor = sigma_floor_base * boost
            scale = max(ewma_mix, sigma_floor, 1e-9)
        else:
            scale = max(self._ewma_abs_delta, 1e-9)
            
        # 尺度零检查
        if scale <= 1e-9:
            return False, True, None
            
        # 计算Delta-Z
        z = self._last_delta / scale
        
        # Winsorize截断
        z = max(min(z, self.cfg.winsor_limit), -self.cfg.winsor_limit)
        
        return False, False, z

    def get_z_stats(self) -> Dict[str, Any]:
        """
        获取Z-score统计信息（P1.1新增方法）
        
        返回:
            Dict: 包含Z-score相关统计信息
        """
        if self.cfg.z_mode == "delta":
            warmup, std_zero, z_val = self._peek_delta_z()
            base = {
                "z_mode": "delta",
                "z_value": z_val,
                "warmup": warmup,
                "std_zero": std_zero,
                "ewma_abs_delta": self._ewma_abs_delta,
                "trades_count": self._trades_count,
                "last_delta": self._last_delta,
                "alpha": self._alpha,
                "winsor_limit": self.cfg.winsor_limit,
                "freeze_min": self.cfg.freeze_min,
                "stale_threshold_ms": self.cfg.stale_threshold_ms,
            }
            # 附加尺度诊断，便于 Quiet 档归因
            base.update(self._last_scale_diag or {})
            return base
        else:
            warmup, std_zero, z_val = self._peek_z()
            return {
                "z_mode": "level",
                "z_value": z_val,
                "warmup": warmup,
                "std_zero": std_zero,
                "hist_size": len(self._hist),
                "z_window": self.cfg.z_window,
                "warmup_min": self.cfg.warmup_min,
            }
