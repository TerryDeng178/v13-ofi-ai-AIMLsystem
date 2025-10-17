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

@dataclass
class CVDConfig:
    """CVD计算器配置类"""
    z_window: int = 300           # Z-score滚动窗口大小
    ema_alpha: float = 0.2        # EMA平滑系数
    use_tick_rule: bool = True    # 无 is_buy 时回退到 Tick Rule
    warmup_min: int = 5           # 冷启动阈值下限

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
        "bad_points", "_last_price", "_last_event_time_ms", "_last_side"
    )
    
    def __init__(self, symbol: str, cfg: Optional[CVDConfig] = None) -> None:
        """
        初始化CVD计算器
        
        参数:
            symbol: 交易对符号（如"ETHUSDT"）
            cfg: CVD配置对象，默认None使用默认配置
        """
        self.symbol = (symbol or "").upper()
        self.cfg = cfg or CVDConfig()
        self.cvd: float = 0.0
        self.ema_cvd: Optional[float] = None
        self._hist: deque[float] = deque(maxlen=self.cfg.z_window)
        self.bad_points: int = 0
        self._last_price: Optional[float] = None
        self._last_event_time_ms: Optional[int] = None
        self._last_side: Optional[bool] = None  # 用于 Tick Rule price==last_price 情况

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

    def get_state(self) -> Dict[str, Any]:
        """
        获取计算器当前状态
        
        返回:
            Dict: 包含symbol, cvd, z_cvd, ema_cvd等状态信息
        """
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
            },
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
                else:  # price == last_price，沿用上一笔方向
                    side = self._last_side
        
        if side is None:
            self.bad_points += 1
            return self._result(None, warmup=None, std_zero=None, event_time_ms=event_time_ms)

        # 更新累计
        delta = float(qty) if side else -float(qty)
        self.cvd += delta

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
            self._last_event_time_ms = int(event_time_ms)
        self._last_side = side  # 记录方向用于下次 Tick Rule

        # Z-score：上一窗口为基线
        self._hist.append(self.cvd)
        z_val, warmup, std_zero = self._z_last_excl()
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
