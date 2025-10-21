# -*- coding: utf-8 -*-
"""
Real OFI Calculator - Task 1.2.1 (L1 OFI版本)
真实OFI计算器（快照模式 + L1价跃迁敏感）

功能：
- 基于订单簿快照计算L1 OFI (Order Flow Imbalance)
- 最优价跃迁冲击 + 5档深度加权计算
- Z-score标准化（"上一窗口"基线 + std_zero标记）
- EMA平滑
- 纯计算，无I/O操作

核心实现要点：
1. L1 OFI（价跃迁敏感）：
   - 最优档位：检测价格跃迁，计算冲击项
   - 价上涨：新最优价队列为正冲击，旧队列为负冲击
   - 价下跌：旧最优价队列为负冲击，新队列为正冲击
   - 其余档位：标准数量变化 Δbid_qty_k - Δask_qty_k

2. 权重与档位：
   - 默认权重 [0.4, 0.25, 0.2, 0.1, 0.05]
   - 按K档裁剪/填充并归一化，负值截为0，权重和为1

3. 输入清洗：
   - _pad_snapshot 保障价格为有限值、数量非负
   - 异常数据计入 bad_points

4. Z-score（优化版）：
   - 基线="上一窗口"（不包含当前ofi），避免当前值稀释
   - warmup_threshold = max(5, z_window//5)，不足返回 z_ofi=None
   - std <= 1e-9 则 z_ofi=0.0 且 meta.std_zero=True

5. EMA：
   - ema_alpha可配，首次用当前ofi初始化，其后标准递推

6. 状态与边界：
   - reset()/get_state() 可观测
   - L2增量模式显式 NotImplementedError（后续任务再做）

作者: V13 OFI+CVD+AI System
创建时间: 2025-10-17
最后优化: 2025-10-21 (L1 OFI价跃迁敏感版本)
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

@dataclass
class OFIConfig:
    """OFI计算器配置类"""
    levels: int = 5  # 订单簿档位数
    weights: Optional[List[float]] = None  # 自定义权重，默认None使用标准权重
    z_window: int = 300  # Z-score滚动窗口大小
    ema_alpha: float = 0.2  # EMA平滑系数

def _is_finite_number(x: float) -> bool:
    """
    检查是否为有效的有限数字
    
    参数:
        x: 待检查的数字
    
    返回:
        bool: 是否为有效有限数字
    """
    try:
        y = float(x)
        return y == y and y not in (float('inf'), float('-inf'))
    except Exception:
        return False

class RealOFICalculator:
    """
    真实OFI计算器（快照模式）
    
    核心功能:
    1. 基于5档订单簿快照计算OFI
    2. 深度加权: [0.4, 0.25, 0.2, 0.1, 0.05]
    3. Z-score标准化（滚动窗口300）
    4. EMA平滑（alpha=0.2）
    
    计算公式:
    - OFI_k = w_k * (Δbid_qty_k - Δask_qty_k)
    - OFI = Σ OFI_k (k=0 to K-1)
    - z_ofi = (OFI - mean(OFI_hist)) / std(OFI_hist)
    - ema_ofi = alpha * OFI + (1-alpha) * ema_ofi_prev
    
    使用示例:
        >>> config = OFIConfig(levels=5, z_window=300)
        >>> calc = RealOFICalculator("ETHUSDT", config)
        >>> bids = [[3245.5, 10.5], [3245.4, 8.3], ...]
        >>> asks = [[3245.6, 11.2], [3245.7, 9.5], ...]
        >>> result = calc.update_with_snapshot(bids, asks)
        >>> print(f"OFI={result['ofi']:.4f}, Z-score={result['z_ofi']:.4f}")
    """
    
    __slots__ = (
        "symbol", "K", "w", "z_window", "ema_alpha",
        "bids", "asks", "prev_bids", "prev_asks",
        "ofi_hist", "ema_ofi", "bad_points",
        "bid_jump_up_cnt", "bid_jump_down_cnt", "ask_jump_up_cnt", "ask_jump_down_cnt",
        "bid_jump_up_impact_sum", "bid_jump_down_impact_sum", 
        "ask_jump_up_impact_sum", "ask_jump_down_impact_sum"
    )
    
    def __init__(self, symbol: str, cfg: OFIConfig = None, config_loader=None):
        """
        初始化OFI计算器
        
        参数:
            symbol: 交易对符号（如"ETHUSDT"）
            cfg: OFI配置对象，默认None使用默认配置
            config_loader: 配置加载器实例，用于从统一配置系统加载参数
        """
        if config_loader:
            # 从统一配置系统加载参数
            cfg = self._load_from_config_loader(config_loader, symbol)
        elif cfg is None:
            cfg = OFIConfig()
            
        self.symbol = (symbol or "").upper()
        self.K = int(cfg.levels) if cfg.levels and cfg.levels > 0 else 5
        
        # 初始化权重（默认5档标准权重）
        default_w = [0.4, 0.25, 0.2, 0.1, 0.05]
        if cfg.weights is None:
            # 使用默认权重，裁剪或填充到K档
            w_raw = default_w[:self.K] if len(default_w) >= self.K else (
                default_w + [0.0] * max(0, self.K - len(default_w))
            )
        else:
            # 使用自定义权重
            w_raw = [float(x) for x in cfg.weights[:self.K]] + [0.0] * max(0, self.K - len(cfg.weights))
        
        # 归一化权重（确保总和为1）
        total = sum(max(0.0, x) for x in w_raw)
        if total <= 0.0:
            raise ValueError("weights must have positive sum")
        self.w = [max(0.0, x) / total for x in w_raw]
        
        self.z_window = int(cfg.z_window) if cfg.z_window and cfg.z_window > 0 else 300
        self.ema_alpha = float(cfg.ema_alpha)
        
        # 初始化订单簿缓存
        self.bids = [[0.0, 0.0] for _ in range(self.K)]
        self.asks = [[0.0, 0.0] for _ in range(self.K)]
        self.prev_bids = [[0.0, 0.0] for _ in range(self.K)]
        self.prev_asks = [[0.0, 0.0] for _ in range(self.K)]
        
        # 初始化历史数据
        self.ofi_hist = deque(maxlen=self.z_window)
        self.ema_ofi: Optional[float] = None
        self.bad_points = 0
        
        # L1价跃迁诊断统计
        self.bid_jump_up_cnt = 0
        self.bid_jump_down_cnt = 0
        self.ask_jump_up_cnt = 0
        self.ask_jump_down_cnt = 0
        self.bid_jump_up_impact_sum = 0.0
        self.bid_jump_down_impact_sum = 0.0
        self.ask_jump_up_impact_sum = 0.0
        self.ask_jump_down_impact_sum = 0.0
    
    def _load_from_config_loader(self, config_loader, symbol: str) -> OFIConfig:
        """
        从统一配置系统加载OFI参数
        
        参数:
            config_loader: 配置加载器实例
            symbol: 交易对符号
            
        返回:
            OFI配置对象
        """
        try:
            # 获取OFI配置
            ofi_config = config_loader.get('components.ofi', {})
            binance_config = config_loader.get('binance', {})
            
            # 提取配置参数
            levels = ofi_config.get('levels', binance_config.get('ofi', {}).get('levels', 5))
            weights = ofi_config.get('weights', binance_config.get('ofi', {}).get('weights', [0.4, 0.25, 0.2, 0.1, 0.05]))
            z_window = ofi_config.get('z_window', binance_config.get('ofi', {}).get('window_size', 300))
            ema_alpha = ofi_config.get('ema_alpha', 0.2)
            
            # 创建配置对象
            return OFIConfig(
                levels=levels,
                weights=weights,
                z_window=z_window,
                ema_alpha=ema_alpha
            )
            
        except Exception as e:
            # 如果配置加载失败，使用默认配置并记录警告
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load OFI config from config_loader: {e}. Using default config.")
            return OFIConfig()  # 坏数据点计数器

    def _pad_snapshot(self, arr: List[Tuple[float, float]]) -> List[List[float]]:
        """
        填充订单簿快照到K档，处理无效数据
        
        参数:
            arr: 订单簿数据 [(价格, 数量), ...]
        
        返回:
            List[List[float]]: K档订单簿 [[价格, 数量], ...]
        """
        out = [[0.0, 0.0] for _ in range(self.K)]
        n = min(len(arr or []), self.K)
        bad = False
        
        for i in range(n):
            p, q = arr[i]
            # 检查价格有效性
            if not _is_finite_number(p):
                bad = True
                p = 0.0
            # 检查数量有效性（必须非负）
            if not _is_finite_number(q) or float(q) < 0:
                bad = True
                q = 0.0
            out[i][0] = float(p)
            out[i][1] = float(q)
        
        if bad:
            self.bad_points += 1
        
        return out

    @staticmethod
    def _mean_std(values: List[float]) -> Tuple[float, float]:
        """
        计算均值和标准差（样本标准差）
        
        参数:
            values: 数值列表
        
        返回:
            Tuple[float, float]: (均值, 标准差)
        """
        n = len(values)
        if n == 0:
            return 0.0, 0.0
        
        m = sum(values) / n
        if n == 1:
            return m, 0.0
        
        var = sum((x - m) * (x - m) for x in values) / (n - 1)
        return m, var ** 0.5

    def reset(self) -> None:
        """
        重置计算器状态，清空所有历史数据
        """
        for i in range(self.K):
            self.bids[i][0] = self.bids[i][1] = 0.0
            self.asks[i][0] = self.asks[i][1] = 0.0
            self.prev_bids[i][0] = self.prev_bids[i][1] = 0.0
            self.prev_asks[i][0] = self.prev_asks[i][1] = 0.0
        
        self.ofi_hist.clear()
        self.ema_ofi = None
        self.bad_points = 0

    def get_state(self) -> Dict:
        """
        获取计算器当前状态
        
        返回:
            Dict: 包含symbol, levels, weights, bids, asks等状态信息
        """
        return {
            "symbol": self.symbol,
            "levels": self.K,
            "weights": list(self.w),
            "bids": [list(x) for x in self.bids],
            "asks": [list(x) for x in self.asks],
            "bad_points": self.bad_points,
            "ema_ofi": self.ema_ofi,
            "ofi_hist_len": len(self.ofi_hist),
        }

    def update_with_snapshot(
        self, 
        bids: List[Tuple[float, float]], 
        asks: List[Tuple[float, float]], 
        event_time_ms: Optional[int] = None
    ) -> Dict:
        """
        基于订单簿快照更新OFI
        
        参数:
            bids: 买单列表 [(价格, 数量), ...] 按价格降序
            asks: 卖单列表 [(价格, 数量), ...] 按价格升序
            event_time_ms: 事件时间戳（毫秒），可选
        
        返回:
            Dict: {
                "symbol": 交易对,
                "event_time_ms": 事件时间,
                "ofi": OFI值,
                "k_components": 各档OFI贡献 [ofi_0, ofi_1, ...],
                "z_ofi": Z-score标准化后的OFI (warmup期间为None),
                "ema_ofi": EMA平滑后的OFI,
                "meta": {
                    "levels": 档位数,
                    "weights": 权重列表,
                    "bad_points": 坏数据点计数,
                    "warmup": 是否在warmup期
                }
            }
        """
        # 保存上一帧订单簿
        for i in range(self.K):
            self.prev_bids[i][0] = self.bids[i][0]
            self.prev_bids[i][1] = self.bids[i][1]
            self.prev_asks[i][0] = self.asks[i][0]
            self.prev_asks[i][1] = self.asks[i][1]
        
        # 更新当前订单簿
        self.bids = self._pad_snapshot(bids)
        self.asks = self._pad_snapshot(asks)

        # 计算L1 OFI（最优价跃迁敏感版本）
        k_components = []
        ofi_val = 0.0
        
        # L1 OFI: 最优价跃迁冲击 + 其余档位数量变化
        for i in range(self.K):
            if i == 0:  # 最优档位：处理价跃迁冲击
                # 检查bid最优价是否变化
                bid_price_changed = abs(self.bids[i][0] - self.prev_bids[i][0]) > 1e-8
                ask_price_changed = abs(self.asks[i][0] - self.prev_asks[i][0]) > 1e-8
                
                if bid_price_changed or ask_price_changed:
                    # 价跃迁冲击：新最优价队列为正冲击，旧最优价队列为负冲击
                    bid_impact = 0.0
                    ask_impact = 0.0
                    
                    if self.bids[i][0] > self.prev_bids[i][0]:  # bid价上涨
                        self.bid_jump_up_cnt += 1
                        # 新最优价队列为正冲击
                        bid_impact = self.bids[i][1]
                        # 旧最优价队列为负冲击（如果存在）
                        if self.prev_bids[i][1] > 0:
                            bid_impact -= self.prev_bids[i][1]
                        self.bid_jump_up_impact_sum += bid_impact
                    elif self.bids[i][0] < self.prev_bids[i][0]:  # bid价下跌
                        self.bid_jump_down_cnt += 1
                        # 旧最优价队列为负冲击
                        bid_impact = -self.prev_bids[i][1]
                        # 新最优价队列为正冲击
                        if self.bids[i][1] > 0:
                            bid_impact += self.bids[i][1]
                        self.bid_jump_down_impact_sum += bid_impact
                    else:  # 价格不变，用数量变化
                        bid_impact = self.bids[i][1] - self.prev_bids[i][1]
                    
                    # ask端对称处理（负号）
                    if self.asks[i][0] > self.prev_asks[i][0]:  # ask价上涨
                        self.ask_jump_up_cnt += 1
                        # 旧最优价队列为负冲击
                        ask_impact = -self.prev_asks[i][1]
                        # 新最优价队列为正冲击
                        if self.asks[i][1] > 0:
                            ask_impact += self.asks[i][1]
                        self.ask_jump_up_impact_sum += ask_impact
                    elif self.asks[i][0] < self.prev_asks[i][0]:  # ask价下跌
                        self.ask_jump_down_cnt += 1
                        # 新最优价队列为正冲击
                        ask_impact = self.asks[i][1]
                        # 旧最优价队列为负冲击（如果存在）
                        if self.prev_asks[i][1] > 0:
                            ask_impact -= self.prev_asks[i][1]
                        self.ask_jump_down_impact_sum += ask_impact
                    else:  # 价格不变，用数量变化
                        ask_impact = self.asks[i][1] - self.prev_asks[i][1]
                    
                    # L1冲击：bid为正，ask为负
                    comp = self.w[i] * (bid_impact - ask_impact)
                else:
                    # 价格不变，用标准数量变化
                    delta_b = self.bids[i][1] - self.prev_bids[i][1]
                    delta_a = self.asks[i][1] - self.prev_asks[i][1]
                    comp = self.w[i] * (delta_b - delta_a)
            else:
                # 其余档位：标准数量变化
                delta_b = self.bids[i][1] - self.prev_bids[i][1]
                delta_a = self.asks[i][1] - self.prev_asks[i][1]
                comp = self.w[i] * (delta_b - delta_a)
            
            k_components.append(comp)
            ofi_val += comp

        # 计算Z-score（基于"上一窗口"，不包含当前ofi_val）
        z_ofi = None
        warmup = False
        std_zero = False
        warmup_threshold = max(5, self.z_window // 5)
        
        # 统一从历史窗口获取数据（不包含当前值）
        arr = list(self.ofi_hist)
        if len(arr) < warmup_threshold:
            warmup = True
        else:
            m, s = self._mean_std(arr)
            if s <= 1e-9:
                z_ofi = 0.0
                std_zero = True  # 标记标准差为0的情况
            else:
                z_ofi = (ofi_val - m) / s

        # 更新EMA
        if self.ema_ofi is None:
            self.ema_ofi = ofi_val
        else:
            a = self.ema_alpha
            self.ema_ofi = a * ofi_val + (1.0 - a) * self.ema_ofi
        
        # 更新OFI历史（放在Z-score计算后，确保"上一窗口"口径）
        self.ofi_hist.append(ofi_val)

        return {
            "symbol": self.symbol,
            "event_time_ms": event_time_ms,
            "ofi": ofi_val,
            "k_components": k_components,
            "z_ofi": z_ofi,
            "ema_ofi": self.ema_ofi,
            "meta": {
                "levels": self.K,
                "weights": list(self.w),
                "bad_points": self.bad_points,
                "warmup": warmup,
                "std_zero": std_zero,  # 新增：标准差为0标记
                # L1价跃迁诊断统计
                "bid_jump_up_cnt": self.bid_jump_up_cnt,
                "bid_jump_down_cnt": self.bid_jump_down_cnt,
                "ask_jump_up_cnt": self.ask_jump_up_cnt,
                "ask_jump_down_cnt": self.ask_jump_down_cnt,
                "bid_jump_up_impact_sum": self.bid_jump_up_impact_sum,
                "bid_jump_down_impact_sum": self.bid_jump_down_impact_sum,
                "ask_jump_up_impact_sum": self.ask_jump_up_impact_sum,
                "ask_jump_down_impact_sum": self.ask_jump_down_impact_sum,
            },
        }

    def update_with_l2_delta(self, deltas, event_time_ms: Optional[int] = None):
        """
        基于L2增量更新OFI（Task 1.2.1暂不实现）
        
        注意:
            Task 1.2.1仅实现快照模式，增量模式将在后续任务中实现
        
        异常:
            NotImplementedError: 此版本暂不支持增量模式
        """
        raise NotImplementedError("Task 1.2.1 implements snapshot mode only.")

