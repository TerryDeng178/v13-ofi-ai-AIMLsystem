#!/usr/bin/env python3
"""
V10.0 增强OFI计算器 - 3级加权OFI + 深度学习集成
结合V10深度学习功能，实现实时3级加权OFI计算
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
from typing import Dict, List, Optional, Tuple
import time
import json

class V10EnhancedOFI:
    """
    V10.0 增强OFI计算器
    支持3级加权OFI计算和深度学习信号生成
    """
    
    def __init__(self, micro_window_ms: int = 100, z_window_seconds: int = 900, 
                 levels: int = 3, weights: List[float] = None):
        self.w = micro_window_ms
        self.zn = int(max(10, z_window_seconds * 1000 // self.w))
        self.levels = levels
        self.weights = weights or [0.5, 0.3, 0.2]  # 3级权重：第1级50%，第2级30%，第3级20%
        
        # 历史数据存储
        self.cur_bucket = None
        self.bucket_sum = 0.0
        self.history = deque(maxlen=self.zn)
        self.t_series = deque(maxlen=self.zn)
        self.last_best = None
        
        # 3级OFI计算
        self.level_contributions = [0.0] * self.levels
        self.level_history = [deque(maxlen=self.zn) for _ in range(self.levels)]
        
        # V10深度学习模型
        self.dl_model = None
        self.feature_buffer = deque(maxlen=60)  # 60个时间步的特征
        self.signal_history = deque(maxlen=1000)
        
        # 实时优化参数
        self.adaptive_thresholds = True
        self.performance_window = 50
        self.adaptation_rate = 0.1
        
    def on_best(self, t: int, bid: float, bid_sz: float, ask: float, ask_sz: float):
        """处理最优买卖价更新"""
        self.last_best = (t, bid, bid_sz, ask, ask_sz)
        
    def on_l2(self, t: int, typ: str, side: str, price: float, qty: float):
        """处理L2订单簿更新"""
        if not self.last_best:
            return
            
        _, bid, bid_sz, ask, ask_sz = self.last_best
        is_add = (typ == "l2_add")
        
        # 计算3级加权OFI贡献
        contributions = self._calculate_level_contributions(
            is_add, side, price, qty, bid, bid_sz, ask, ask_sz
        )
        
        # 更新各级贡献
        for i, contrib in enumerate(contributions):
            self.level_contributions[i] += contrib
            
        # 计算加权OFI
        weighted_ofi = sum(w * c for w, c in zip(self.weights, self.level_contributions))
        
        # 更新桶数据
        bucket = (t // self.w) * self.w
        if self.cur_bucket is None:
            self.cur_bucket = bucket
            
        if bucket != self.cur_bucket:
            # 保存历史数据
            self.history.append(self.bucket_sum)
            self.t_series.append(self.cur_bucket)
            
            # 保存各级历史数据
            for i, contrib in enumerate(self.level_contributions):
                self.level_history[i].append(contrib)
                
            # 重置当前桶
            self.bucket_sum = weighted_ofi
            self.level_contributions = [0.0] * self.levels
            self.cur_bucket = bucket
        else:
            self.bucket_sum = weighted_ofi
            
    def _calculate_level_contributions(self, is_add: bool, side: str, price: float, 
                                     qty: float, bid: float, bid_sz: float, 
                                     ask: float, ask_sz: float) -> List[float]:
        """计算3级OFI贡献"""
        contributions = [0.0] * self.levels
        
        if not self.last_best:
            return contributions
            
        # 第1级：最优买卖价
        is_bid1 = abs(price - bid) < 1e-9
        is_ask1 = abs(price - ask) < 1e-9
        
        if is_add and is_bid1:
            contributions[0] += qty
        if is_add and is_ask1:
            contributions[0] -= qty
        if (not is_add) and is_bid1:
            contributions[0] -= qty
        if (not is_add) and is_ask1:
            contributions[0] += qty
            
        # 第2级：次优买卖价（简化计算）
        if is_add and side == 'bid' and not is_bid1:
            contributions[1] += qty * 0.5
        if is_add and side == 'ask' and not is_ask1:
            contributions[1] -= qty * 0.5
        if (not is_add) and side == 'bid' and not is_bid1:
            contributions[1] -= qty * 0.5
        if (not is_add) and side == 'ask' and not is_ask1:
            contributions[1] += qty * 0.5
            
        # 第3级：更深层级（简化计算）
        if is_add and side == 'bid':
            contributions[2] += qty * 0.3
        if is_add and side == 'ask':
            contributions[2] -= qty * 0.3
        if (not is_add) and side == 'bid':
            contributions[2] -= qty * 0.3
        if (not is_add) and side == 'ask':
            contributions[2] += qty * 0.3
            
        return contributions
        
    def read(self) -> Optional[Dict]:
        """读取当前OFI值"""
        if len(self.history) < max(10, self.zn // 10):
            return None
            
        arr = np.array(self.history, dtype=float)
        z = (arr[-1] - arr.mean()) / (arr.std(ddof=0) + 1e-9)
        
        # 计算各级OFI
        level_ofis = []
        level_zs = []
        for i in range(self.levels):
            if len(self.level_history[i]) > 0:
                level_arr = np.array(self.level_history[i], dtype=float)
                level_ofi = level_arr[-1] if len(level_arr) > 0 else 0.0
                level_z = (level_ofi - level_arr.mean()) / (level_arr.std(ddof=0) + 1e-9)
                level_ofis.append(level_ofi)
                level_zs.append(level_z)
            else:
                level_ofis.append(0.0)
                level_zs.append(0.0)
        
        # 计算加权OFI
        weighted_ofi = sum(w * ofi for w, ofi in zip(self.weights, level_ofis))
        weighted_z = sum(w * z for w, z in zip(self.weights, level_zs))
        
        return {
            "t": self.t_series[-1],
            "ofi": float(arr[-1]),
            "ofi_z": float(z),
            "weighted_ofi": float(weighted_ofi),
            "weighted_ofi_z": float(weighted_z),
            "level_ofis": level_ofis,
            "level_zs": level_zs,
            "weights": self.weights
        }
        
    def create_features(self, ofi_data: Dict, market_data: Dict) -> np.ndarray:
        """创建深度学习特征"""
        features = []
        
        # OFI特征
        features.extend([
            ofi_data.get("ofi", 0.0),
            ofi_data.get("ofi_z", 0.0),
            ofi_data.get("weighted_ofi", 0.0),
            ofi_data.get("weighted_ofi_z", 0.0)
        ])
        
        # 各级OFI特征
        level_ofis = ofi_data.get("level_ofis", [0.0] * self.levels)
        level_zs = ofi_data.get("level_zs", [0.0] * self.levels)
        features.extend(level_ofis)
        features.extend(level_zs)
        
        # 市场数据特征
        if market_data:
            features.extend([
                market_data.get("bid", 0.0),
                market_data.get("ask", 0.0),
                market_data.get("bid_sz", 0.0),
                market_data.get("ask_sz", 0.0),
                market_data.get("spread", 0.0),
                market_data.get("mid_price", 0.0)
            ])
        else:
            features.extend([0.0] * 6)
            
        # 时间特征
        current_time = time.time()
        features.extend([
            current_time % 86400,  # 一天中的秒数
            current_time % 3600,   # 一小时中的秒数
            current_time % 60      # 一分钟中的秒数
        ])
        
        return np.array(features, dtype=np.float32)
        
    def predict_signal(self, features: np.ndarray) -> Dict:
        """使用深度学习模型预测信号"""
        if self.dl_model is None:
            # 如果没有模型，使用简单的规则
            ofi_z = features[1] if len(features) > 1 else 0.0
            weighted_ofi_z = features[3] if len(features) > 3 else 0.0
            
            signal_strength = abs(weighted_ofi_z)
            signal_side = 1 if weighted_ofi_z > 2.0 else -1 if weighted_ofi_z < -2.0 else 0
            
            return {
                "signal_side": signal_side,
                "signal_strength": signal_strength,
                "confidence": min(1.0, signal_strength / 3.0),
                "model_type": "rule_based"
            }
        
        # 使用深度学习模型预测
        try:
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0)
                prediction = self.dl_model(input_tensor)
                
                signal_strength = float(prediction[0][0])
                signal_side = 1 if signal_strength > 0.5 else -1 if signal_strength < -0.5 else 0
                confidence = min(1.0, abs(signal_strength))
                
                return {
                    "signal_side": signal_side,
                    "signal_strength": signal_strength,
                    "confidence": confidence,
                    "model_type": "deep_learning"
                }
        except Exception as e:
            print(f"深度学习预测失败: {e}")
            return {
                "signal_side": 0,
                "signal_strength": 0.0,
                "confidence": 0.0,
                "model_type": "error"
            }
            
    def update_performance(self, signal_result: Dict, actual_return: float):
        """更新性能指标"""
        if signal_result["signal_side"] != 0:
            self.signal_history.append({
                "signal_side": signal_result["signal_side"],
                "signal_strength": signal_result["signal_strength"],
                "confidence": signal_result["confidence"],
                "actual_return": actual_return,
                "timestamp": time.time()
            })
            
        # 自适应调整阈值
        if self.adaptive_thresholds and len(self.signal_history) >= self.performance_window:
            self._adapt_thresholds()
            
    def _adapt_thresholds(self):
        """自适应调整阈值"""
        if len(self.signal_history) < self.performance_window:
            return
            
        recent_signals = list(self.signal_history)[-self.performance_window:]
        
        # 计算胜率
        correct_signals = sum(1 for s in recent_signals 
                            if (s["signal_side"] > 0 and s["actual_return"] > 0) or 
                               (s["signal_side"] < 0 and s["actual_return"] < 0))
        win_rate = correct_signals / len(recent_signals) if recent_signals else 0.0
        
        # 根据胜率调整权重
        if win_rate < 0.4:  # 胜率过低，降低权重
            self.weights = [w * (1 - self.adaptation_rate) for w in self.weights]
        elif win_rate > 0.6:  # 胜率过高，提高权重
            self.weights = [w * (1 + self.adaptation_rate) for w in self.weights]
            
        # 归一化权重
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
            
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.signal_history:
            return {"total_signals": 0, "win_rate": 0.0, "avg_confidence": 0.0}
            
        total_signals = len(self.signal_history)
        correct_signals = sum(1 for s in self.signal_history 
                            if (s["signal_side"] > 0 and s["actual_return"] > 0) or 
                               (s["signal_side"] < 0 and s["actual_return"] < 0))
        win_rate = correct_signals / total_signals if total_signals > 0 else 0.0
        avg_confidence = np.mean([s["confidence"] for s in self.signal_history])
        
        return {
            "total_signals": total_signals,
            "win_rate": win_rate,
            "avg_confidence": avg_confidence,
            "current_weights": self.weights
        }
