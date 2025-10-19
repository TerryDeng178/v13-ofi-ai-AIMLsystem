"""
OFI+CVD融合指标模块

实现订单流不平衡(OFI)和累积成交量差值(CVD)的融合信号生成，
包含时间对齐、降级机制、去噪三件套等核心功能。

Author: V13 OFI+CVD AI Trading System
Date: 2025-10-19
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import math
import time
from enum import Enum


class SignalType(Enum):
    """信号类型枚举"""
    NEUTRAL = "neutral"
    BUY = "buy"
    STRONG_BUY = "strong_buy"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class OFICVDFusionConfig:
    """OFI+CVD融合配置"""
    # 权重配置
    w_ofi: float = 0.6
    w_cvd: float = 0.4
    
    # 信号阈值
    fuse_buy: float = 1.5
    fuse_strong_buy: float = 2.5
    fuse_sell: float = -1.5
    fuse_strong_sell: float = -2.5
    
    # 一致性阈值
    min_consistency: float = 0.3
    strong_min_consistency: float = 0.7
    
    # 数据处理
    z_clip: float = 5.0
    max_lag: float = 0.300  # 最大时间差(秒)
    
    # 去噪参数
    hysteresis_exit: float = 1.2
    cooldown_secs: float = 1.0
    min_consecutive: int = 2
    
    # 暖启动
    min_warmup_samples: int = 30


class OFI_CVD_Fusion:
    """
    OFI+CVD融合信号生成器
    
    核心功能:
    1. 权重融合: w_ofi * z_ofi + w_cvd * z_cvd
    2. 时间对齐: 处理OFI/CVD不同步问题
    3. 降级机制: 数据缺失时降级为单因子
    4. 去噪三件套: 迟滞/冷却/最小持续
    5. 暖启动保护: 数据不足时返回neutral
    """
    
    def __init__(self, cfg: OFICVDFusionConfig = None):
        """
        初始化融合器
        
        Args:
            cfg: 融合配置，默认使用标准配置
        """
        self.cfg = cfg or OFICVDFusionConfig()
        
        # 权重归一化
        total_weight = self.cfg.w_ofi + self.cfg.w_cvd
        if total_weight <= 0:
            raise ValueError("权重和必须大于0")
        self.w_ofi = self.cfg.w_ofi / total_weight
        self.w_cvd = self.cfg.w_cvd / total_weight
        
        # 状态管理
        self._last_signal = SignalType.NEUTRAL
        self._last_emit_ts: Optional[float] = None
        self._streak = 0
        self._warmup_count = 0
        self._is_warmup = True
        
        # 统计信息
        self._stats = {
            'total_updates': 0,
            'downgrades': 0,
            'warmup_returns': 0,
            'invalid_inputs': 0,
            'lag_exceeded': 0
        }
    
    def _consistency(self, z_ofi: float, z_cvd: float) -> float:
        """
        计算信号一致性
        
        Args:
            z_ofi: OFI Z-score
            z_cvd: CVD Z-score
            
        Returns:
            一致性得分 (0-1)
        """
        if z_ofi == 0 or z_cvd == 0:
            return 0.0
        
        # 方向一致性检查
        if math.copysign(1, z_ofi) != math.copysign(1, z_cvd):
            return 0.0
        
        # 强度一致性 (较小值/较大值)
        abs_ofi, abs_cvd = abs(z_ofi), abs(z_cvd)
        return min(abs_ofi, abs_cvd) / max(abs_ofi, abs_cvd)
    
    def _clip_z_score(self, z: float) -> float:
        """裁剪Z-score到合理范围"""
        return max(-self.cfg.z_clip, min(self.cfg.z_clip, z))
    
    def _check_warmup(self) -> bool:
        """检查是否还在暖启动期"""
        if self._warmup_count < self.cfg.min_warmup_samples:
            self._warmup_count += 1
            return True
        self._is_warmup = False
        return False
    
    def _apply_denoising(self, signal: SignalType, fusion_score: float, 
                        ts: float) -> tuple[SignalType, list]:
        """
        应用去噪三件套
        
        Args:
            signal: 原始信号
            fusion_score: 融合得分
            ts: 时间戳
            
        Returns:
            (去噪后的信号, 去噪原因列表)
        """
        denoising_reasons = []
        original_signal = signal
        
        # 1. 冷却时间检查
        if (self._last_emit_ts and 
            (ts - self._last_emit_ts) < self.cfg.cooldown_secs and
            signal != SignalType.NEUTRAL):
            denoising_reasons.append("cooldown")
            return SignalType.NEUTRAL, denoising_reasons
        
        # 2. 迟滞处理 - 只在信号强度下降时应用
        if (self._last_signal == SignalType.STRONG_BUY and 
            signal == SignalType.BUY and 
            fusion_score > self.cfg.hysteresis_exit):
            # 从强买入降级到买入时，如果得分仍然很高，保持强买入
            denoising_reasons.append("hysteresis_hold")
            signal = SignalType.STRONG_BUY
        elif (self._last_signal == SignalType.STRONG_SELL and 
              signal == SignalType.SELL and 
              fusion_score < -self.cfg.hysteresis_exit):
            # 从强卖出降级到卖出时，如果得分仍然很低，保持强卖出
            denoising_reasons.append("hysteresis_hold")
            signal = SignalType.STRONG_SELL
        elif (self._last_signal in [SignalType.BUY, SignalType.STRONG_BUY] and 
              signal == SignalType.NEUTRAL and 
              fusion_score > self.cfg.hysteresis_exit):
            # 从买入信号变为中性时，如果得分仍然较高，保持买入
            denoising_reasons.append("hysteresis_hold")
            signal = self._last_signal
        elif (self._last_signal in [SignalType.SELL, SignalType.STRONG_SELL] and 
              signal == SignalType.NEUTRAL and 
              fusion_score < -self.cfg.hysteresis_exit):
            # 从卖出信号变为中性时，如果得分仍然较低，保持卖出
            denoising_reasons.append("hysteresis_hold")
            signal = self._last_signal
        
        # 3. 最小持续检查 - 暂时禁用，先确保基本信号生成正常
        # TODO: 重新设计最小持续逻辑
        
        return signal, denoising_reasons
    
    def update(self, z_ofi: float, z_cvd: float, ts: float,
               price: Optional[float] = None, lag_sec: float = 0.0) -> Dict[str, Any]:
        """
        更新融合信号
        
        Args:
            z_ofi: OFI Z-score
            z_cvd: CVD Z-score
            ts: 事件时间戳
            price: 可选价格信息
            lag_sec: OFI/CVD时间差(秒)
            
        Returns:
            融合结果字典
        """
        self._stats['total_updates'] += 1
        reason_codes = []
        
        # 1. 输入验证
        if any(x is None or math.isinf(x) or math.isnan(x) 
               for x in [z_ofi, z_cvd]):
            self._stats['invalid_inputs'] += 1
            return {
                "fusion_score": 0.0,
                "signal": SignalType.NEUTRAL.value,
                "consistency": 0.0,
                "ofi_weight": self.w_ofi,  # 修复：使用实际权重而非0
                "cvd_weight": self.w_cvd,  # 修复：使用实际权重而非0
                "reason_codes": ["invalid_input"],
                "components": {"ofi": 0.0, "cvd": 0.0},
                "warmup": self._is_warmup,  # 修复：使用真实暖启动状态
                "stats": self._stats.copy()  # 修复：添加stats字段
            }
        
        # 2. 暖启动检查
        if self._check_warmup():
            self._stats['warmup_returns'] += 1
            return {
                "fusion_score": 0.0,
                "signal": SignalType.NEUTRAL.value,
                "consistency": 0.0,
                "ofi_weight": self.w_ofi,
                "cvd_weight": self.w_cvd,
                "reason_codes": ["warmup"],
                "components": {"ofi": 0.0, "cvd": 0.0},
                "warmup": True,
                "stats": self._stats.copy()  # 修复：添加stats字段
            }
        
        # 3. 数据裁剪
        z_ofi_clipped = self._clip_z_score(z_ofi)
        z_cvd_clipped = self._clip_z_score(z_cvd)
        
        # 4. 时间对齐检查与单因子降级
        w_ofi, w_cvd = self.w_ofi, self.w_cvd
        if lag_sec > self.cfg.max_lag:
            self._stats['lag_exceeded'] += 1
            self._stats['downgrades'] += 1  # 添加降级统计
            reason_codes.append("lag_exceeded")
            # 实现单因子降级：保留更"强"的一侧
            if abs(z_ofi_clipped) >= abs(z_cvd_clipped):
                w_ofi, w_cvd = 1.0, 0.0
                reason_codes.append("degraded_ofi_only")
            else:
                w_ofi, w_cvd = 0.0, 1.0
                reason_codes.append("degraded_cvd_only")
        
        # 5. 融合计算
        fusion_score = w_ofi * z_ofi_clipped + w_cvd * z_cvd_clipped
        consistency = self._consistency(z_ofi_clipped, z_cvd_clipped)
        
        # 6. 信号生成
        signal = SignalType.NEUTRAL
        if (fusion_score > self.cfg.fuse_strong_buy and 
            consistency > self.cfg.strong_min_consistency):
            signal = SignalType.STRONG_BUY
        elif (fusion_score > self.cfg.fuse_buy and 
              consistency > self.cfg.min_consistency):
            signal = SignalType.BUY
        elif (fusion_score < self.cfg.fuse_strong_sell and 
              consistency > self.cfg.strong_min_consistency):
            signal = SignalType.STRONG_SELL
        elif (fusion_score < self.cfg.fuse_sell and 
              consistency > self.cfg.min_consistency):
            signal = SignalType.SELL
        
        # 7. 去噪处理
        signal, denoising_reasons = self._apply_denoising(signal, fusion_score, ts)
        reason_codes.extend(denoising_reasons)
        
        # 8. 状态更新
        if signal != SignalType.NEUTRAL:
            self._last_emit_ts = ts
        self._last_signal = signal if signal != SignalType.NEUTRAL else self._last_signal
        
        return {
            "fusion_score": fusion_score,
            "signal": signal.value,
            "consistency": consistency,
            "ofi_weight": w_ofi,
            "cvd_weight": w_cvd,
            "reason_codes": reason_codes,
            "components": {
                "ofi": w_ofi * z_ofi_clipped, 
                "cvd": w_cvd * z_cvd_clipped
            },
            "warmup": False,
            "stats": self._stats.copy()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()
    
    def reset(self):
        """重置状态"""
        self._last_signal = SignalType.NEUTRAL
        self._last_emit_ts = None
        self._streak = 0
        self._warmup_count = 0
        self._is_warmup = True
        self._stats = {
            'total_updates': 0,
            'downgrades': 0,
            'warmup_returns': 0,
            'invalid_inputs': 0,
            'lag_exceeded': 0
        }


def create_fusion_config(**kwargs) -> OFICVDFusionConfig:
    """
    创建融合配置的便捷函数
    
    Args:
        **kwargs: 配置参数
        
    Returns:
        配置对象
    """
    return OFICVDFusionConfig(**kwargs)


# 默认配置实例
DEFAULT_CONFIG = OFICVDFusionConfig()
