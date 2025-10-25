"""
OFI+CVD融合指标模块

实现订单流不平衡(OFI)和累积成交量差值(CVD)的融合信号生成，
包含时间对齐、降级机制、去噪三件套等核心功能。

Author: V13 OFI+CVD AI Trading System
Date: 2025-10-19
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import math
import time
from enum import Enum
from pathlib import Path


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
    
    # 信号阈值 - 进攻版配置（提升交易频率）
    fuse_buy: float = 1.2          # 1.5 → 1.2 (降低买入门槛)
    fuse_strong_buy: float = 2.0   # 2.5 → 2.0 (降低强买入门槛)
    fuse_sell: float = -1.2        # -1.5 → -1.2 (降低卖出门槛)
    fuse_strong_sell: float = -2.0 # -2.5 → -2.0 (降低强卖出门槛)
    
    # 一致性阈值 - 进攻版配置
    min_consistency: float = 0.2   # 0.3 → 0.2 (降低一致性要求)
    strong_min_consistency: float = 0.6  # 0.7 → 0.6 (降低强一致性要求)
    
    # 数据处理 - 进攻版配置
    z_clip: float = 4.0            # 5.0 → 4.0 (放宽Z-score裁剪)
    max_lag: float = 0.25         # 0.800 → 0.25 (收紧时间对齐要求)
    
    # 去噪参数 - 进攻版配置（大幅提升触发率）
    hysteresis_exit: float = 0.6  # 0.8 → 0.6 (进一步减小迟滞)
    cooldown_secs: float = 0.6    # 0.3 → 0.6 (适度冷却，避免过度频繁)
    min_consecutive: int = 1      # 保持1 (最低持续门槛)
    
    # 暖启动 - 进攻版配置
    min_warmup_samples: int = 20   # 10 → 20 (平衡快速点火与数据质量)


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
    
    def __init__(self, cfg: OFICVDFusionConfig = None, config_loader=None):
        """
        初始化融合器
        
        Args:
            cfg: 融合配置，默认使用标准配置
            config_loader: 配置加载器实例，用于从统一配置系统加载参数
        """
        if config_loader:
            # 从统一配置系统加载参数
            self.cfg = self._load_from_config_loader(config_loader)
        else:
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
    
    def _load_from_config_loader(self, config_loader) -> OFICVDFusionConfig:
        """
        从统一配置系统加载融合指标参数
        
        Args:
            config_loader: 配置加载器实例
            
        Returns:
            融合指标配置对象
        """
        try:
            # 获取融合指标配置
            fusion_config = config_loader.get('fusion_metrics', {})
            
            # 提取权重配置
            weights = fusion_config.get('weights', {})
            w_ofi = weights.get('w_ofi', 0.6)
            w_cvd = weights.get('w_cvd', 0.4)
            
            # 提取阈值配置
            thresholds = fusion_config.get('thresholds', {})
            fuse_buy = thresholds.get('fuse_buy', 1.5)
            fuse_strong_buy = thresholds.get('fuse_strong_buy', 2.5)
            fuse_sell = thresholds.get('fuse_sell', -1.5)
            fuse_strong_sell = thresholds.get('fuse_strong_sell', -2.5)
            
            # 提取一致性配置（支持分场景）
            consistency = fusion_config.get('consistency', {})
            regime_consistency = consistency.get('regime_consistency', {})
            
            # 获取当前regime（从核心算法传递）
            current_regime = getattr(self, '_current_regime', 'normal')
            regime_config = regime_consistency.get(current_regime, {})
            
            if regime_config:
                # 使用分场景一致性阈值
                min_consistency = regime_config.get('min_consistency', consistency.get('min_consistency', 0.15))
                strong_min_consistency = regime_config.get('strong_min_consistency', consistency.get('strong_min_consistency', 0.4))
            else:
                # 使用基础一致性阈值
                min_consistency = consistency.get('min_consistency', 0.15)
                strong_min_consistency = consistency.get('strong_min_consistency', 0.4)
            
            # 提取数据处理配置
            data_processing = fusion_config.get('data_processing', {})
            z_clip = data_processing.get('z_clip', 5.0)
            max_lag = data_processing.get('max_lag', 0.300)
            warmup_samples = data_processing.get('warmup_samples', 30)
            
            # 提取去噪配置
            denoising = fusion_config.get('denoising', {})
            hysteresis_exit = denoising.get('hysteresis_exit', 1.2)
            cooldown_secs = denoising.get('cooldown_secs', 1.0)
            min_duration = denoising.get('min_duration', 2)
            
            # 创建配置对象
            return OFICVDFusionConfig(
                w_ofi=w_ofi,
                w_cvd=w_cvd,
                fuse_buy=fuse_buy,
                fuse_strong_buy=fuse_strong_buy,
                fuse_sell=fuse_sell,
                fuse_strong_sell=fuse_strong_sell,
                min_consistency=min_consistency,
                strong_min_consistency=strong_min_consistency,
                z_clip=z_clip,
                max_lag=max_lag,
                hysteresis_exit=hysteresis_exit,
                cooldown_secs=cooldown_secs,
                min_consecutive=min_duration,
                min_warmup_samples=warmup_samples
            )
            
        except Exception as e:
            # 如果配置加载失败，使用默认配置并记录警告
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load fusion metrics config from config_loader: {e}. Using default config.")
            return OFICVDFusionConfig()
    
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
                        ts: float, consistency: float = 0.0) -> tuple[SignalType, list]:
        """
        应用去噪三件套 - 增强版
        
        Args:
            signal: 原始信号
            fusion_score: 融合得分
            ts: 时间戳
            consistency: 一致性得分
            
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
        
        # 2. 一致性加权迟滞处理 - 增强版
        consistency_bonus = max(0.0, consistency - 0.3) * 0.5  # 一致性加分
        adjusted_hysteresis = self.cfg.hysteresis_exit + consistency_bonus
        
        if (self._last_signal == SignalType.STRONG_BUY and 
            signal == SignalType.BUY and 
            fusion_score > adjusted_hysteresis):
            # 从强买入降级到买入时，如果得分仍然很高，保持强买入
            denoising_reasons.append("hysteresis_hold")
            signal = SignalType.STRONG_BUY
        elif (self._last_signal == SignalType.STRONG_SELL and 
              signal == SignalType.SELL and 
              fusion_score < -adjusted_hysteresis):
            # 从强卖出降级到卖出时，如果得分仍然很低，保持强卖出
            denoising_reasons.append("hysteresis_hold")
            signal = SignalType.STRONG_SELL
        elif (self._last_signal in [SignalType.BUY, SignalType.STRONG_BUY] and 
              signal == SignalType.NEUTRAL and 
              fusion_score > adjusted_hysteresis):
            # 从买入信号变为中性时，如果得分仍然较高，保持买入
            denoising_reasons.append("hysteresis_hold")
            signal = self._last_signal
        elif (self._last_signal in [SignalType.SELL, SignalType.STRONG_SELL] and 
              signal == SignalType.NEUTRAL and 
              fusion_score < -adjusted_hysteresis):
            # 从卖出信号变为中性时，如果得分仍然较低，保持卖出
            denoising_reasons.append("hysteresis_hold")
            signal = self._last_signal
        
        # 3. 一致性驱动的信号强度调整
        if consistency > 0.7:  # 高一致性时放宽阈值
            if signal == SignalType.BUY and fusion_score > self.cfg.fuse_buy * 0.8:
                signal = SignalType.BUY
                denoising_reasons.append("consistency_boost")
            elif signal == SignalType.SELL and fusion_score < self.cfg.fuse_sell * 0.8:
                signal = SignalType.SELL
                denoising_reasons.append("consistency_boost")
        elif consistency < 0.3:  # 低一致性时严格节流
            if signal in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.SELL, SignalType.STRONG_SELL]:
                signal = SignalType.NEUTRAL
                denoising_reasons.append("low_consistency_throttle")
        
        # 4. 最小持续检查 - 场景自适应版本
        if signal != SignalType.NEUTRAL and self._last_signal == signal:
            self._streak += 1
        else:
            self._streak = 0
        
        # 动态调整min_consecutive：活跃场景和高一致性时放宽
        current_regime = getattr(self, '_current_regime', 'normal')
        effective_min_consecutive = self.cfg.min_consecutive
        
        # 活跃场景(A_H/A_L)放宽要求
        if current_regime in ['A_H', 'A_L']:
            effective_min_consecutive = 0  # 活跃场景立即触发
        # 高一致性时放宽要求
        elif consistency > 0.7:
            effective_min_consecutive = max(0, self.cfg.min_consecutive - 1)
            
        if signal != SignalType.NEUTRAL and self._streak < effective_min_consecutive:
            signal = SignalType.NEUTRAL
            denoising_reasons.append("min_duration")
        
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
        
        # 7. 去噪处理 - 传递一致性参数
        signal, denoising_reasons = self._apply_denoising(signal, fusion_score, ts, consistency)
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

# 为OFI_CVD_Fusion类添加set_thresholds方法
def set_thresholds(self, **kwargs):
    """
    统一入口设置阈值参数
    
    Args:
        **kwargs: 阈值参数
    """
    # 仅允许改 cfg 上已有字段
    for k, v in kwargs.items():
        if v is not None and hasattr(self.cfg, k):
            setattr(self.cfg, k, v)

    # 如果权重被修改，重新归一化
    tw = self.cfg.w_ofi + self.cfg.w_cvd
    if tw <= 0:
        raise ValueError("权重和必须大于0")
    self.w_ofi = self.cfg.w_ofi / tw
    self.w_cvd = self.cfg.w_cvd / tw

# 动态添加方法到类
OFI_CVD_Fusion.set_thresholds = set_thresholds
