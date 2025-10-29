"""
Divergence组件配置Schema
"""

from pydantic import BaseModel, Field
from typing import Optional


class DivergenceConfig(BaseModel):
    """背离检测配置"""
    min_strength: float = Field(..., ge=0.0, le=1.0, description="最小背离强度")
    min_separation_secs: float = Field(..., gt=0.0, description="最小枢轴间距（秒）")
    count_conflict_only_when_fusion_ge: float = Field(..., description="冲突计数条件")
    lookback_periods: int = Field(..., gt=0, description="回看周期数")
    swing_L: int = Field(..., gt=0, description="枢轴检测窗口长度")
    ema_k: int = Field(..., gt=0, description="EMA平滑参数")
    z_hi: float = Field(..., gt=0.0, description="高强度阈值")
    z_mid: float = Field(..., gt=0.0, description="中等强度阈值")
    min_separation: int = Field(..., gt=0, description="最小枢轴间距")
    cooldown_secs: float = Field(..., ge=0.0, description="冷却时间（秒）")
    warmup_min: int = Field(..., ge=0, description="暖启动最小样本数")
    max_lag: float = Field(..., ge=0.0, description="最大滞后时间（秒）")
    use_fusion: bool = Field(..., description="是否使用融合指标")
    cons_min: float = Field(..., ge=0.0, le=1.0, description="最小一致性阈值")

