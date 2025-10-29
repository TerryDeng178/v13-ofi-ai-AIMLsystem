"""
CVD组件配置Schema
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict


class CVDConfig(BaseModel):
    """CVD计算器配置"""
    z_mode: str = Field(..., description="Z模式：delta或level")
    z_window: int = Field(..., gt=0, description="Z窗口大小")
    half_life_trades: int = Field(..., gt=0, description="Delta-Z半衰期（笔数）")
    winsor_limit: float = Field(..., gt=0.0, description="Winsor限制")
    freeze_min: int = Field(..., gt=0, description="冻结最小样本数")
    stale_threshold_ms: int = Field(..., gt=0, description="Stale冻结阈值（毫秒）")
    scale_mode: str = Field(..., description="尺度模式：hybrid或single")
    ewma_fast_hl: int = Field(..., gt=0, description="快EWMA半衰期")
    mad_window_trades: int = Field(..., gt=0, description="MAD窗口大小")
    mad_scale_factor: float = Field(..., gt=0.0, description="MAD还原系数")
    scale_fast_weight: float = Field(..., ge=0.0, le=1.0, description="快EWMA权重")
    scale_slow_weight: float = Field(..., ge=0.0, le=1.0, description="慢EWMA权重")
    mad_multiplier: float = Field(..., gt=0.0, description="MAD倍数")
    post_stale_freeze: int = Field(..., ge=0, description="空窗后冻结笔数")
    ema_alpha: float = Field(..., gt=0.0, le=1.0, description="EMA系数")
    use_tick_rule: bool = Field(..., description="使用Tick规则")
    warmup_min: int = Field(..., ge=0, description="暖启动最小样本数")
    
    # 分品种覆盖（可选）
    symbol_overrides: Optional[Dict[str, Dict[str, float]]] = Field(
        None, 
        description="分品种配置覆盖，例如 {'BTCUSDT': {'winsor_limit': 1.9}}"
    )


