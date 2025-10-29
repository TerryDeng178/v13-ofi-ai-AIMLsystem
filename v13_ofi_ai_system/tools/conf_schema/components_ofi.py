"""
OFI组件配置Schema
"""

from pydantic import BaseModel, Field


class OFIConfig(BaseModel):
    """OFI计算器配置（锁定参数）"""
    z_window: int = Field(..., gt=0, description="Z窗口大小，锁定值：80")
    ema_alpha: float = Field(..., gt=0.0, le=1.0, description="EMA系数，锁定值：0.30")
    z_clip: float = Field(..., ge=0.0, description="Z裁剪值，锁定值：3.0")
    reset_on_gap_ms: int = Field(..., gt=0, description="间隔重置阈值（毫秒）")
    reset_on_session_change: bool = Field(..., description="会话切换重置")
    per_symbol_window: bool = Field(..., description="按交易对独立窗口")
    levels: int = Field(..., gt=0, description="订单簿档位数")
    weights: Optional[list] = Field(None, description="权重列表，None表示使用标准权重")


