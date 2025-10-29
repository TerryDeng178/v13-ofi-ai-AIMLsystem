"""
Strategy组件配置Schema
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class StrategyHysteresis(BaseModel):
    """策略迟滞配置"""
    window_secs: int = Field(..., gt=0, description="窗口大小（秒）")
    min_active_windows: int = Field(..., gt=0, description="最小active确认窗口数")
    min_quiet_windows: int = Field(..., gt=0, description="最小quiet确认窗口数")


class StrategySchedule(BaseModel):
    """策略调度配置"""
    enabled: bool = Field(..., description="是否启用")
    timezone: str = Field(..., description="时区")
    calendar: str = Field(..., description="交易日历")
    enabled_weekdays: List[str] = Field(..., description="启用的工作日")
    holidays: List[str] = Field(default_factory=list, description="节假日列表")
    active_windows: List[Dict[str, Any]] = Field(default_factory=list, description="活跃窗口")
    wrap_midnight: bool = Field(..., description="跨午夜包装")


class StrategyMarket(BaseModel):
    """市场触发配置"""
    enabled: bool = Field(..., description="是否启用")
    window_secs: int = Field(..., gt=0, description="窗口大小（秒）")
    min_trades_per_min: float = Field(..., ge=0.0, description="每分钟最小交易数")
    min_quote_updates_per_sec: float = Field(..., ge=0.0, description="每秒最小报价更新数")
    max_spread_bps: float = Field(..., gt=0.0, description="最大点差（bps）")
    min_volatility_bps: float = Field(..., ge=0.0, description="最小波动率（bps）")
    min_volume_usd: float = Field(..., ge=0.0, description="最小交易量（USD）")
    use_median: bool = Field(..., description="使用中位数")
    winsorize_percentile: int = Field(..., ge=1, le=100, description="Winsorize百分位")


class StrategyTriggers(BaseModel):
    """策略触发配置"""
    schedule: StrategySchedule
    market: StrategyMarket


class StrategyConfig(BaseModel):
    """策略模式配置"""
    mode: str = Field(..., pattern="^(auto|active|quiet)$", description="模式：auto/active/quiet")
    hysteresis: StrategyHysteresis
    triggers: StrategyTriggers
    scenarios_file: Optional[str] = Field(None, description="场景参数文件路径")

