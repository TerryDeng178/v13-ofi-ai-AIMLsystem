"""
Runtime运行时配置Schema
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class LoggingSink(BaseModel):
    """日志输出配置"""
    type: str = Field(..., description="输出类型：jsonl/sqlite/null")
    enabled: bool = Field(..., description="是否启用")
    rotation_minutes: Optional[int] = Field(None, gt=0, description="轮转间隔（分钟）")
    fsync_ms: Optional[int] = Field(None, ge=0, description="同步间隔（毫秒）")
    batch_size: Optional[int] = Field(None, gt=0, description="批处理大小")


class OutputConfig(BaseModel):
    """输出配置"""
    sinks: List[LoggingSink] = Field(..., description="输出列表")
    weak_signal_threshold: float = Field(..., ge=0.0, le=1.0, description="弱信号阈值")


class GuardsConfig(BaseModel):
    """护栏配置"""
    spread_bps_cap: float = Field(..., gt=0.0, description="点差上限（bps）")
    max_missing_msgs_rate: float = Field(..., ge=0.0, le=1.0, description="最大消息丢失率")
    max_event_lag_sec: float = Field(..., ge=0.0, description="最大事件滞后（秒）")
    exit_cooldown_sec: float = Field(..., ge=0.0, description="退出冷却时间（秒）")
    reconnect_cooldown_sec: float = Field(..., ge=0.0, description="重连冷却时间（秒）")
    resync_cooldown_sec: float = Field(..., ge=0.0, description="重同步冷却时间（秒）")
    reverse_prevention_sec: float = Field(..., ge=0.0, description="反转预防时间（秒）")
    warmup_period_sec: float = Field(..., ge=0.0, description="暖启动周期（秒）")
    
    @property
    def cooldown_hierarchy_valid(self) -> bool:
        """验证冷却时间层级关系"""
        return (self.reconnect_cooldown_sec >= self.exit_cooldown_sec and
                self.resync_cooldown_sec >= self.reconnect_cooldown_sec)


class PerformanceConfig(BaseModel):
    """性能配置"""
    max_queue_size: int = Field(..., gt=0, description="最大队列大小")
    batch_size: int = Field(..., gt=0, description="批处理大小")
    flush_interval_ms: int = Field(..., ge=0, description="刷新间隔（毫秒）")


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = Field(..., pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", description="日志级别")
    debug: bool = Field(..., description="是否调试模式")
    heartbeat_interval_sec: int = Field(..., gt=0, description="心跳间隔（秒）")


class RuntimeConfig(BaseModel):
    """运行时配置"""
    logging: LoggingConfig
    performance: PerformanceConfig
    guards: Optional[GuardsConfig] = None
    output: Optional[OutputConfig] = None

