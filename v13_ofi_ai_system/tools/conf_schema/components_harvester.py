"""
Harvester组件配置Schema（Pydantic模型）
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HarvesterPathsConfig(BaseModel):
    """路径配置"""
    output_dir: str = Field(default="./data/ofi_cvd", description="输出目录（权威库）")
    preview_dir: str = Field(default="./preview/ofi_cvd", description="预览库目录")
    artifacts_dir: str = Field(default="./artifacts", description="产物目录")


class HarvesterBuffersConfig(BaseModel):
    """缓冲区配置"""
    high: Dict[str, int] = Field(
        default={
            "prices": 20000,
            "orderbook": 12000,
            "ofi": 8000,
            "cvd": 8000,
            "fusion": 5000,
            "events": 5000,
            "features": 8000
        },
        description="高水位触发落盘"
    )
    emergency: Dict[str, int] = Field(
        default={
            "prices": 40000,
            "orderbook": 24000,
            "ofi": 16000,
            "cvd": 16000,
            "fusion": 10000,
            "events": 10000,
            "features": 16000
        },
        description="紧急水位触发溢写"
    )


class HarvesterFilesConfig(BaseModel):
    """文件配置"""
    max_rows_per_file: int = Field(default=50000, ge=1000, description="单文件最大行数")
    parquet_rotate_sec: int = Field(default=60, ge=10, description="Parquet轮转间隔（秒）")


class HarvesterConcurrencyConfig(BaseModel):
    """并发配置"""
    save_concurrency: int = Field(default=2, ge=1, le=10, description="保存并发度")


class HarvesterTimeoutsConfig(BaseModel):
    """超时配置"""
    stream_idle_sec: int = Field(default=120, ge=60, description="流空闲超时（秒）")
    trade_timeout: int = Field(default=150, ge=60, description="交易流超时（秒）")
    orderbook_timeout: int = Field(default=180, ge=60, description="订单簿流超时（秒）")
    health_check_interval: int = Field(default=25, ge=1, description="健康检查间隔（秒），≥1避免整除风险")
    backoff_reset_secs: int = Field(default=300, ge=60, description="退避复位阈值（秒）")


class HarvesterHealthConfig(BaseModel):
    """健康监控配置"""
    data_timeout: int = Field(default=300, ge=60, description="数据超时（秒）")
    max_connection_errors: int = Field(default=10, ge=1, le=100, description="最大连接错误数")


class HarvesterThresholdsConfig(BaseModel):
    """阈值配置"""
    extreme_traffic_threshold: int = Field(default=30000, ge=10000, description="极端流量阈值")
    extreme_rotate_sec: int = Field(default=30, ge=10, description="极端流量轮转间隔（秒）")
    ofi_max_lag_ms: int = Field(default=800, ge=100, description="OFI最大滞后（毫秒）")


class HarvesterDedupConfig(BaseModel):
    """去重配置"""
    lru_size: int = Field(default=32768, ge=1024, description="去重LRU大小")
    queue_drop_threshold: int = Field(default=1000, ge=100, description="队列丢弃告警阈值")


class HarvesterScenarioConfig(BaseModel):
    """场景配置"""
    win_secs: int = Field(default=300, ge=60, description="窗口大小（秒）")
    active_tps: float = Field(default=0.1, ge=0.01, description="Active阈值（TPS）")
    vol_split: float = Field(default=0.5, ge=0.1, le=0.9, description="波动分割点")
    fee_tier: str = Field(default="TM", description="手续费档位")


class HarvesterTuningConfig(BaseModel):
    """运行期工况常量配置"""
    orderbook_buf_len: int = Field(default=1024, ge=256, description="订单簿缓冲区长度")
    features_lookback_secs: int = Field(default=60, ge=10, description="特征回溯窗口（秒）")


class HarvesterConfig(BaseModel):
    """Harvester组件配置"""
    symbols: List[str] = Field(default=["BTCUSDT", "ETHUSDT"], min_items=1, description="交易对列表")
    paths: HarvesterPathsConfig = Field(default_factory=HarvesterPathsConfig)
    buffers: HarvesterBuffersConfig = Field(default_factory=HarvesterBuffersConfig)
    files: HarvesterFilesConfig = Field(default_factory=HarvesterFilesConfig)
    concurrency: HarvesterConcurrencyConfig = Field(default_factory=HarvesterConcurrencyConfig)
    timeouts: HarvesterTimeoutsConfig = Field(default_factory=HarvesterTimeoutsConfig)
    health: HarvesterHealthConfig = Field(default_factory=HarvesterHealthConfig)
    thresholds: HarvesterThresholdsConfig = Field(default_factory=HarvesterThresholdsConfig)
    dedup: HarvesterDedupConfig = Field(default_factory=HarvesterDedupConfig)
    scenario: HarvesterScenarioConfig = Field(default_factory=HarvesterScenarioConfig)
    tuning: HarvesterTuningConfig = Field(default_factory=HarvesterTuningConfig)

