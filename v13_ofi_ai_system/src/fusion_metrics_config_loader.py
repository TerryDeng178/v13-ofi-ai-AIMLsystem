"""
融合指标收集器配置加载器

从统一配置系统加载融合指标收集器的所有参数
支持环境变量覆盖和配置热更新

Author: V13 OFI+CVD AI System
Created: 2025-10-20
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging

from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

@dataclass
class HistoryConfig:
    """历史记录配置"""
    max_records: int = 1000
    cleanup_interval: int = 300  # 5分钟

@dataclass
class CollectionConfig:
    """收集配置"""
    update_interval: float = 1.0  # 1秒
    batch_size: int = 10
    enable_warmup: bool = True
    warmup_samples: int = 50

@dataclass
class PerformanceConfig:
    """性能配置"""
    max_collection_rate: int = 100  # 每秒最大收集次数
    memory_limit_mb: int = 50
    gc_threshold: float = 0.8  # 内存使用率超过80%时触发GC

@dataclass
class MonitoringConfig:
    """监控配置"""
    prometheus_port: int = 8005
    prometheus_path: str = "/metrics"
    prometheus_scrape_interval: str = "5s"
    alerts_enabled: bool = True

@dataclass
class HotReloadConfig:
    """热更新配置"""
    enabled: bool = True
    watch_file: bool = True
    reload_delay: float = 1.0
    backup_config: bool = True
    log_changes: bool = True

@dataclass
class FusionMetricsCollectorConfig:
    """融合指标收集器完整配置"""
    # 基础配置
    enabled: bool = True
    
    # 子配置
    history: HistoryConfig = None
    collection: CollectionConfig = None
    performance: PerformanceConfig = None
    monitoring: MonitoringConfig = None
    hot_reload: HotReloadConfig = None
    
    def __post_init__(self):
        """初始化默认值"""
        if self.history is None:
            self.history = HistoryConfig()
        if self.collection is None:
            self.collection = CollectionConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.hot_reload is None:
            self.hot_reload = HotReloadConfig()

class FusionMetricsCollectorConfigLoader:
    """融合指标收集器配置加载器"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
    
    def load_config(self) -> FusionMetricsCollectorConfig:
        """
        从统一配置系统加载融合指标收集器配置
        
        Returns:
            FusionMetricsCollectorConfig: 融合指标收集器配置对象
        """
        try:
            # 获取融合指标收集器配置
            fusion_config = self.config_loader.get('fusion_metrics_collector', {})
            
            # 基础配置
            enabled = fusion_config.get('enabled', True)
            
            # 历史记录配置
            history_raw = fusion_config.get('history', {})
            history = HistoryConfig(
                max_records=history_raw.get('max_records', 1000),
                cleanup_interval=history_raw.get('cleanup_interval', 300)
            )
            
            # 收集配置
            collection_raw = fusion_config.get('collection', {})
            collection = CollectionConfig(
                update_interval=collection_raw.get('update_interval', 1.0),
                batch_size=collection_raw.get('batch_size', 10),
                enable_warmup=collection_raw.get('enable_warmup', True),
                warmup_samples=collection_raw.get('warmup_samples', 50)
            )
            
            # 性能配置
            performance_raw = fusion_config.get('performance', {})
            performance = PerformanceConfig(
                max_collection_rate=performance_raw.get('max_collection_rate', 100),
                memory_limit_mb=performance_raw.get('memory_limit_mb', 50),
                gc_threshold=performance_raw.get('gc_threshold', 0.8)
            )
            
            # 监控配置
            monitoring_raw = fusion_config.get('monitoring', {})
            prometheus_raw = monitoring_raw.get('prometheus', {})
            monitoring = MonitoringConfig(
                prometheus_port=prometheus_raw.get('port', 8005),
                prometheus_path=prometheus_raw.get('path', '/metrics'),
                prometheus_scrape_interval=prometheus_raw.get('scrape_interval', '5s'),
                alerts_enabled=monitoring_raw.get('alerts', {}).get('enabled', True)
            )
            
            # 热更新配置
            hot_reload_raw = fusion_config.get('hot_reload', {})
            hot_reload = HotReloadConfig(
                enabled=hot_reload_raw.get('enabled', True),
                watch_file=hot_reload_raw.get('watch_file', True),
                reload_delay=hot_reload_raw.get('reload_delay', 1.0),
                backup_config=hot_reload_raw.get('backup_config', True),
                log_changes=hot_reload_raw.get('log_changes', True)
            )
            
            return FusionMetricsCollectorConfig(
                enabled=enabled,
                history=history,
                collection=collection,
                performance=performance,
                monitoring=monitoring,
                hot_reload=hot_reload
            )
            
        except Exception as e:
            logger.error(f"Error loading fusion metrics collector config: {e}. Using default config.")
            return FusionMetricsCollectorConfig()
    
    def get_history_config(self) -> HistoryConfig:
        """获取历史记录配置"""
        config = self.load_config()
        return config.history
    
    def get_collection_config(self) -> CollectionConfig:
        """获取收集配置"""
        config = self.load_config()
        return config.collection
    
    def get_performance_config(self) -> PerformanceConfig:
        """获取性能配置"""
        config = self.load_config()
        return config.performance
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """获取监控配置"""
        config = self.load_config()
        return config.monitoring
    
    def get_hot_reload_config(self) -> HotReloadConfig:
        """获取热更新配置"""
        config = self.load_config()
        return config.hot_reload
