"""
交易流处理配置加载器

从统一配置系统加载交易流处理的所有参数
支持环境变量覆盖和配置热更新

Author: V13 OFI+CVD AI System
Created: 2025-10-20
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging

from utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

@dataclass
class QueueConfig:
    """队列配置"""
    size: int = 4096  # 扩容以降低丢弃率
    max_size: int = 8192
    backpressure_threshold: float = 0.8

@dataclass
class LoggingConfig:
    """日志配置"""
    print_every: int = 100
    stats_interval: float = 60.0
    log_level: str = "INFO"

@dataclass
class WebSocketConfig:
    """WebSocket配置"""
    heartbeat_timeout: int = 30
    backoff_max: int = 15
    ping_interval: int = 20
    close_timeout: int = 10
    reconnect_delay: float = 1.0
    max_reconnect_attempts: int = 10

@dataclass
class PerformanceConfig:
    """性能配置"""
    watermark_ms: int = 300  # 降低到300ms进一步加速处理
    batch_size: int = 10
    max_processing_rate: int = 1000
    memory_limit_mb: int = 100

@dataclass
class MonitoringConfig:
    """监控配置"""
    prometheus_port: int = 8008
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
class TradeStreamConfig:
    """交易流处理完整配置"""
    # 基础配置
    enabled: bool = True
    
    # 子配置
    queue: QueueConfig = None
    logging: LoggingConfig = None
    websocket: WebSocketConfig = None
    performance: PerformanceConfig = None
    monitoring: MonitoringConfig = None
    hot_reload: HotReloadConfig = None
    
    def __post_init__(self):
        """初始化默认值"""
        if self.queue is None:
            self.queue = QueueConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.websocket is None:
            self.websocket = WebSocketConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.hot_reload is None:
            self.hot_reload = HotReloadConfig()

class TradeStreamConfigLoader:
    """交易流处理配置加载器"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
    
    def load_config(self) -> TradeStreamConfig:
        """
        从统一配置系统加载交易流处理配置
        
        Returns:
            TradeStreamConfig: 交易流处理配置对象
        """
        try:
            # 获取交易流处理配置
            trade_config = self.config_loader.get('trade_stream', {})
            
            # 基础配置
            enabled = trade_config.get('enabled', True)
            
            # 队列配置
            queue_raw = trade_config.get('queue', {})
            queue = QueueConfig(
                size=queue_raw.get('size', 1024),
                max_size=queue_raw.get('max_size', 2048),
                backpressure_threshold=queue_raw.get('backpressure_threshold', 0.8)
            )
            
            # 日志配置
            logging_raw = trade_config.get('logging', {})
            logging_config = LoggingConfig(
                print_every=logging_raw.get('print_every', 100),
                stats_interval=logging_raw.get('stats_interval', 60.0),
                log_level=logging_raw.get('log_level', 'INFO')
            )
            
            # WebSocket配置
            websocket_raw = trade_config.get('websocket', {})
            websocket = WebSocketConfig(
                heartbeat_timeout=websocket_raw.get('heartbeat_timeout', 30),
                backoff_max=websocket_raw.get('backoff_max', 15),
                ping_interval=websocket_raw.get('ping_interval', 20),
                close_timeout=websocket_raw.get('close_timeout', 10),
                reconnect_delay=websocket_raw.get('reconnect_delay', 1.0),
                max_reconnect_attempts=websocket_raw.get('max_reconnect_attempts', 10)
            )
            
            # 性能配置
            performance_raw = trade_config.get('performance', {})
            performance = PerformanceConfig(
                watermark_ms=performance_raw.get('watermark_ms', 1000),
                batch_size=performance_raw.get('batch_size', 10),
                max_processing_rate=performance_raw.get('max_processing_rate', 1000),
                memory_limit_mb=performance_raw.get('memory_limit_mb', 100)
            )
            
            # 监控配置
            monitoring_raw = trade_config.get('monitoring', {})
            prometheus_raw = monitoring_raw.get('prometheus', {})
            monitoring = MonitoringConfig(
                prometheus_port=prometheus_raw.get('port', 8008),
                prometheus_path=prometheus_raw.get('path', '/metrics'),
                prometheus_scrape_interval=prometheus_raw.get('scrape_interval', '5s'),
                alerts_enabled=monitoring_raw.get('alerts', {}).get('enabled', True)
            )
            
            # 热更新配置
            hot_reload_raw = trade_config.get('hot_reload', {})
            hot_reload = HotReloadConfig(
                enabled=hot_reload_raw.get('enabled', True),
                watch_file=hot_reload_raw.get('watch_file', True),
                reload_delay=hot_reload_raw.get('reload_delay', 1.0),
                backup_config=hot_reload_raw.get('backup_config', True),
                log_changes=hot_reload_raw.get('log_changes', True)
            )
            
            return TradeStreamConfig(
                enabled=enabled,
                queue=queue,
                logging=logging_config,
                websocket=websocket,
                performance=performance,
                monitoring=monitoring,
                hot_reload=hot_reload
            )
            
        except Exception as e:
            logger.error(f"Error loading trade stream config: {e}. Using default config.")
            return TradeStreamConfig()
    
    def get_queue_config(self) -> QueueConfig:
        """获取队列配置"""
        config = self.load_config()
        return config.queue
    
    def get_logging_config(self) -> LoggingConfig:
        """获取日志配置"""
        config = self.load_config()
        return config.logging
    
    def get_websocket_config(self) -> WebSocketConfig:
        """获取WebSocket配置"""
        config = self.load_config()
        return config.websocket
    
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
