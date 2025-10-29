#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket配置管理模块
支持从统一配置系统加载WebSocket参数，同时保持向后兼容性
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass


@dataclass
class WebSocketConfig:
    """WebSocket配置类"""
    
    # 基础连接配置
    timeout: int = 30
    reconnect_interval: int = 5
    max_reconnect_attempts: int = 0
    ping_interval: int = 20
    
    # 详细连接配置
    heartbeat_timeout: int = 60
    max_backoff_interval: int = 30
    
    # 数据流配置
    depth_levels: int = 5
    update_frequency: int = 100
    buffer_size: int = 1000
    backpressure_threshold: float = 0.8
    
    # 日志配置
    log_level: str = "INFO"
    rotate_interval: int = 60
    max_file_size: int = 5000000
    backup_count: int = 7
    enable_ndjson: bool = True
    
    # 性能监控配置
    stats_interval: int = 60
    enable_metrics: bool = True
    enable_latency_stats: bool = True
    
    # 数据存储配置
    data_dir: str = "data/order_book"
    log_dir: str = "logs"
    enable_compression: bool = True
    compression_format: str = "gzip"
    
    # WebSocket URL配置
    ws_url_template: str = "wss://fstream.binancefuture.com/stream?streams={symbol}@depth@100ms"
    rest_snap_url_template: str = "https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit=1000"


class WebSocketConfigLoader:
    """WebSocket配置加载器"""
    
    def __init__(self, config_loader=None):
        """
        初始化WebSocket配置加载器
        
        Args:
            config_loader: 统一配置加载器实例，如果为None则使用默认配置
        """
        self.config_loader = config_loader
        self._config = None
    
    def load_config(self, symbol: str = "ethusdt") -> WebSocketConfig:
        """
        加载WebSocket配置
        
        Args:
            symbol: 交易对符号
            
        Returns:
            WebSocketConfig: WebSocket配置对象
        """
        if self.config_loader is None:
            # 使用默认配置
            return WebSocketConfig()
        
        try:
            # 从统一配置系统加载
            websocket_config = self.config_loader.get('websocket', {})
            
            # 提取基础配置
            timeout = websocket_config.get('timeout', 30)
            reconnect_interval = websocket_config.get('reconnect_interval', 5)
            max_reconnect_attempts = websocket_config.get('max_reconnect_attempts', 0)
            ping_interval = websocket_config.get('ping_interval', 20)
            
            # 提取详细连接配置
            connection_config = websocket_config.get('connection', {})
            heartbeat_timeout = connection_config.get('heartbeat_timeout', 60)
            max_backoff_interval = connection_config.get('max_backoff_interval', 30)
            
            # 提取数据流配置
            stream_config = websocket_config.get('stream', {})
            depth_levels = stream_config.get('depth_levels', 5)
            update_frequency = stream_config.get('update_frequency', 100)
            buffer_size = stream_config.get('buffer_size', 1000)
            backpressure_threshold = stream_config.get('backpressure_threshold', 0.8)
            
            # 提取日志配置
            logging_config = websocket_config.get('logging', {})
            log_level = logging_config.get('level', 'INFO')
            rotate_interval = logging_config.get('rotate_interval', 60)
            max_file_size = logging_config.get('max_file_size', 5000000)
            backup_count = logging_config.get('backup_count', 7)
            enable_ndjson = logging_config.get('enable_ndjson', True)
            
            # 提取性能监控配置
            monitoring_config = websocket_config.get('monitoring', {})
            stats_interval = monitoring_config.get('stats_interval', 60)
            enable_metrics = monitoring_config.get('enable_metrics', True)
            enable_latency_stats = monitoring_config.get('enable_latency_stats', True)
            
            # 提取数据存储配置
            storage_config = websocket_config.get('storage', {})
            data_dir = storage_config.get('data_dir', 'data/order_book')
            log_dir = storage_config.get('log_dir', 'logs')
            enable_compression = storage_config.get('enable_compression', True)
            compression_format = storage_config.get('compression_format', 'gzip')
            
            # 创建配置对象
            config = WebSocketConfig(
                timeout=timeout,
                reconnect_interval=reconnect_interval,
                max_reconnect_attempts=max_reconnect_attempts,
                ping_interval=ping_interval,
                heartbeat_timeout=heartbeat_timeout,
                max_backoff_interval=max_backoff_interval,
                depth_levels=depth_levels,
                update_frequency=update_frequency,
                buffer_size=buffer_size,
                backpressure_threshold=backpressure_threshold,
                log_level=log_level,
                rotate_interval=rotate_interval,
                max_file_size=max_file_size,
                backup_count=backup_count,
                enable_ndjson=enable_ndjson,
                stats_interval=stats_interval,
                enable_metrics=enable_metrics,
                enable_latency_stats=enable_latency_stats,
                data_dir=data_dir,
                log_dir=log_dir,
                enable_compression=enable_compression,
                compression_format=compression_format
            )
            
            return config
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load WebSocket config from config_loader: {e}. Using default config.")
            return WebSocketConfig()
    
    def get_ws_url(self, symbol: str) -> str:
        """
        获取WebSocket URL
        
        Args:
            symbol: 交易对符号
            
        Returns:
            str: WebSocket URL
        """
        config = self.load_config(symbol)
        return config.ws_url_template.format(symbol=symbol.lower())
    
    def get_rest_snap_url(self, symbol: str) -> str:
        """
        获取REST快照URL
        
        Args:
            symbol: 交易对符号
            
        Returns:
            str: REST快照URL
        """
        config = self.load_config(symbol)
        return config.rest_snap_url_template.format(symbol=symbol.upper())


def create_websocket_config(config_loader=None, symbol: str = "ethusdt") -> WebSocketConfig:
    """
    创建WebSocket配置的便捷函数
    
    Args:
        config_loader: 统一配置加载器实例
        symbol: 交易对符号
        
    Returns:
        WebSocketConfig: WebSocket配置对象
    """
    loader = WebSocketConfigLoader(config_loader)
    return loader.load_config(symbol)


# 向后兼容性：提供默认配置实例
DEFAULT_WEBSOCKET_CONFIG = WebSocketConfig()
