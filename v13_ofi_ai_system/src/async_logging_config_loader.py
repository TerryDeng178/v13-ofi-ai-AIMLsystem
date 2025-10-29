"""
异步日志配置加载器

从统一配置系统加载异步日志的所有参数

Author: V13 OFI+CVD AI System
Created: 2025-10-20
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.utils.config_loader import ConfigLoader

@dataclass
class AsyncLoggingConfig:
    """异步日志配置"""
    # 日志级别
    level: str = "INFO"
    
    # 队列配置
    queue_max: int = 10000
    
    # 文件轮转配置
    rotate: str = "interval"  # 'interval' 或 'size'
    rotate_sec: int = 60  # 按时间轮转：秒数
    max_bytes: int = 5_000_000  # 按大小轮转：最大字节数
    backups: int = 7  # 保留备份数
    
    # 控制台输出
    to_console: bool = True
    
    def get_log_level(self) -> int:
        """获取日志级别对象"""
        return getattr(logging, self.level.upper(), logging.INFO)

class AsyncLoggingConfigLoader:
    """异步日志配置加载器"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
    
    def load_config(self, component: str = "websocket") -> AsyncLoggingConfig:
        """
        从统一配置系统加载异步日志配置
        
        Args:
            component: 组件名称（用于加载组件特定的日志配置）
        
        Returns:
            AsyncLoggingConfig: 异步日志配置对象
        """
        try:
            # 尝试从组件特定配置加载
            component_config = self.config_loader.get(f'{component}.logging', {})
            
            # 如果组件没有特定配置，使用全局日志配置
            if not component_config:
                component_config = self.config_loader.get('logging', {})
            
            return AsyncLoggingConfig(
                level=component_config.get('log_level', 'INFO'),
                queue_max=component_config.get('queue_max', 10000),
                rotate=component_config.get('rotate', 'interval'),
                rotate_sec=component_config.get('rotate_sec', 60),
                max_bytes=component_config.get('max_bytes', 5_000_000),
                backups=component_config.get('backups', 7),
                to_console=component_config.get('to_console', True)
            )
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load async logging config: {e}. Using default config.")
            return AsyncLoggingConfig()

