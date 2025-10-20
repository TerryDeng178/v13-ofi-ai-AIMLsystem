"""
背离检测配置加载器

从统一配置系统加载背离检测模块的所有参数
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
class DivergenceConfig:
    """背离检测配置"""
    # 枢轴检测参数
    swing_L: int = 12
    ema_k: int = 5
    
    # 强度阈值
    z_hi: float = 1.5
    z_mid: float = 0.7
    
    # 去噪参数
    min_separation: int = 6
    cooldown_secs: float = 1.0
    warmup_min: int = 100
    max_lag: float = 0.300
    
    # 融合参数
    use_fusion: bool = True
    
    # 性能配置
    max_events_per_second: int = 100
    batch_size: int = 10
    queue_size: int = 1000
    
    # 监控配置
    prometheus_port: int = 8004
    prometheus_path: str = "/metrics"
    prometheus_scrape_interval: str = "5s"
    
    # 热更新配置
    hot_reload_enabled: bool = True
    hot_reload_watch_file: bool = True
    hot_reload_delay: float = 1.0
    hot_reload_backup_config: bool = True
    hot_reload_log_changes: bool = True

class DivergenceConfigLoader:
    """背离检测配置加载器"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
    
    def load_config(self) -> DivergenceConfig:
        """
        从统一配置系统加载背离检测配置
        
        Returns:
            DivergenceConfig: 背离检测配置对象
        """
        try:
            # 获取背离检测配置
            divergence_config = self.config_loader.get('divergence_detection', {})
            
            # 枢轴检测参数
            pivot_config = divergence_config.get('pivot_detection', {})
            swing_L = pivot_config.get('swing_L', 12)
            ema_k = pivot_config.get('ema_k', 5)
            
            # 强度阈值
            thresholds = divergence_config.get('thresholds', {})
            z_hi = thresholds.get('z_hi', 1.5)
            z_mid = thresholds.get('z_mid', 0.7)
            
            # 去噪参数
            denoising = divergence_config.get('denoising', {})
            min_separation = denoising.get('min_separation', 6)
            cooldown_secs = denoising.get('cooldown_secs', 1.0)
            warmup_min = denoising.get('warmup_min', 100)
            max_lag = denoising.get('max_lag', 0.300)
            
            # 融合参数
            fusion = divergence_config.get('fusion', {})
            use_fusion = fusion.get('use_fusion', True)
            
            # 性能配置
            performance = divergence_config.get('performance', {})
            max_events_per_second = performance.get('max_events_per_second', 100)
            batch_size = performance.get('batch_size', 10)
            queue_size = performance.get('queue_size', 1000)
            
            # 监控配置
            monitoring = divergence_config.get('monitoring', {})
            prometheus = monitoring.get('prometheus', {})
            prometheus_port = prometheus.get('port', 8004)
            prometheus_path = prometheus.get('path', '/metrics')
            prometheus_scrape_interval = prometheus.get('scrape_interval', '5s')
            
            # 热更新配置
            hot_reload = divergence_config.get('hot_reload', {})
            hot_reload_enabled = hot_reload.get('enabled', True)
            hot_reload_watch_file = hot_reload.get('watch_file', True)
            hot_reload_delay = hot_reload.get('reload_delay', 1.0)
            hot_reload_backup_config = hot_reload.get('backup_config', True)
            hot_reload_log_changes = hot_reload.get('log_changes', True)
            
            return DivergenceConfig(
                swing_L=swing_L,
                ema_k=ema_k,
                z_hi=z_hi,
                z_mid=z_mid,
                min_separation=min_separation,
                cooldown_secs=cooldown_secs,
                warmup_min=warmup_min,
                max_lag=max_lag,
                use_fusion=use_fusion,
                max_events_per_second=max_events_per_second,
                batch_size=batch_size,
                queue_size=queue_size,
                prometheus_port=prometheus_port,
                prometheus_path=prometheus_path,
                prometheus_scrape_interval=prometheus_scrape_interval,
                hot_reload_enabled=hot_reload_enabled,
                hot_reload_watch_file=hot_reload_watch_file,
                hot_reload_delay=hot_reload_delay,
                hot_reload_backup_config=hot_reload_backup_config,
                hot_reload_log_changes=hot_reload_log_changes
            )
            
        except Exception as e:
            logger.error(f"Error loading divergence detection config: {e}. Using default config.")
            return DivergenceConfig()
    
    def get_pivot_config(self) -> Dict[str, Any]:
        """获取枢轴检测配置"""
        config = self.load_config()
        return {
            'swing_L': config.swing_L,
            'ema_k': config.ema_k
        }
    
    def get_thresholds(self) -> Dict[str, Any]:
        """获取阈值配置"""
        config = self.load_config()
        return {
            'z_hi': config.z_hi,
            'z_mid': config.z_mid
        }
    
    def get_denoising_config(self) -> Dict[str, Any]:
        """获取去噪配置"""
        config = self.load_config()
        return {
            'min_separation': config.min_separation,
            'cooldown_secs': config.cooldown_secs,
            'warmup_min': config.warmup_min,
            'max_lag': config.max_lag
        }
    
    def get_fusion_config(self) -> Dict[str, Any]:
        """获取融合配置"""
        config = self.load_config()
        return {
            'use_fusion': config.use_fusion
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """获取性能配置"""
        config = self.load_config()
        return {
            'max_events_per_second': config.max_events_per_second,
            'batch_size': config.batch_size,
            'queue_size': config.queue_size
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取监控配置"""
        config = self.load_config()
        return {
            'prometheus_port': config.prometheus_port,
            'prometheus_path': config.prometheus_path,
            'prometheus_scrape_interval': config.prometheus_scrape_interval
        }
    
    def get_hot_reload_config(self) -> Dict[str, Any]:
        """获取热更新配置"""
        config = self.load_config()
        return {
            'enabled': config.hot_reload_enabled,
            'watch_file': config.hot_reload_watch_file,
            'reload_delay': config.hot_reload_delay,
            'backup_config': config.hot_reload_backup_config,
            'log_changes': config.hot_reload_log_changes
        }
