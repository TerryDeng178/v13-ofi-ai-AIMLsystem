"""
策略模式管理器配置加载器

从统一配置系统加载策略模式管理器的所有参数
支持环境变量覆盖和配置热更新

Author: V13 OFI+CVD AI System
Created: 2025-10-20
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

@dataclass
class TimeWindow:
    """时间窗口配置"""
    start: str
    end: str
    timezone: str = "Asia/Hong_Kong"

@dataclass
class ScheduleConfig:
    """时间表触发器配置"""
    enabled: bool = True
    timezone: str = "Asia/Hong_Kong"
    calendar: str = "CRYPTO"
    enabled_weekdays: List[str] = field(default_factory=lambda: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    holidays: List[str] = field(default_factory=list)
    active_windows: List[TimeWindow] = field(default_factory=list)
    wrap_midnight: bool = True

@dataclass
class MarketConfig:
    """市场触发器配置"""
    enabled: bool = True
    window_secs: int = 60
    min_trades_per_min: float = 500.0
    min_quote_updates_per_sec: float = 100.0
    max_spread_bps: float = 5.0
    min_volatility_bps: float = 10.0
    min_volume_usd: float = 1000000.0
    use_median: bool = True
    winsorize_percentile: float = 95.0

@dataclass
class HysteresisConfig:
    """迟滞配置"""
    window_secs: int = 60
    min_active_windows: int = 3
    min_quiet_windows: int = 6

@dataclass
class FeaturesConfig:
    """特性开关配置"""
    dynamic_mode_enabled: bool = True
    dry_run: bool = False

@dataclass
class MonitoringConfig:
    """监控配置"""
    prometheus_port: int = 8006
    prometheus_path: str = "/metrics"
    prometheus_scrape_interval: str = "5s"
    alerts_enabled: bool = True

@dataclass
class HotReloadConfig:
    """热更新配置"""
    enabled: bool = True
    watch_file: bool = True
    reload_delay: float = 2.0
    backup_config: bool = True
    log_changes: bool = True

@dataclass
class StrategyModeConfig:
    """策略模式管理器完整配置"""
    # 基础配置
    default_mode: str = "auto"  # auto | active | quiet
    
    # 子配置
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    hot_reload: HotReloadConfig = field(default_factory=HotReloadConfig)

class StrategyModeConfigLoader:
    """策略模式管理器配置加载器"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
    
    def load_config(self) -> StrategyModeConfig:
        """
        从统一配置系统加载策略模式管理器配置
        
        Returns:
            StrategyModeConfig: 策略模式管理器配置对象
        """
        try:
            # 获取策略模式配置
            strategy_config = self.config_loader.get('strategy_mode', {})
            
            # 基础配置
            default_mode = strategy_config.get('default_mode', 'auto')
            
            # 迟滞配置
            hysteresis_raw = strategy_config.get('hysteresis', {})
            hysteresis = HysteresisConfig(
                window_secs=hysteresis_raw.get('window_secs', 60),
                min_active_windows=hysteresis_raw.get('min_active_windows', 3),
                min_quiet_windows=hysteresis_raw.get('min_quiet_windows', 6)
            )
            
            # 时间表触发器配置
            schedule_raw = strategy_config.get('triggers', {}).get('schedule', {})
            active_windows = []
            for window_raw in schedule_raw.get('active_windows', []):
                active_windows.append(TimeWindow(
                    start=window_raw.get('start', '09:00'),
                    end=window_raw.get('end', '16:00'),
                    timezone=window_raw.get('timezone', 'Asia/Hong_Kong')
                ))
            
            schedule = ScheduleConfig(
                enabled=schedule_raw.get('enabled', True),
                timezone=schedule_raw.get('timezone', 'Asia/Hong_Kong'),
                calendar=schedule_raw.get('calendar', 'CRYPTO'),
                enabled_weekdays=schedule_raw.get('enabled_weekdays', ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
                holidays=schedule_raw.get('holidays', []),
                active_windows=active_windows,
                wrap_midnight=schedule_raw.get('wrap_midnight', True)
            )
            
            # 市场触发器配置
            market_raw = strategy_config.get('triggers', {}).get('market', {})
            market = MarketConfig(
                enabled=market_raw.get('enabled', True),
                window_secs=market_raw.get('window_secs', 60),
                min_trades_per_min=market_raw.get('min_trades_per_min', 500.0),
                min_quote_updates_per_sec=market_raw.get('min_quote_updates_per_sec', 100.0),
                max_spread_bps=market_raw.get('max_spread_bps', 5.0),
                min_volatility_bps=market_raw.get('min_volatility_bps', 10.0),
                min_volume_usd=market_raw.get('min_volume_usd', 1000000.0),
                use_median=market_raw.get('use_median', True),
                winsorize_percentile=market_raw.get('winsorize_percentile', 95.0)
            )
            
            # 特性开关配置
            features_raw = strategy_config.get('features', {})
            features = FeaturesConfig(
                dynamic_mode_enabled=features_raw.get('dynamic_mode_enabled', True),
                dry_run=features_raw.get('dry_run', False)
            )
            
            # 监控配置
            monitoring_raw = strategy_config.get('monitoring', {})
            prometheus_raw = monitoring_raw.get('prometheus', {})
            monitoring = MonitoringConfig(
                prometheus_port=prometheus_raw.get('port', 8006),
                prometheus_path=prometheus_raw.get('path', '/metrics'),
                prometheus_scrape_interval=prometheus_raw.get('scrape_interval', '5s'),
                alerts_enabled=monitoring_raw.get('alerts', {}).get('enabled', True)
            )
            
            # 热更新配置
            hot_reload_raw = strategy_config.get('hot_reload', {})
            hot_reload = HotReloadConfig(
                enabled=hot_reload_raw.get('enabled', True),
                watch_file=hot_reload_raw.get('watch_file', True),
                reload_delay=hot_reload_raw.get('reload_delay', 2.0),
                backup_config=hot_reload_raw.get('backup_config', True),
                log_changes=hot_reload_raw.get('log_changes', True)
            )
            
            return StrategyModeConfig(
                default_mode=default_mode,
                hysteresis=hysteresis,
                schedule=schedule,
                market=market,
                features=features,
                monitoring=monitoring,
                hot_reload=hot_reload
            )
            
        except Exception as e:
            logger.error(f"Error loading strategy mode config: {e}. Using default config.")
            return StrategyModeConfig()
    
    def get_hysteresis_config(self) -> HysteresisConfig:
        """获取迟滞配置"""
        config = self.load_config()
        return config.hysteresis
    
    def get_schedule_config(self) -> ScheduleConfig:
        """获取时间表配置"""
        config = self.load_config()
        return config.schedule
    
    def get_market_config(self) -> MarketConfig:
        """获取市场配置"""
        config = self.load_config()
        return config.market
    
    def get_features_config(self) -> FeaturesConfig:
        """获取特性配置"""
        config = self.load_config()
        return config.features
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """获取监控配置"""
        config = self.load_config()
        return config.monitoring
    
    def get_hot_reload_config(self) -> HotReloadConfig:
        """获取热更新配置"""
        config = self.load_config()
        return config.hot_reload
