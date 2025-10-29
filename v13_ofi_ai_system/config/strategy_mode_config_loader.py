# -*- coding: utf-8 -*-
"""
StrategyModeConfigLoader
把统一配置（strategy_mode）解析为策略模式管理器的强类型配置对象。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class HysteresisConfig:
    """迟滞配置"""
    window_secs: int = 60
    min_active_windows: int = 3
    min_quiet_windows: int = 6

@dataclass
class ScheduleConfig:
    """时间表配置"""
    enabled: bool = True
    timezone: str = "Asia/Hong_Kong"
    calendar: str = "CRYPTO"
    enabled_weekdays: List[str] = None
    holidays: List[str] = None
    active_windows: List[Dict[str, str]] = None
    wrap_midnight: bool = True
    
    def __post_init__(self):
        if self.enabled_weekdays is None:
            self.enabled_weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        if self.holidays is None:
            self.holidays = []
        if self.active_windows is None:
            self.active_windows = []

@dataclass
class MarketConfig:
    """市场触发器配置"""
    enabled: bool = True
    window_secs: int = 60
    min_trades_per_min: float = 150.0
    min_quote_updates_per_sec: float = 40.0
    max_spread_bps: float = 5.0
    min_volatility_bps: float = 4.0
    min_volume_usd: float = 200000.0
    use_median: bool = True
    winsorize_percentile: int = 95

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
class FeaturesConfig:
    """特性配置"""
    dynamic_mode_enabled: bool = True
    throttle_in_quiet: bool = True
    sample_ratio_quiet: float = 0.3
    dry_run: bool = False
    cli_override_priority: bool = True

@dataclass
class StrategyModeConfig:
    """策略模式管理器完整配置"""
    # 基础配置
    default_mode: str = "auto"  # auto | active | quiet
    weak_signal_threshold: float = 0.12
    
    # 子配置
    hysteresis: HysteresisConfig = None
    schedule: ScheduleConfig = None
    market: MarketConfig = None
    monitoring: MonitoringConfig = None
    hot_reload: HotReloadConfig = None
    features: FeaturesConfig = None
    
    # 场景参数
    scenario_parameters: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        """初始化默认值"""
        if self.hysteresis is None:
            self.hysteresis = HysteresisConfig()
        if self.schedule is None:
            self.schedule = ScheduleConfig()
        if self.market is None:
            self.market = MarketConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.hot_reload is None:
            self.hot_reload = HotReloadConfig()
        if self.features is None:
            self.features = FeaturesConfig()
        if self.scenario_parameters is None:
            self.scenario_parameters = {}

class StrategyModeConfigLoader:
    """策略模式管理器配置加载器"""
    
    def __init__(self, unified_loader):
        """
        初始化配置加载器
        
        Args:
            unified_loader: 统一配置加载器实例
        """
        self.cfg = unified_loader
    
    def load_config(self) -> StrategyModeConfig:
        """
        从统一配置系统加载策略模式管理器配置
        
        Returns:
            StrategyModeConfig: 策略模式管理器配置对象
        """
        try:
            # 获取策略模式配置
            strategy_config = self.cfg.get("strategy_mode", {})
            
            # 基础配置
            default_mode = strategy_config.get("default_mode", "auto")
            weak_signal_threshold = strategy_config.get("weak_signal_threshold", 0.12)
            
            # 迟滞配置
            hysteresis_raw = strategy_config.get("hysteresis", {})
            hysteresis = HysteresisConfig(
                window_secs=hysteresis_raw.get("window_secs", 60),
                min_active_windows=hysteresis_raw.get("min_active_windows", 3),
                min_quiet_windows=hysteresis_raw.get("min_quiet_windows", 6)
            )
            
            # 时间表配置
            schedule_raw = strategy_config.get("triggers", {}).get("schedule", {})
            schedule = ScheduleConfig(
                enabled=schedule_raw.get("enabled", True),
                timezone=schedule_raw.get("timezone", "Asia/Hong_Kong"),
                calendar=schedule_raw.get("calendar", "CRYPTO"),
                enabled_weekdays=schedule_raw.get("enabled_weekdays", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
                holidays=schedule_raw.get("holidays", []),
                active_windows=schedule_raw.get("active_windows", []),
                wrap_midnight=schedule_raw.get("wrap_midnight", True)
            )
            
            # 市场触发器配置
            market_raw = strategy_config.get("triggers", {}).get("market", {})
            market = MarketConfig(
                enabled=market_raw.get("enabled", True),
                window_secs=market_raw.get("window_secs", 60),
                min_trades_per_min=market_raw.get("min_trades_per_min", 150.0),
                min_quote_updates_per_sec=market_raw.get("min_quote_updates_per_sec", 40.0),
                max_spread_bps=market_raw.get("max_spread_bps", 5.0),
                min_volatility_bps=market_raw.get("min_volatility_bps", 4.0),
                min_volume_usd=market_raw.get("min_volume_usd", 200000.0),
                use_median=market_raw.get("use_median", True),
                winsorize_percentile=market_raw.get("winsorize_percentile", 95)
            )
            
            # 监控配置
            monitoring_raw = strategy_config.get("monitoring", {})
            prometheus_raw = monitoring_raw.get("prometheus", {})
            monitoring = MonitoringConfig(
                prometheus_port=prometheus_raw.get("port", 8006),
                prometheus_path=prometheus_raw.get("path", "/metrics"),
                prometheus_scrape_interval=prometheus_raw.get("scrape_interval", "5s"),
                alerts_enabled=monitoring_raw.get("alerts", {}).get("enabled", True)
            )
            
            # 热更新配置
            hot_reload_raw = strategy_config.get("hot_reload", {})
            hot_reload = HotReloadConfig(
                enabled=hot_reload_raw.get("enabled", True),
                watch_file=hot_reload_raw.get("watch_file", True),
                reload_delay=hot_reload_raw.get("reload_delay", 2.0),
                backup_config=hot_reload_raw.get("backup_config", True),
                log_changes=hot_reload_raw.get("log_changes", True)
            )
            
            # 特性配置
            features_raw = self.cfg.get("features", {}).get("strategy", {})
            features = FeaturesConfig(
                dynamic_mode_enabled=features_raw.get("dynamic_mode_enabled", True),
                throttle_in_quiet=features_raw.get("throttle_in_quiet", True),
                sample_ratio_quiet=features_raw.get("sample_ratio_quiet", 0.3),
                dry_run=features_raw.get("dry_run", False),
                cli_override_priority=features_raw.get("cli_override_priority", True)
            )
            
            # 场景参数
            scenario_parameters = strategy_config.get("scenario_parameters", {})
            
            return StrategyModeConfig(
                default_mode=default_mode,
                weak_signal_threshold=weak_signal_threshold,
                hysteresis=hysteresis,
                schedule=schedule,
                market=market,
                monitoring=monitoring,
                hot_reload=hot_reload,
                features=features,
                scenario_parameters=scenario_parameters
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
        """获取市场触发器配置"""
        config = self.load_config()
        return config.market
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """获取监控配置"""
        config = self.load_config()
        return config.monitoring
    
    def get_hot_reload_config(self) -> HotReloadConfig:
        """获取热更新配置"""
        config = self.load_config()
        return config.hot_reload
    
    def get_features_config(self) -> FeaturesConfig:
        """获取特性配置"""
        config = self.load_config()
        return config.features
    
    def get_scenario_parameters(self) -> Dict[str, Dict[str, Any]]:
        """获取场景参数"""
        config = self.load_config()
        return config.scenario_parameters

