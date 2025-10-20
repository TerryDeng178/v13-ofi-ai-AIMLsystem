# -*- coding: utf-8 -*-
"""
策略模式管理器 (Strategy Mode Manager)

实现活跃/不活跃模式的自动切换，基于：
1. 时间表触发器（时段判定）
2. 市场活跃度触发器（成交量/报价/波动/点差）
3. 迟滞逻辑（防止抖动）
4. 原子热更新（Copy-on-Write/RCU）

作者: V13 Team
创建日期: 2025-10-19
"""

import os
import sys
import io
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from collections import deque
import pytz
import numpy as np

# Fix Windows UTF-8 output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logger = logging.getLogger(__name__)

# Prometheus-style metrics (简化实现，实际应使用 prometheus_client)
class PrometheusMetrics:
    """简化的Prometheus指标收集器"""
    
    def __init__(self):
        self.metrics = {}
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """设置Gauge指标"""
        key = self._make_key(name, labels)
        self.metrics[key] = {'type': 'gauge', 'value': value, 'labels': labels or {}}
    
    def inc_counter(self, name: str, labels: Dict[str, str] = None, value: float = 1.0):
        """增加Counter指标"""
        key = self._make_key(name, labels)
        if key not in self.metrics:
            self.metrics[key] = {'type': 'counter', 'value': 0.0, 'labels': labels or {}}
        self.metrics[key]['value'] += value
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录Histogram观测值"""
        key = self._make_key(name, labels)
        if key not in self.metrics:
            self.metrics[key] = {'type': 'histogram', 'values': [], 'labels': labels or {}}
        self.metrics[key]['values'].append(value)
    
    def set_info(self, name: str, labels: Dict[str, str]):
        """设置Info指标"""
        key = self._make_key(name, labels)
        self.metrics[key] = {'type': 'info', 'labels': labels}
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """生成指标键"""
        if not labels:
            return name
        label_str = ','.join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有指标"""
        return self.metrics.copy()

# 全局指标实例
_metrics = PrometheusMetrics()


class StrategyMode(Enum):
    """策略模式枚举"""
    ACTIVE = "active"
    QUIET = "quiet"


class TriggerReason(Enum):
    """模式切换原因"""
    SCHEDULE = "schedule"      # 时间表触发
    MARKET = "market"          # 市场指标触发
    MANUAL = "manual"          # 人工手动触发
    HYSTERESIS = "hysteresis"  # 迟滞逻辑触发


class MarketActivity:
    """市场活跃度数据"""
    def __init__(self):
        self.trades_per_min: float = 0.0
        self.quote_updates_per_sec: float = 0.0
        self.spread_bps: float = 0.0
        self.volatility_bps: float = 0.0
        self.volume_usd: float = 0.0
        self.timestamp: float = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'trades_per_min': self.trades_per_min,
            'quote_updates_per_sec': self.quote_updates_per_sec,
            'spread_bps': self.spread_bps,
            'volatility_bps': self.volatility_bps,
            'volume_usd': self.volume_usd,
            'timestamp': self.timestamp
        }


class StrategyModeManager:
    """
    策略模式管理器
    
    职责：
    1. 判定当前应处于何种模式（active/quiet）
    2. 管理模式切换的迟滞逻辑（防抖）
    3. 原子地应用参数变更
    4. 记录切换事件和指标
    """
    
    def __init__(self, config: Dict[str, Any] = None, config_loader=None):
        """
        初始化模式管理器
        
        Args:
            config: 配置字典（包含strategy配置段），默认None使用默认配置
            config_loader: 配置加载器实例，用于从统一配置系统加载参数
        """
        if config_loader:
            # 从统一配置系统加载参数
            self.config = self._load_from_config_loader(config_loader)
            logger.debug(f"Loaded config from config_loader: {self.config}")
            self.strategy_config = self.config.get('strategy', {}) if self.config else {}
        else:
            self.config = config or {}
            self.strategy_config = self.config.get('strategy', {})
        
        logger.debug(f"Final config: {self.config}")
        logger.debug(f"Final strategy_config: {self.strategy_config}")
        
        # 当前模式
        self.current_mode = StrategyMode.QUIET  # 默认从保守模式开始
        
        # 模式配置
        self.mode_setting = self.strategy_config.get('mode', 'auto')  # auto | active | quiet
        
        # 迟滞配置
        hysteresis = self.strategy_config.get('hysteresis', {})
        self.window_secs = hysteresis.get('window_secs', 60)
        self.min_active_windows = hysteresis.get('min_active_windows', 3)
        self.min_quiet_windows = hysteresis.get('min_quiet_windows', 6)
        
        # 触发器配置
        triggers = self.strategy_config.get('triggers', {})
        
        # 时间表触发器
        schedule_config = triggers.get('schedule', {})
        self.schedule_enabled = schedule_config.get('enabled', True)
        self.timezone = pytz.timezone(schedule_config.get('timezone', 'Asia/Hong_Kong'))
        self.calendar = schedule_config.get('calendar', 'CRYPTO')
        self.enabled_weekdays = schedule_config.get('enabled_weekdays', ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        self.holidays = schedule_config.get('holidays', [])
        # 处理时间窗口配置
        active_windows_raw = schedule_config.get('active_windows', [])
        if active_windows_raw and isinstance(active_windows_raw[0], dict):
            # 新格式：字典列表
            self.active_windows = self._parse_time_windows_dict(active_windows_raw)
        else:
            # 旧格式：字符串列表
            self.active_windows = self._parse_time_windows(active_windows_raw)
        self.wrap_midnight = schedule_config.get('wrap_midnight', True)
        
        # 市场触发器
        market_config = triggers.get('market', {})
        self.market_enabled = market_config.get('enabled', True)
        self.market_window_secs = market_config.get('window_secs', 60)
        self.min_trades_per_min = market_config.get('min_trades_per_min', 500)
        self.min_quote_updates_per_sec = market_config.get('min_quote_updates_per_sec', 100)
        self.max_spread_bps = market_config.get('max_spread_bps', 5)
        self.min_volatility_bps = market_config.get('min_volatility_bps', 10)
        self.min_volume_usd = market_config.get('min_volume_usd', 1000000)
        self.use_median = market_config.get('use_median', True)
        self.winsorize_percentile = market_config.get('winsorize_percentile', 95)
        
        # 市场活跃度样本窗口（用于滑动窗口+去噪）
        self.market_samples = deque(maxlen=int(self.market_window_secs / 10))  # 假设每10秒采样一次
        
        # 活跃度判定历史（用于迟滞逻辑）
        self.activity_history = deque(maxlen=max(self.min_active_windows, self.min_quiet_windows))
        
        # 统计指标
        self.mode_start_time = time.time()
        self.time_in_mode = {StrategyMode.ACTIVE: 0.0, StrategyMode.QUIET: 0.0}
        self.transitions_count = {
            'quiet_to_active': 0,
            'active_to_quiet': 0
        }
        
        # 参数锁（用于原子热更新）
        self.params_lock = threading.RLock()
        self.current_params = None  # 当前生效的参数快照
        
        # 特性开关
        features = self.config.get('features', {}).get('strategy', {})
        self.dynamic_mode_enabled = features.get('dynamic_mode_enabled', True)
        self.dry_run = features.get('dry_run', False)
        
        # 初始化Prometheus指标
        self._init_metrics()
        
        logger.info(f"StrategyModeManager initialized: mode={self.mode_setting}, "
                   f"schedule_enabled={self.schedule_enabled}, market_enabled={self.market_enabled}")
    
    def _load_from_config_loader(self, config_loader) -> Dict[str, Any]:
        """
        从统一配置系统加载策略模式管理器参数
        
        Args:
            config_loader: 统一配置加载器实例
            
        Returns:
            Dict[str, Any]: 策略模式管理器配置字典
        """
        try:
            # 导入策略模式配置加载器
            from src.strategy_mode_config_loader import StrategyModeConfigLoader
            
            # 创建策略模式配置加载器
            strategy_config_loader = StrategyModeConfigLoader(config_loader)
            config = strategy_config_loader.load_config()
            
            # 转换为原始配置格式
            return {
                'strategy': {
                    'mode': config.default_mode,
                    'hysteresis': {
                        'window_secs': config.hysteresis.window_secs,
                        'min_active_windows': config.hysteresis.min_active_windows,
                        'min_quiet_windows': config.hysteresis.min_quiet_windows
                    },
                    'triggers': {
                        'schedule': {
                            'enabled': config.schedule.enabled,
                            'timezone': config.schedule.timezone,
                            'calendar': config.schedule.calendar,
                            'enabled_weekdays': config.schedule.enabled_weekdays,
                            'holidays': config.schedule.holidays,
                            'active_windows': [
                                {
                                    'start': w.start,
                                    'end': w.end,
                                    'timezone': w.timezone
                                } for w in config.schedule.active_windows
                            ],
                            'wrap_midnight': config.schedule.wrap_midnight
                        },
                        'market': {
                            'enabled': config.market.enabled,
                            'window_secs': config.market.window_secs,
                            'min_trades_per_min': config.market.min_trades_per_min,
                            'min_quote_updates_per_sec': config.market.min_quote_updates_per_sec,
                            'max_spread_bps': config.market.max_spread_bps,
                            'min_volatility_bps': config.market.min_volatility_bps,
                            'min_volume_usd': config.market.min_volume_usd,
                            'use_median': config.market.use_median,
                            'winsorize_percentile': config.market.winsorize_percentile
                        }
                    }
                },
                'features': {
                    'strategy': {
                        'dynamic_mode_enabled': config.features.dynamic_mode_enabled,
                        'dry_run': config.features.dry_run
                    }
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to load strategy mode config from config_loader: {e}. Using default config.")
            return {
                'strategy': {
                    'mode': 'auto',
                    'hysteresis': {'window_secs': 60, 'min_active_windows': 3, 'min_quiet_windows': 6},
                    'triggers': {
                        'schedule': {'enabled': True, 'timezone': 'Asia/Hong_Kong', 'calendar': 'CRYPTO', 
                                   'enabled_weekdays': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 
                                   'holidays': [], 'active_windows': [], 'wrap_midnight': True},
                        'market': {'enabled': True, 'window_secs': 60, 'min_trades_per_min': 500, 
                                 'min_quote_updates_per_sec': 100, 'max_spread_bps': 5, 'min_volatility_bps': 10, 
                                 'min_volume_usd': 1000000, 'use_median': True, 'winsorize_percentile': 95}
                    }
                },
                'features': {'strategy': {'dynamic_mode_enabled': True, 'dry_run': False}}
            }
    
    def _init_metrics(self):
        """初始化Prometheus指标（13个策略相关指标）"""
        # 1. strategy_mode_info - 当前模式信息
        _metrics.set_info('strategy_mode_info', {'mode': self.current_mode.value})
        
        # 2. strategy_mode_active - 当前是否活跃（0=quiet, 1=active）
        _metrics.set_gauge('strategy_mode_active', 0.0 if self.current_mode == StrategyMode.QUIET else 1.0)
        
        # 3. strategy_mode_last_change_timestamp - 最后切换时间（Unix秒）
        _metrics.set_gauge('strategy_mode_last_change_timestamp', self.mode_start_time)
        
        # 4-5. strategy_mode_transitions_total - 模式切换次数（按from/to/reason标签）
        # 在切换时更新
        
        # 6. strategy_time_in_mode_seconds_total - 各模式累计时长（按mode标签）
        _metrics.set_gauge('strategy_time_in_mode_seconds_total', 0.0, {'mode': 'active'})
        _metrics.set_gauge('strategy_time_in_mode_seconds_total', 0.0, {'mode': 'quiet'})
        
        # 7-8. strategy_trigger_schedule_active, strategy_trigger_market_active - 触发器状态
        _metrics.set_gauge('strategy_trigger_schedule_active', 0.0)
        _metrics.set_gauge('strategy_trigger_market_active', 0.0)
        
        # 9-13. 触发因子指标（在update时更新）
        _metrics.set_gauge('strategy_trigger_trades_per_min', 0.0)
        _metrics.set_gauge('strategy_trigger_quote_updates_per_sec', 0.0)
        _metrics.set_gauge('strategy_trigger_spread_bps', 0.0)
        _metrics.set_gauge('strategy_trigger_volatility_bps', 0.0)
        _metrics.set_gauge('strategy_trigger_volume_usd', 0.0)  # 新增 volume_usd
        
        # 14. strategy_params_update_duration_ms - 参数更新耗时（Histogram）
        # 在apply_params时观测
        
        # 15. strategy_params_update_failures_total - 参数更新失败次数（按module标签）
        # 预初始化为0，确保"开箱即见"
        _metrics.inc_counter('strategy_params_update_failures_total', {'module': 'init'}, value=0.0)
        
        logger.debug("Prometheus metrics registered (13 strategy metrics, samples will be generated at runtime)")
    
    def _parse_time_windows(self, windows: List[str]) -> List[Tuple[int, int]]:
        """
        解析时间窗口字符串
        
        Args:
            windows: ["09:00-12:00", "21:00-02:00"] 格式的列表
        
        Returns:
            [(540, 720), (1260, 120)] 格式的列表（分钟数，支持跨午夜）
        """
        parsed = []
        for window_str in windows:
            try:
                start_str, end_str = window_str.split('-')
                start_h, start_m = map(int, start_str.split(':'))
                end_h, end_m = map(int, end_str.split(':'))
                
                start_mins = start_h * 60 + start_m
                end_mins = end_h * 60 + end_m
                
                parsed.append((start_mins, end_mins))
            except Exception as e:
                logger.error(f"Failed to parse time window '{window_str}': {e}")
        
        return parsed
    
    def _parse_time_windows_dict(self, windows: List[Dict[str, str]]) -> List[Tuple[int, int]]:
        """
        解析时间窗口字典
        
        Args:
            windows: [{"start": "09:00", "end": "12:00", "timezone": "Asia/Hong_Kong"}] 格式的列表
        
        Returns:
            [(540, 720), (1260, 120)] 格式的列表（分钟数，支持跨午夜）
        """
        parsed = []
        for window_dict in windows:
            try:
                start_str = window_dict.get('start', '09:00')
                end_str = window_dict.get('end', '16:00')
                
                start_h, start_m = map(int, start_str.split(':'))
                end_h, end_m = map(int, end_str.split(':'))
                
                start_mins = start_h * 60 + start_m
                end_mins = end_h * 60 + end_m
                
                parsed.append((start_mins, end_mins))
            except Exception as e:
                logger.error(f"Failed to parse time window dict '{window_dict}': {e}")
        
        return parsed
    
    def _is_in_time_window(self, current_mins: int, window: Tuple[int, int]) -> bool:
        """
        判断当前时间是否在窗口内（支持跨午夜）
        
        Args:
            current_mins: 当前时间（午夜以来的分钟数）
            window: (start_mins, end_mins)
        
        Returns:
            True if in window
        """
        start, end = window
        
        if end > start:
            # 正常窗口：09:00-12:00
            return start <= current_mins < end
        else:
            # 跨午夜窗口：21:00-02:00
            # 如果配置显式禁用了跨午夜，则返回False
            if not self.wrap_midnight:
                logger.warning(f"Time window {start}-{end} appears to wrap midnight but wrap_midnight=False")
                return False
            return current_mins >= start or current_mins < end
    
    def check_schedule_active(self, dt: Optional[datetime] = None) -> bool:
        """
        检查时间表触发器是否判定为活跃
        
        Args:
            dt: 待检查的时间（默认为当前时间）
        
        Returns:
            True if schedule判定为活跃
        """
        if not self.schedule_enabled:
            return False
        
        if dt is None:
            dt = datetime.now(self.timezone)
        else:
            # 处理naive datetime：如果没有时区信息，先localize再转换
            if dt.tzinfo is None:
                dt = self.timezone.localize(dt)
            else:
                dt = dt.astimezone(self.timezone)
        
        # 检查星期几
        weekday = dt.strftime('%a')  # Mon, Tue, Wed, ...
        if weekday not in self.enabled_weekdays:
            return False
        
        # 检查节假日
        date_str = dt.strftime('%Y-%m-%d')
        if date_str in self.holidays:
            return False
        
        # 检查活跃时段
        current_mins = dt.hour * 60 + dt.minute
        
        for window in self.active_windows:
            if self._is_in_time_window(current_mins, window):
                return True
        
        return False
    
    def check_market_active(self, activity: MarketActivity) -> bool:
        """
        检查市场触发器是否判定为活跃（基于滑动窗口+稳健统计）
        
        Args:
            activity: 市场活跃度数据
        
        Returns:
            True if market判定为活跃
        """
        if not self.market_enabled:
            return False
        
        # 将当前活跃度样本加入窗口
        self.market_samples.append(activity)
        
        # 如果样本不足，使用当前单次快照（启动初期）
        if len(self.market_samples) < 3:
            conditions = [
                activity.trades_per_min >= self.min_trades_per_min,
                activity.quote_updates_per_sec >= self.min_quote_updates_per_sec,
                activity.spread_bps <= self.max_spread_bps,
                activity.volatility_bps >= self.min_volatility_bps,
                activity.volume_usd >= self.min_volume_usd
            ]
            return all(conditions)
        
        # 基于滑动窗口计算稳健统计量
        trades = [s.trades_per_min for s in self.market_samples]
        quotes = [s.quote_updates_per_sec for s in self.market_samples]
        spreads = [s.spread_bps for s in self.market_samples]
        volatilities = [s.volatility_bps for s in self.market_samples]
        volumes = [s.volume_usd for s in self.market_samples]
        
        # 应用 winsorize（去极值）
        def winsorize(values, percentile):
            """将超过指定分位数的值截断"""
            threshold = np.percentile(values, percentile)
            return [min(v, threshold) for v in values]
        
        if self.winsorize_percentile < 100:
            trades = winsorize(trades, self.winsorize_percentile)
            quotes = winsorize(quotes, self.winsorize_percentile)
            volatilities = winsorize(volatilities, self.winsorize_percentile)
            volumes = winsorize(volumes, self.winsorize_percentile)
        
        # 使用中位数或平均值
        if self.use_median:
            avg_trades = np.median(trades)
            avg_quotes = np.median(quotes)
            avg_spread = np.median(spreads)  # spread 越小越好，用中位数
            avg_volatility = np.median(volatilities)
            avg_volume = np.median(volumes)
        else:
            avg_trades = np.mean(trades)
            avg_quotes = np.mean(quotes)
            avg_spread = np.mean(spreads)
            avg_volatility = np.mean(volatilities)
            avg_volume = np.mean(volumes)
        
        # 所有条件必须同时满足（基于窗口统计量）
        conditions = [
            avg_trades >= self.min_trades_per_min,
            avg_quotes >= self.min_quote_updates_per_sec,
            avg_spread <= self.max_spread_bps,
            avg_volatility >= self.min_volatility_bps,
            avg_volume >= self.min_volume_usd
        ]
        
        return all(conditions)
    
    def decide_mode(self, activity: Optional[MarketActivity] = None) -> Tuple[StrategyMode, TriggerReason, Dict[str, Any]]:
        """
        决策当前应处于何种模式
        
        Args:
            activity: 市场活跃度数据（可选）
        
        Returns:
            (目标模式, 触发原因, 触发因子快照)
        """
        # 如果是手动固定模式，直接返回
        if self.mode_setting in ['active', 'quiet']:
            target_mode = StrategyMode.ACTIVE if self.mode_setting == 'active' else StrategyMode.QUIET
            triggers = self._get_trigger_snapshot(activity)
            return target_mode, TriggerReason.MANUAL, triggers
        
        # 自动模式：检查触发器
        schedule_active = self.check_schedule_active()
        market_active = False
        if activity:
            market_active = self.check_market_active(activity)
        
        # 综合判定：schedule OR market
        is_active = schedule_active or market_active
        
        # 记录历史（用于迟滞逻辑）
        self.activity_history.append((time.time(), is_active, schedule_active, market_active))
        
        # 迟滞逻辑
        if len(self.activity_history) >= self.min_active_windows:
            # 检查是否连续满足active条件
            recent_active = [h[1] for h in list(self.activity_history)[-self.min_active_windows:]]
            if all(recent_active) and self.current_mode == StrategyMode.QUIET:
                # 切换到active
                reason = TriggerReason.SCHEDULE if schedule_active else TriggerReason.MARKET
                triggers = self._get_trigger_snapshot(activity)
                return StrategyMode.ACTIVE, reason, triggers
        
        if len(self.activity_history) >= self.min_quiet_windows:
            # 检查是否连续不满足active条件
            recent_inactive = [not h[1] for h in list(self.activity_history)[-self.min_quiet_windows:]]
            if all(recent_inactive) and self.current_mode == StrategyMode.ACTIVE:
                # 切换到quiet
                triggers = self._get_trigger_snapshot(activity)
                return StrategyMode.QUIET, TriggerReason.HYSTERESIS, triggers
        
        # 保持当前模式
        triggers = self._get_trigger_snapshot(activity)
        return self.current_mode, TriggerReason.HYSTERESIS, triggers
    
    def _get_trigger_snapshot(self, activity: Optional[MarketActivity]) -> Dict[str, Any]:
        """获取触发因子快照"""
        schedule_active = self.check_schedule_active()
        market_active = False
        
        snapshot = {
            'schedule_active': schedule_active,
            'market_active': False,
            'timestamp': datetime.now(self.timezone).isoformat()
        }
        
        if activity:
            market_active = self.check_market_active(activity)
            snapshot['market_active'] = market_active
            snapshot.update(activity.to_dict())
        
        # 更新触发器指标
        _metrics.set_gauge('strategy_trigger_schedule_active', 1.0 if schedule_active else 0.0)
        _metrics.set_gauge('strategy_trigger_market_active', 1.0 if market_active else 0.0)
        
        if activity:
            _metrics.set_gauge('strategy_trigger_trades_per_min', activity.trades_per_min)
            _metrics.set_gauge('strategy_trigger_quote_updates_per_sec', activity.quote_updates_per_sec)
            _metrics.set_gauge('strategy_trigger_spread_bps', activity.spread_bps)
            _metrics.set_gauge('strategy_trigger_volatility_bps', activity.volatility_bps)
            _metrics.set_gauge('strategy_trigger_volume_usd', activity.volume_usd)  # 新增 volume_usd 上报
        
        return snapshot
    
    def apply_params(self, mode: StrategyMode) -> Tuple[bool, List[str]]:
        """
        原子地应用参数变更（Copy-on-Write模式）
        
        Args:
            mode: 目标模式
        
        Returns:
            (是否成功, 失败的模块列表)
        """
        start_time = time.time()
        
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would apply params for mode: {mode.value}")
            # 记录耗时
            duration_ms = (time.time() - start_time) * 1000
            _metrics.observe_histogram('strategy_params_update_duration_ms', duration_ms, {'result': 'dry_run'})
            return True, []
        
        mode_key = mode.value
        params_config = self.strategy_config.get('params', {})
        
        # 1. 创建新参数快照
        new_params = {}
        for module in ['ofi', 'cvd', 'risk', 'performance']:
            if module in params_config:
                new_params[module] = params_config[module].get(mode_key, {}).copy()
        
        # 2. 原子切换
        failed_modules = []
        with self.params_lock:
            try:
                # TODO: 实际应用到各个子模块
                # 这里需要与OFI/CVD/Risk/Performance模块集成
                # 示例：
                # self.ofi_module.update_params(new_params['ofi'])
                # self.cvd_module.update_params(new_params['cvd'])
                # ...
                
                self.current_params = new_params
                
                # 记录成功指标
                duration_ms = (time.time() - start_time) * 1000
                _metrics.observe_histogram('strategy_params_update_duration_ms', duration_ms, {'result': 'success'})
                
                logger.info(f"✅ Applied params for mode: {mode.value} (took {duration_ms:.2f}ms)")
                return True, []
                
            except Exception as e:
                logger.error(f"❌ Failed to apply params for mode {mode.value}: {e}")
                
                # 记录失败指标
                duration_ms = (time.time() - start_time) * 1000
                _metrics.observe_histogram('strategy_params_update_duration_ms', duration_ms, {'result': 'rollback'})
                _metrics.inc_counter('strategy_params_update_failures_total', {'module': 'unknown'})
                
                # 回滚逻辑在这里实现
                failed_modules.append('unknown')
                return False, failed_modules
    
    def _compute_params_diff(self, old_mode: StrategyMode, new_mode: StrategyMode) -> Dict[str, str]:
        """
        计算参数差异（白名单字段）
        
        Args:
            old_mode: 旧模式
            new_mode: 新模式
        
        Returns:
            差异字典（格式："key": "old_value → new_value"）
        """
        # 白名单：只展示这些关键参数的变化
        whitelist = [
            ('ofi', 'bucket_ms'),
            ('ofi', 'depth_levels'),
            ('ofi', 'watermark_ms'),
            ('cvd', 'window_ticks'),
            ('cvd', 'ema_span'),
            ('cvd', 'denoise_sigma'),
            ('risk', 'position_limit'),
            ('risk', 'order_rate_limit_per_min'),
            ('performance', 'print_every'),
            ('performance', 'flush_metrics_interval_ms')
        ]
        
        params_config = self.strategy_config.get('params', {})
        old_params = {}
        new_params = {}
        
        for module in ['ofi', 'cvd', 'risk', 'performance']:
            if module in params_config:
                old_params[module] = params_config[module].get(old_mode.value, {})
                new_params[module] = params_config[module].get(new_mode.value, {})
        
        diff = {}
        for module, key in whitelist:
            if module in old_params and module in new_params:
                old_val = old_params[module].get(key)
                new_val = new_params[module].get(key)
                if old_val is not None and new_val is not None and old_val != new_val:
                    diff[f"{module}.{key}"] = f"{old_val} → {new_val}"
        
        # 截断：最多显示10个差异
        if len(diff) > 10:
            truncated = dict(list(diff.items())[:10])
            truncated['_truncated'] = f"... and {len(diff) - 10} more"
            return truncated
        
        return diff
    
    def update_mode(self, activity: Optional[MarketActivity] = None) -> Dict[str, Any]:
        """
        更新模式（主要入口方法）
        
        Args:
            activity: 市场活跃度数据
        
        Returns:
            切换事件信息（如果发生切换）
        """
        # 优先级1: 检查动态切换总开关
        if not self.dynamic_mode_enabled and self.mode_setting == 'auto':
            # 开关关闭但模式为auto：仅刷新触发器指标，不做任何模式变更
            triggers = self._get_trigger_snapshot(activity)
            logger.debug(f"Dynamic mode switching disabled, keeping current mode: {self.current_mode.value}")
            return {}
        
        # 决策目标模式
        target_mode, reason, triggers = self.decide_mode(activity)
        
        # 检查是否需要切换
        if target_mode == self.current_mode:
            # 无需切换，仅更新统计和指标
            elapsed = time.time() - self.mode_start_time
            self.time_in_mode[self.current_mode] += elapsed
            self.mode_start_time = time.time()
            
            # 更新时长指标
            _metrics.set_gauge('strategy_time_in_mode_seconds_total', 
                             self.time_in_mode[StrategyMode.ACTIVE], {'mode': 'active'})
            _metrics.set_gauge('strategy_time_in_mode_seconds_total', 
                             self.time_in_mode[StrategyMode.QUIET], {'mode': 'quiet'})
            
            return {}
        
        # 需要切换
        old_mode = self.current_mode
        start_time = time.time()
        
        # 计算参数差异
        params_diff = self._compute_params_diff(old_mode, target_mode)
        
        # 应用参数
        success, failed_modules = self.apply_params(target_mode)
        
        if not success:
            # 应用失败，回滚
            event = {
                'event': 'mode_change_failed',
                'from': old_mode.value,
                'to': target_mode.value,
                'reason': reason.value,
                'timestamp': datetime.now(self.timezone).isoformat(),
                'config_version': self.config.get('system', {}).get('version', 'unknown'),
                'env': self.config.get('system', {}).get('environment', 'unknown'),
                'triggers': triggers,
                'params_diff': params_diff,
                'update_duration_ms': (time.time() - start_time) * 1000,
                'rollback': True,
                'failed_modules': failed_modules
            }
            
            # 结构化日志（JSON格式）
            logger.error(f"❌ Mode change failed: {json.dumps(event, ensure_ascii=False)}")
            
            return event
        
        # 切换成功
        elapsed = time.time() - self.mode_start_time
        self.time_in_mode[old_mode] += elapsed
        
        self.current_mode = target_mode
        self.mode_start_time = time.time()
        
        # 更新计数
        transition_key = f"{old_mode.value}_to_{target_mode.value}"
        self.transitions_count[transition_key] = self.transitions_count.get(transition_key, 0) + 1
        
        # 更新Prometheus指标
        _metrics.set_info('strategy_mode_info', {'mode': target_mode.value})
        _metrics.set_gauge('strategy_mode_active', 1.0 if target_mode == StrategyMode.ACTIVE else 0.0)
        _metrics.set_gauge('strategy_mode_last_change_timestamp', self.mode_start_time)
        _metrics.inc_counter('strategy_mode_transitions_total', {
            'from': old_mode.value,
            'to': target_mode.value,
            'reason': reason.value
        })
        _metrics.set_gauge('strategy_time_in_mode_seconds_total', 
                         self.time_in_mode[StrategyMode.ACTIVE], {'mode': 'active'})
        _metrics.set_gauge('strategy_time_in_mode_seconds_total', 
                         self.time_in_mode[StrategyMode.QUIET], {'mode': 'quiet'})
        
        # 构造切换事件
        event = {
            'event': 'mode_changed',
            'from': old_mode.value,
            'to': target_mode.value,
            'reason': reason.value,
            'timestamp': datetime.now(self.timezone).isoformat(),
            'config_version': self.config.get('system', {}).get('version', 'unknown'),
            'env': self.config.get('system', {}).get('environment', 'unknown'),
            'triggers': triggers,
            'params_diff': params_diff,
            'update_duration_ms': (time.time() - start_time) * 1000,
            'rollback': False,
            'failed_modules': []
        }
        
        # 结构化日志（JSON格式）
        logger.info(f"🔄 Mode changed: {json.dumps(event, ensure_ascii=False)}")
        
        return event
    
    def get_current_mode(self) -> StrategyMode:
        """获取当前模式"""
        return self.current_mode
    
    def get_mode_stats(self) -> Dict[str, Any]:
        """获取模式统计信息"""
        # 更新当前模式的时长
        elapsed = time.time() - self.mode_start_time
        time_in_mode = self.time_in_mode.copy()
        time_in_mode[self.current_mode] += elapsed
        
        return {
            'current_mode': self.current_mode.value,
            'mode_start_time': self.mode_start_time,
            'time_in_mode_seconds': {
                'active': time_in_mode[StrategyMode.ACTIVE],
                'quiet': time_in_mode[StrategyMode.QUIET]
            },
            'transitions_count': self.transitions_count,
            'activity_history_length': len(self.activity_history)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取所有Prometheus指标"""
        return _metrics.get_all()


# 示例使用
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 示例配置
    config = {
        'system': {
            'version': 'v13.0.7',
            'environment': 'development'
        },
        'strategy': {
            'mode': 'auto',
            'hysteresis': {
                'window_secs': 60,
                'min_active_windows': 3,
                'min_quiet_windows': 6
            },
            'triggers': {
                'schedule': {
                    'enabled': True,
                    'timezone': 'Asia/Hong_Kong',
                    'calendar': 'CRYPTO',
                    'enabled_weekdays': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    'holidays': [],
                    'active_windows': ['09:00-12:00', '14:00-17:00', '21:00-02:00', '06:00-08:00']
                },
                'market': {
                    'enabled': True,
                    'min_trades_per_min': 500,
                    'min_quote_updates_per_sec': 100,
                    'max_spread_bps': 5,
                    'min_volatility_bps': 10,
                    'min_volume_usd': 1000000
                }
            },
            'params': {}
        },
        'features': {
            'strategy': {
                'dynamic_mode_enabled': True,
                'dry_run': False
            }
        }
    }
    
    # 创建管理器
    manager = StrategyModeManager(config)
    
    # 测试时间表判定
    print(f"\n✅ Current mode: {manager.get_current_mode().value}")
    print(f"✅ Schedule active: {manager.check_schedule_active()}")
    
    # 测试市场活跃度判定
    activity = MarketActivity()
    activity.trades_per_min = 600
    activity.quote_updates_per_sec = 120
    activity.spread_bps = 4.5
    activity.volatility_bps = 15
    activity.volume_usd = 2000000
    
    print(f"✅ Market active: {manager.check_market_active(activity)}")
    
    # 测试模式切换
    event = manager.update_mode(activity)
    if event:
        print(f"\n🔄 Mode change event: {event}")
    
    # 获取统计
    stats = manager.get_mode_stats()
    print(f"\n📊 Mode stats: {stats}")

