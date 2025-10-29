"""
未消费键检测：检测配置中未被任何组件Schema使用的键
用于防止拼写错误和悬空配置
"""

from typing import Dict, Any, Set, List, Optional
from pathlib import Path


# 组件Schema消费的键白名单
# 格式：{component: {set of key paths}}
SCHEMA_CONSUMED_KEYS = {
    'ofi': {
        'components.ofi.z_window',
        'components.ofi.ema_alpha',
        'components.ofi.z_clip',
        'components.ofi.reset_on_gap_ms',
        'components.ofi.reset_on_session_change',
        'components.ofi.per_symbol_window',
        'components.ofi.levels',
        'components.ofi.weights',
    },
    'cvd': {
        'components.cvd.z_mode',
        'components.cvd.z_window',
        'components.cvd.half_life_trades',
        'components.cvd.winsor_limit',
        'components.cvd.freeze_min',
        'components.cvd.stale_threshold_ms',
        'components.cvd.soft_freeze_ms',
        'components.cvd.hard_freeze_ms',
        'components.cvd.scale_mode',
        'components.cvd.ewma_fast_hl',
        'components.cvd.mad_window_trades',
        'components.cvd.mad_scale_factor',
        'components.cvd.scale_fast_weight',
        'components.cvd.scale_slow_weight',
        'components.cvd.mad_multiplier',
        'components.cvd.post_stale_freeze',
        'components.cvd.ema_alpha',
        'components.cvd.use_tick_rule',
        'components.cvd.warmup_min',
        'components.cvd.symbol_overrides',
    },
    'fusion': {
        'components.fusion.thresholds.fuse_buy',
        'components.fusion.thresholds.fuse_sell',
        'components.fusion.thresholds.fuse_strong_buy',
        'components.fusion.thresholds.fuse_strong_sell',
        'components.fusion.consistency.min_consistency',
        'components.fusion.consistency.strong_min_consistency',
        'components.fusion.weights.w_ofi',
        'components.fusion.weights.w_cvd',
        'components.fusion.smoothing.z_window',
        'components.fusion.smoothing.winsorize_percentile',
        'components.fusion.smoothing.mad_k',
    },
    'divergence': {
        'components.divergence.min_strength',
        'components.divergence.min_separation_secs',
        'components.divergence.count_conflict_only_when_fusion_ge',
        'components.divergence.lookback_periods',
        'components.divergence.swing_L',
        'components.divergence.ema_k',
        'components.divergence.z_hi',
        'components.divergence.z_mid',
        'components.divergence.min_separation',
        'components.divergence.cooldown_secs',
        'components.divergence.warmup_min',
        'components.divergence.max_lag',
        'components.divergence.use_fusion',
        'components.divergence.cons_min',
    },
    'strategy': {
        'components.strategy.mode',
        'components.strategy.hysteresis.window_secs',
        'components.strategy.hysteresis.min_active_windows',
        'components.strategy.hysteresis.min_quiet_windows',
        'components.strategy.triggers.schedule.enabled',
        'components.strategy.triggers.schedule.timezone',
        'components.strategy.triggers.schedule.calendar',
        'components.strategy.triggers.schedule.enabled_weekdays',
        'components.strategy.triggers.schedule.holidays',
        'components.strategy.triggers.schedule.active_windows',
        'components.strategy.triggers.schedule.wrap_midnight',
        'components.strategy.triggers.market.enabled',
        'components.strategy.triggers.market.window_secs',
        'components.strategy.triggers.market.min_trades_per_min',
        'components.strategy.triggers.market.min_quote_updates_per_sec',
        'components.strategy.triggers.market.max_spread_bps',
        'components.strategy.triggers.market.min_volatility_bps',
        'components.strategy.triggers.market.min_volume_usd',
        'components.strategy.triggers.market.use_median',
        'components.strategy.triggers.market.winsorize_percentile',
        'components.strategy.scenarios_file',
    },
    'core_algo': {
        'components.core_algo.guards.spread_bps_cap',
        'components.core_algo.guards.max_missing_msgs_rate',
        'components.core_algo.guards.max_event_lag_sec',
        'components.core_algo.guards.exit_cooldown_sec',
        'components.core_algo.guards.reconnect_cooldown_sec',
        'components.core_algo.guards.resync_cooldown_sec',
        'components.core_algo.guards.reverse_prevention_sec',
        'components.core_algo.guards.warmup_period_sec',
        'components.core_algo.output.sinks',
        'components.core_algo.output.weak_signal_threshold',
    },
    'harvester': {
        'components.harvester.symbols',
        'components.harvester.paths.output_dir',
        'components.harvester.paths.preview_dir',
        'components.harvester.paths.artifacts_dir',
        'components.harvester.buffers.high.prices',
        'components.harvester.buffers.high.orderbook',
        'components.harvester.buffers.high.ofi',
        'components.harvester.buffers.high.cvd',
        'components.harvester.buffers.high.fusion',
        'components.harvester.buffers.high.events',
        'components.harvester.buffers.high.features',
        'components.harvester.buffers.emergency.prices',
        'components.harvester.buffers.emergency.orderbook',
        'components.harvester.buffers.emergency.ofi',
        'components.harvester.buffers.emergency.cvd',
        'components.harvester.buffers.emergency.fusion',
        'components.harvester.buffers.emergency.events',
        'components.harvester.buffers.emergency.features',
        'components.harvester.files.max_rows_per_file',
        'components.harvester.files.parquet_rotate_sec',
        'components.harvester.concurrency.save_concurrency',
        'components.harvester.timeouts.stream_idle_sec',
        'components.harvester.timeouts.trade_timeout',
        'components.harvester.timeouts.orderbook_timeout',
        'components.harvester.timeouts.health_check_interval',
        'components.harvester.timeouts.backoff_reset_secs',
        'components.harvester.thresholds.extreme_traffic_threshold',
        'components.harvester.thresholds.extreme_rotate_sec',
        'components.harvester.thresholds.ofi_max_lag_ms',
        'components.harvester.dedup.lru_size',
        'components.harvester.dedup.queue_drop_threshold',
        'components.harvester.scenario.win_secs',
        'components.harvester.scenario.active_tps',
        'components.harvester.scenario.vol_split',
        'components.harvester.scenario.fee_tier',
    },
    # 全局运行时配置
    'runtime': {
        'logging.level',
        'logging.debug',
        'logging.heartbeat_interval_sec',
        'performance.max_queue_size',
        'performance.batch_size',
        'performance.flush_interval_ms',
    },
}

# 允许的全局键（不在components下，但允许存在）
ALLOWED_GLOBAL_KEYS = {
    'system',
    'divergence_detection',
    'fusion_metrics',
    'gating',
    'data_source',
    'paths',
    'monitoring',
}


def _get_all_keys(cfg: Dict[str, Any], prefix: str = '') -> Set[str]:
    """递归获取配置中的所有键路径"""
    keys = set()
    
    for key, value in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            keys.add(full_key)
            keys.update(_get_all_keys(value, full_key))
        else:
            keys.add(full_key)
    
    return keys


def check_unconsumed_keys(cfg: Dict[str, Any], component: Optional[str] = None,
                          fail_on_unconsumed: bool = False) -> List[str]:
    """
    检查未消费的配置键
    
    Args:
        cfg: 配置字典
        component: 组件名称，None表示检查所有组件
        fail_on_unconsumed: 如果为True，发现未消费键时抛出异常
    
    Returns:
        未消费键的路径列表
    """
    all_keys = _get_all_keys(cfg)
    
    # 收集所有允许的键
    allowed_keys = set()
    
    if component:
        # 只检查指定组件的键
        if component in SCHEMA_CONSUMED_KEYS:
            allowed_keys.update(SCHEMA_CONSUMED_KEYS[component])
        # 添加运行时配置
        allowed_keys.update(SCHEMA_CONSUMED_KEYS.get('runtime', set()))
    else:
        # 检查所有组件
        for comp_keys in SCHEMA_CONSUMED_KEYS.values():
            allowed_keys.update(comp_keys)
    
    # 添加全局允许的键（允许其子键）
    for global_key in ALLOWED_GLOBAL_KEYS:
        allowed_keys.add(global_key)
        # 允许所有以该全局键为前缀的键
        for key in all_keys:
            if key.startswith(f"{global_key}."):
                allowed_keys.add(key)
    
    # 特殊处理：允许__meta__和__invariants__（运行时包元信息）
    allowed_keys.add('__meta__')
    allowed_keys.add('__invariants__')
    for key in all_keys:
        if key.startswith('__meta__.') or key.startswith('__invariants__.'):
            allowed_keys.add(key)
    
    # 找出未消费的键
    unconsumed = []
    for key in all_keys:
        if key not in allowed_keys:
            # 检查是否是某个允许键的子键或父键
            is_related = False
            for allowed in allowed_keys:
                if key == allowed or key.startswith(f"{allowed}.") or allowed.startswith(f"{key}."):
                    is_related = True
                    break
            if not is_related:
                unconsumed.append(key)
    
    if fail_on_unconsumed and unconsumed:
        raise ValueError(
            f"发现未消费的配置键（可能是拼写错误）：\n" +
            "\n".join([f"  - {key}" for key in sorted(unconsumed)])
        )
    
    return sorted(unconsumed)

