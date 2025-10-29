"""
配置不变量约束检查
检查互斥、依赖、和为1等约束
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


class InvariantError(Exception):
    """不变量违反错误"""
    def __init__(self, message: str, path: str, suggestion: str = ""):
        self.message = message
        self.path = path
        self.suggestion = suggestion
        super().__init__(f"{message} (path: {path})" + (f"\n建议: {suggestion}" if suggestion else ""))


def _get_nested_value(cfg: Dict[str, Any], path: str) -> Any:
    """通过点号路径获取嵌套值"""
    keys = path.split('.')
    current = cfg
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def validate_invariants(cfg: Dict[str, Any], component: Optional[str] = None) -> List[InvariantError]:
    """
    验证配置不变量约束
    
    Args:
        cfg: 配置字典
        component: 组件名称（ofi/cvd/fusion/divergence/strategy/core_algo），None表示全量检查
    
    Returns:
        错误列表，空列表表示通过
    """
    errors: List[InvariantError] = []
    
    # Fusion 权重约束：weights.ofi + weights.cvd == 1.0
    if component is None or component in ('fusion', 'core_algo'):
        fusion_weights = _get_nested_value(cfg, 'components.fusion.weights')
        if fusion_weights:
            w_ofi = fusion_weights.get('w_ofi', fusion_weights.get('ofi', 0.0))
            w_cvd = fusion_weights.get('w_cvd', fusion_weights.get('cvd', 0.0))
            
            total = w_ofi + w_cvd
            if abs(total - 1.0) > 1e-6:
                errors.append(InvariantError(
                    f"Fusion weights must sum to 1.0, got w_ofi={w_ofi}, w_cvd={w_cvd}, sum={total}",
                    "components.fusion.weights",
                    f"请调整权重使 w_ofi + w_cvd = 1.0，例如 w_ofi={w_ofi/total:.3f}, w_cvd={w_cvd/total:.3f}"
                ))
    
    # Fusion 阈值约束：strong 必须大于等于对应的普通阈值
    if component is None or component in ('fusion', 'core_algo'):
        fusion_thresholds = _get_nested_value(cfg, 'components.fusion.thresholds')
        if fusion_thresholds:
            fuse_buy = fusion_thresholds.get('fuse_buy', 0.0)
            fuse_strong_buy = fusion_thresholds.get('fuse_strong_buy', 0.0)
            fuse_sell = fusion_thresholds.get('fuse_sell', 0.0)
            fuse_strong_sell = fusion_thresholds.get('fuse_strong_sell', 0.0)
            
            if fuse_strong_buy < fuse_buy:
                errors.append(InvariantError(
                    f"fuse_strong_buy ({fuse_strong_buy}) must be >= fuse_buy ({fuse_buy})",
                    "components.fusion.thresholds.fuse_strong_buy",
                    f"建议设置 fuse_strong_buy >= {fuse_buy}"
                ))
            
            if fuse_strong_sell > fuse_sell:  # 负数比较
                errors.append(InvariantError(
                    f"fuse_strong_sell ({fuse_strong_sell}) must be <= fuse_sell ({fuse_sell})",
                    "components.fusion.thresholds.fuse_strong_sell",
                    f"建议设置 fuse_strong_sell <= {fuse_sell}"
                ))
        
        # 一致性约束：strong_min_consistency >= min_consistency
        fusion_consistency = _get_nested_value(cfg, 'components.fusion.consistency')
        if fusion_consistency:
            min_cons = fusion_consistency.get('min_consistency', 0.0)
            strong_min_cons = fusion_consistency.get('strong_min_consistency', 0.0)
            
            if strong_min_cons < min_cons:
                errors.append(InvariantError(
                    f"strong_min_consistency ({strong_min_cons}) must be >= min_consistency ({min_cons})",
                    "components.fusion.consistency.strong_min_consistency",
                    f"建议设置 strong_min_consistency >= {min_cons}"
                ))
    
    # OFI 参数范围约束
    if component is None or component == 'ofi':
        ofi = _get_nested_value(cfg, 'components.ofi')
        if ofi:
            z_window = ofi.get('z_window', 0)
            if z_window <= 0:
                errors.append(InvariantError(
                    f"z_window must be > 0, got {z_window}",
                    "components.ofi.z_window",
                    "建议设置 z_window > 0"
                ))
            
            ema_alpha = ofi.get('ema_alpha', 0.0)
            if not (0 < ema_alpha <= 1):
                errors.append(InvariantError(
                    f"ema_alpha must be in (0, 1], got {ema_alpha}",
                    "components.ofi.ema_alpha",
                    "建议设置 0 < ema_alpha <= 1"
                ))
            
            z_clip = ofi.get('z_clip', 0.0)
            if z_clip < 0:
                errors.append(InvariantError(
                    f"z_clip must be >= 0, got {z_clip}",
                    "components.ofi.z_clip",
                    "建议设置 z_clip >= 0"
                ))
    
    # CVD 参数范围约束
    if component is None or component == 'cvd':
        cvd = _get_nested_value(cfg, 'components.cvd')
        if cvd:
            winsor_limit = cvd.get('winsor_limit', 0.0)
            if winsor_limit <= 0:
                errors.append(InvariantError(
                    f"winsor_limit must be > 0, got {winsor_limit}",
                    "components.cvd.winsor_limit",
                    "建议设置 winsor_limit > 0"
                ))
            
            mad_multiplier = cvd.get('mad_multiplier', 0.0)
            if mad_multiplier <= 0:
                errors.append(InvariantError(
                    f"mad_multiplier must be > 0, got {mad_multiplier}",
                    "components.cvd.mad_multiplier",
                    "建议设置 mad_multiplier > 0"
                ))
            
            z_window = cvd.get('z_window', 0)
            if z_window <= 0:
                errors.append(InvariantError(
                    f"z_window must be > 0, got {z_window}",
                    "components.cvd.z_window",
                    "建议设置 z_window > 0"
                ))
    
    # Winsorize 百分位范围：应该在 1-100
    fusion_smoothing = _get_nested_value(cfg, 'components.fusion.smoothing')
    if fusion_smoothing:
        winsor_pct = fusion_smoothing.get('winsorize_percentile')
        if winsor_pct is not None and not (1 <= winsor_pct <= 100):
            errors.append(InvariantError(
                f"winsorize_percentile must be in [1, 100], got {winsor_pct}",
                "components.fusion.smoothing.winsorize_percentile",
                "建议设置 winsorize_percentile 在 [1, 100] 范围内"
            ))
    
    # 冷却时间关系约束
    guards = _get_nested_value(cfg, 'components.core_algo.guards')
    if guards:
        exit_cooldown = guards.get('exit_cooldown_sec', 0)
        reconnect_cooldown = guards.get('reconnect_cooldown_sec', 0)
        resync_cooldown = guards.get('resync_cooldown_sec', 0)
        
        if reconnect_cooldown < exit_cooldown:
            errors.append(InvariantError(
                f"reconnect_cooldown_sec ({reconnect_cooldown}) should be >= exit_cooldown_sec ({exit_cooldown})",
                "components.core_algo.guards.reconnect_cooldown_sec",
                f"建议设置 reconnect_cooldown_sec >= {exit_cooldown}"
            ))
        
        if resync_cooldown < reconnect_cooldown:
            errors.append(InvariantError(
                f"resync_cooldown_sec ({resync_cooldown}) should be >= reconnect_cooldown_sec ({reconnect_cooldown})",
                "components.core_algo.guards.resync_cooldown_sec",
                f"建议设置 resync_cooldown_sec >= {reconnect_cooldown}"
            ))
    
    # Divergence 参数范围约束
    if component is None or component == 'divergence':
        divergence = _get_nested_value(cfg, 'components.divergence')
        if divergence:
            min_strength = divergence.get('min_strength')
            if min_strength is not None:
                if not (0.0 <= min_strength <= 1.0):
                    errors.append(InvariantError(
                        f"min_strength must be in [0.0, 1.0], got {min_strength}",
                        "components.divergence.min_strength",
                        "建议设置 min_strength between 0.0 and 1.0"
                    ))
            
            min_separation_secs = divergence.get('min_separation_secs')
            if min_separation_secs is not None and min_separation_secs <= 0:
                errors.append(InvariantError(
                    f"min_separation_secs must be > 0, got {min_separation_secs}",
                    "components.divergence.min_separation_secs",
                    "建议设置 min_separation_secs > 0"
                ))
            
            z_hi = divergence.get('z_hi')
            if z_hi is not None and z_hi <= 0:
                errors.append(InvariantError(
                    f"z_hi must be > 0, got {z_hi}",
                    "components.divergence.z_hi",
                    "建议设置 z_hi > 0"
                ))
            
            z_mid = divergence.get('z_mid')
            if z_mid is not None and z_mid <= 0:
                errors.append(InvariantError(
                    f"z_mid must be > 0, got {z_mid}",
                    "components.divergence.z_mid",
                    "建议设置 z_mid > 0"
                ))
            
            cons_min = divergence.get('cons_min')
            if cons_min is not None:
                if not (0.0 <= cons_min <= 1.0):
                    errors.append(InvariantError(
                        f"cons_min must be in [0.0, 1.0], got {cons_min}",
                        "components.divergence.cons_min",
                        "建议设置 cons_min between 0.0 and 1.0"
                    ))
    
    # 策略模式约束
    if component is None or component == 'strategy':
        strategy = _get_nested_value(cfg, 'components.strategy')
        if strategy:
            mode = strategy.get('mode', 'auto')
            if mode not in ('auto', 'active', 'quiet'):
                errors.append(InvariantError(
                    f"strategy.mode must be one of ['auto', 'active', 'quiet'], got '{mode}'",
                    "components.strategy.mode",
                    "建议设置为 'auto', 'active', 或 'quiet'"
                ))
            
            hysteresis = strategy.get('hysteresis', {})
            min_active = hysteresis.get('min_active_windows', 0)
            min_quiet = hysteresis.get('min_quiet_windows', 0)
            
            if min_active <= 0 or min_quiet <= 0:
                errors.append(InvariantError(
                    f"hysteresis windows must be > 0, got min_active={min_active}, min_quiet={min_quiet}",
                    "components.strategy.hysteresis",
                    "建议设置 min_active_windows > 0 和 min_quiet_windows > 0"
                ))
    
    return errors

