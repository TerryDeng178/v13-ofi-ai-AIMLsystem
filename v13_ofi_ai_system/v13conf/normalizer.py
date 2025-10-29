"""
配置键名归一化和单位统一
"""

from typing import Dict, Any, List, Optional
import warnings


# 键名兼容映射（旧键 -> 新键）
KEY_MAPPINGS = {
    # Fusion相关
    'fuseStrongBuy': 'fuse_strong_buy',
    'fuseStrongSell': 'fuse_strong_sell',
    'fuseBuy': 'fuse_buy',
    'fuseSell': 'fuse_sell',
    'minConsistency': 'min_consistency',
    'strongMinConsistency': 'strong_min_consistency',
    
    # Divergence相关
    'Z_HI': 'z_hi',  # 回退到通用键，或映射到 z_hi_long/z_hi_short
    'Z_MID': 'z_mid',
    'consMin': 'cons_min',
    'minSeparation': 'min_separation',
    
    # 通用
    'zWindow': 'z_window',
    'emaAlpha': 'ema_alpha',
    'zClip': 'z_clip',
    'winsorLimit': 'winsor_limit',
    'madMultiplier': 'mad_multiplier',
}

# 单位归一化函数
def normalize_winsorize_percentile(value: Any) -> int:
    """将winsorize百分位归一化为0-100的整数"""
    if isinstance(value, float):
        # 如果是0-1之间的小数，转换为0-100
        if 0 < value < 1:
            return int(value * 100)
        # 如果已经是百分数，转换为整数
        return int(value)
    if isinstance(value, int):
        # 如果>100，可能是错误输入，保持原样但警告
        if value > 100:
            warnings.warn(f"winsorize_percentile value {value} seems incorrect (expected 0-100)")
        return value
    return int(float(value))


def normalize_percentage(value: Any) -> float:
    """将百分比归一化为0-1之间的小数"""
    if isinstance(value, str) and value.endswith('%'):
        return float(value.rstrip('%')) / 100.0
    if isinstance(value, (int, float)):
        # 如果>1，假设是百分比
        if value > 1:
            return value / 100.0
        return float(value)
    return float(value)


NORMALIZERS = {
    'winsorize_percentile': normalize_winsorize_percentile,
    'winsor_limit': lambda v: float(v),  # 保持原值但确保是float
    'mad_multiplier': lambda v: float(v),
    'z_window': lambda v: int(v),
    'ema_alpha': normalize_percentage,  # 如果输入是百分比字符串
    'z_clip': lambda v: float(v),
}


def normalize(cfg: Dict[str, Any], warn_compat: bool = True, 
              path_prefix: str = '') -> Dict[str, Any]:
    """
    归一化配置：键名映射和单位统一
    
    Args:
        cfg: 配置字典
        warn_compat: 是否打印兼容性警告
        path_prefix: 当前路径前缀（用于警告信息）
    
    Returns:
        归一化后的配置字典
    """
    result = {}
    
    for key, value in cfg.items():
        # 检查键名映射
        new_key = KEY_MAPPINGS.get(key, key)
        if new_key != key and warn_compat:
            full_path = f"{path_prefix}.{key}" if path_prefix else key
            warnings.warn(
                f"Compatibility: '{full_path}' → '{new_key}'",
                DeprecationWarning,
                stacklevel=2
            )
        
        # 递归处理嵌套字典
        if isinstance(value, dict):
            new_path = f"{path_prefix}.{new_key}" if path_prefix else new_key
            result[new_key] = normalize(value, warn_compat, new_path)
        # 应用单位归一化
        elif new_key in NORMALIZERS:
            try:
                result[new_key] = NORMALIZERS[new_key](value)
            except (ValueError, TypeError) as e:
                if warn_compat:
                    full_path = f"{path_prefix}.{new_key}" if path_prefix else new_key
                    warnings.warn(
                        f"Failed to normalize '{full_path}': {e}. Keeping original value.",
                        stacklevel=2
                    )
                result[new_key] = value
        else:
            result[new_key] = value
    
    return result


def get_nested_value(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """通过点号分隔的路径获取值"""
    keys = path.split('.')
    current = cfg
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

