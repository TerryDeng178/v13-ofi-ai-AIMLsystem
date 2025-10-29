"""
打印优化：脱敏、折叠、降噪
"""

from typing import Dict, Any

# 需要脱敏的键（打印时显示为***）
SENSITIVE_KEYS = {
    'api_key',
    'secret',
    'password',
    'token',
    'credentials',
}

# 需要折叠的大列表/字典键（超过N个元素时折叠）
FOLD_THRESHOLD = 10


def should_fold(value: Any) -> bool:
    """判断是否应该折叠显示"""
    if isinstance(value, (list, tuple)):
        return len(value) > FOLD_THRESHOLD
    if isinstance(value, dict):
        return len(value) > FOLD_THRESHOLD
    return False


def mask_sensitive(key_path: str, value: Any) -> Any:
    """对敏感值进行脱敏"""
    key_lower = key_path.lower()
    if any(sensitive in key_lower for sensitive in SENSITIVE_KEYS):
        if isinstance(value, str) and len(value) > 0:
            return "***" + value[-4:] if len(value) > 4 else "***"
        return "***"
    return value


def print_config_tree(cfg: Dict[str, Any], sources: Dict[str, str],
                     component: str = None, verbose: bool = False, 
                     indent: str = "", prefix: str = ""):
    """
    打印配置树（脱敏、折叠、降噪）
    
    Args:
        cfg: 配置字典
        sources: 来源追踪字典
        component: 组件名称
        verbose: 是否详细模式（显示每个键的来源）
        indent: 缩进字符串
        prefix: 当前路径前缀
    """
    source_markers = {
        'defaults': '[D]',
        'system': '[S]',
        'overrides': '[O]',
        'env': '[E]',
        'locked': '[L]',
    }
    
    for key, value in sorted(cfg.items()):
        full_key = f"{prefix}.{key}" if prefix else key
        source = sources.get(full_key, 'unknown')
        marker = source_markers.get(source, '[?]')
        
        # 脱敏处理
        masked_value = mask_sensitive(full_key, value)
        
        if isinstance(value, dict):
            # 判断是否折叠
            if should_fold(value) and not verbose:
                print(f"{indent}{marker} {key}: <dict with {len(value)} keys>")
            else:
                print(f"{indent}{marker} {key}:")
                print_config_tree(value, sources, component, verbose, indent + "  ", full_key)
        elif isinstance(value, (list, tuple)):
            # 判断是否折叠
            if should_fold(value) and not verbose:
                print(f"{indent}{marker} {key}: <list with {len(value)} items>")
            else:
                print(f"{indent}{marker} {key}: {masked_value}")
        else:
            if verbose:
                print(f"{indent}{marker} {key}: {masked_value} ({source})")
            else:
                print(f"{indent}{marker} {key}: {masked_value}")


def print_source_summary(sources: Dict[str, str], component: str = None):
    """打印来源统计摘要（只显示计数，不显示逐键）"""
    counts = {'defaults': 0, 'system': 0, 'overrides': 0, 'env': 0, 'locked': 0, 'unknown': 0}
    
    component_prefix = f"components.{component}." if component else ""
    
    for key, source in sources.items():
        if component_prefix and not key.startswith(component_prefix):
            # 只统计该组件的键，或运行时相关配置
            if key not in ['logging', 'performance', 'guards', 'output'] and not key.startswith('logging.') and not key.startswith('performance.'):
                continue
        if source in counts:
            counts[source] += 1
        else:
            counts['unknown'] += 1
    
    print("\n来源统计:")
    for source, count in counts.items():
        if count > 0:
            print(f"  {source}: {count} 个键")

