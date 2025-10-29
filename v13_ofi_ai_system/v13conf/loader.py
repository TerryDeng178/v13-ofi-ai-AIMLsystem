"""
配置加载器：实现四层合并 + OFI锁定 + 环境变量映射
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from collections import defaultdict
import copy


def expand_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    将环境变量映射到配置字典中
    
    规则：V13__A__B__C=value → {"A": {"B": {"C": parsed_value}}}
    支持 bool/int/float/list（逗号分隔）自动类型解析
    """
    env_overrides = {}
    
    for key, value in os.environ.items():
        if not key.startswith('V13__'):
            continue
        
        # 移除 V13__ 前缀并分割路径
        path = key[5:].split('__')
        if not path:
            continue
        
        # 解析值类型
        parsed_value = _parse_env_value(value)
        
        # 构建嵌套字典
        current = env_overrides
        for i, segment in enumerate(path[:-1]):
            if segment not in current:
                current[segment] = {}
            current = current[segment]
        current[path[-1]] = parsed_value
    
    # 深合并到配置中
    return _deep_merge(cfg, env_overrides)


def _parse_env_value(value: str) -> Any:
    """解析环境变量值的类型"""
    value = value.strip()
    
    # 布尔值
    if value.lower() in ('true', '1', 'yes', 'on'):
        return True
    if value.lower() in ('false', '0', 'no', 'off'):
        return False
    
    # 整数
    try:
        return int(value)
    except ValueError:
        pass
    
    # 浮点数
    try:
        return float(value)
    except ValueError:
        pass
    
    # 列表（逗号分隔）
    if ',' in value:
        return [v.strip() for v in value.split(',')]
    
    # 字符串
    return value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并两个字典"""
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def _track_sources(cfg: Dict[str, Any], source_name: str, 
                   sources: Dict[str, str], prefix: str = '') -> None:
    """递归追踪每个配置键的来源"""
    for key, value in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            _track_sources(value, source_name, sources, full_key)
        else:
            # 只记录首次出现的来源（优先级从低到高）
            if full_key not in sources:
                sources[full_key] = source_name


def load_config(base_dir: str = "config", 
                allow_env_override_locked: bool = False) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    加载并合并配置
    
    合并顺序：
    1. defaults.yaml （基础默认值）
    2. system.yaml （系统级配置）
    3. overrides.local.yaml （本地覆盖，.gitignore）
    4. 环境变量 V13__* （运行时覆盖）
    5. locked_ofi_params.yaml 中的 OFI 锁定子树（最高优先级，但可被环境变量突破）
    
    返回：
    - cfg: 合并后的配置字典
    - sources: 每个键的来源追踪字典 {key_path: source_name}
    """
    base_path = Path(base_dir)
    sources: Dict[str, str] = {}
    cfg: Dict[str, Any] = {}
    
    # 1. 加载 defaults.yaml
    defaults_path = base_path / "defaults.yaml"
    if defaults_path.exists():
        with open(defaults_path, 'r', encoding='utf-8') as f:
            defaults = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, defaults)
        _track_sources(defaults, 'defaults', sources)
    
    # 2. 加载 system.yaml
    system_path = base_path / "system.yaml"
    if system_path.exists():
        with open(system_path, 'r', encoding='utf-8') as f:
            system = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, system)
        _track_sources(system, 'system', sources)
    
    # 3. 加载 overrides.local.yaml
    overrides_path = base_path / "overrides.local.yaml"
    if overrides_path.exists():
        with open(overrides_path, 'r', encoding='utf-8') as f:
            overrides = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, overrides)
        _track_sources(overrides, 'overrides', sources)
    
    # 4. 应用环境变量
    env_overrides = {}
    # 记录环境变量设置的所有键路径（用于advance检测环境变量是否设置了OFI参数）
    env_key_paths = []
    
    for key, value in os.environ.items():
        if key.upper().startswith('V13__'):  # 大小写不敏感检查
            # 统一转换为小写路径，避免大小写不匹配问题
            path_upper = key[5:].split('__')
            path = [seg.lower() for seg in path_upper]  # 转换为小写
            if path:
                parsed_value = _parse_env_value(value)
                current = env_overrides
                for segment in path[:-1]:
                    if segment not in current:
                        current[segment] = {}
                    current = current[segment]
                current[path[-1]] = parsed_value
                # 记录完整路径（点号分隔，小写）
                env_key_paths.append('.'.join(path))
    
    if env_overrides:
        cfg = _deep_merge(cfg, env_overrides)
        _track_sources(env_overrides, 'env', sources)
    
    # 提取OFI相关的环境变量键路径（统一小写比较）
    ofi_env_keys = [path for path in env_key_paths if path.startswith('components.ofi.')]
    
    # 5. 应用 OFI 锁定参数
    # 优先级说明：
    # - 默认：locked > env > overrides > system > defaults（锁定最高优先级，不允许环境变量突破）
    # - 当 allow_env_override_locked=True 时：env > locked（允许环境变量突破锁定，用于紧急场景）
    locked_path = base_path / "locked_ofi_params.yaml"
    if locked_path.exists():
        with open(locked_path, 'r', encoding='utf-8') as f:
            locked = yaml.safe_load(f) or {}
        
        # 只合并 ofi_calculator 子树
        if 'ofi_calculator' in locked:
            # 确保components.ofi结构存在（环境变量可能已经创建了这个结构）
            if 'components' not in cfg:
                cfg['components'] = {}
            if 'ofi' not in cfg['components']:
                cfg['components']['ofi'] = {}
            
            # 检查OFI相关键是否已被环境变量覆盖
            # 使用在步骤4中记录的ofi_env_keys（最可靠，因为直接来自环境变量解析）
            ofi_keys_from_env = set(ofi_env_keys)  # 转为set提高查找效率
            
            # 应用锁定参数
            for key, value in locked['ofi_calculator'].items():
                full_key = f"components.ofi.{key}"
                
                if allow_env_override_locked:
                    # 方案A：允许环境变量突破锁定
                    # 如果环境变量已设置该键，跳过锁定（保持环境变量的值）
                    if full_key in ofi_keys_from_env:
                        # 环境变量已设置，保持环境变量的值和来源标记不变
                        # cfg中已经包含了环境变量的值，sources也已经标记为env
                        # 重要：不执行任何操作，保持环境变量的值
                        continue  # 跳过这个键，不应用锁定值
                    else:
                        # 环境变量未设置，应用锁定值
                        cfg['components']['ofi'][key] = value
                        # 标记来源为locked（如果之前不是env）
                        if full_key not in sources or sources[full_key] != 'env':
                            sources[full_key] = 'locked'
                else:
                    # 默认方案：锁定优先级最高，环境变量无法突破
                    # 即使环境变量已设置，也强制覆盖为锁定值
                    cfg['components']['ofi'][key] = value
                    sources[full_key] = 'locked'
    
    return cfg, sources


def get_config_value(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """通过点号分隔的路径获取配置值"""
    keys = path.split('.')
    current = cfg
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

