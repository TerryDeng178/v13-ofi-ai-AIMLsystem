"""
V13 统一配置管理系统
实现单源分层配置：defaults.yaml < system.yaml < overrides.local.yaml < 环境变量 < OFI锁定
"""

from .loader import load_config, expand_env
from .normalizer import normalize
from .invariants import validate_invariants
from .packager import build_runtime_pack, save_runtime_pack
from .unconsumed_keys import check_unconsumed_keys
from .strict_mode import load_strict_runtime_config
from .runtime_loader import load_component_runtime_config, print_component_effective_config

__all__ = [
    'load_config',
    'expand_env',
    'normalize',
    'validate_invariants',
    'build_runtime_pack',
    'save_runtime_pack',
    'check_unconsumed_keys',
    'load_strict_runtime_config',
    'load_component_runtime_config',
    'print_component_effective_config',
]

__version__ = '1.0.0'

