
# -*- coding: utf-8 -*-
"""
兼容层：保持 `from utils.config_loader import load_config, get_config` 可用。
实际实现委托给 `config.unified_config_loader`。
"""
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.unified_config_loader import load_config as _load, UnifiedConfigLoader as UnifiedConfig, UnifiedConfigLoader as ConfigLoader

_singleton = None

def load_config(base_dir: str = None):
    global _singleton
    _singleton = _load(base_dir or Path(__file__).resolve().parents[2] / "config")
    return _singleton

def get_config():
    return _singleton or load_config()
