#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出有效配置脚本
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.unified_config_loader import UnifiedConfigLoader

if __name__ == "__main__":
    loader = UnifiedConfigLoader()
    output_file = Path(__file__).parent.parent / "reports" / "effective-config.json"
    loader.dump_effective(str(output_file))
    print(f"Exported effective config to: {output_file}")

