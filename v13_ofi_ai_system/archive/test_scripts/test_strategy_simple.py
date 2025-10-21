#!/usr/bin/env python3
"""
简化的策略模式管理器测试
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.utils.strategy_mode_manager import StrategyModeManager

def test_simple():
    """简单测试"""
    print("=== 简单策略模式管理器测试 ===")
    
    try:
        # 创建配置加载器
        config_loader = ConfigLoader()
        print("配置加载器创建成功")
        
        # 创建策略模式管理器
        print("创建策略模式管理器...")
        manager = StrategyModeManager(config_loader=config_loader)
        print("策略模式管理器创建成功")
        
        # 检查基础属性
        print(f"config: {manager.config}")
        print(f"strategy_config: {manager.strategy_config}")
        print(f"mode_setting: {manager.mode_setting}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple()
