#!/usr/bin/env python3
"""
调试环境变量覆盖功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader

def debug_env_override():
    """调试环境变量覆盖"""
    print("=== 调试环境变量覆盖功能 ===")
    
    # 设置环境变量
    os.environ['V13__DIVERGENCE_DETECTION__PIVOT_DETECTION__SWING_L'] = '15'
    os.environ['V13__DIVERGENCE_DETECTION__THRESHOLDS__Z_HI'] = '2.0'
    os.environ['V13__DIVERGENCE_DETECTION__DENOISING__COOLDOWN_SECS'] = '2.0'
    os.environ['V13__DIVERGENCE_DETECTION__FUSION__USE_FUSION'] = 'False'
    
    print("设置的环境变量:")
    for key, value in os.environ.items():
        if key.startswith('V13__DIVERGENCE_DETECTION'):
            print(f"  {key} = {value}")
    
    # 创建配置加载器
    config_loader = ConfigLoader()
    
    # 检查原始配置
    print("\n原始配置:")
    swing_l = config_loader.get('divergence_detection.pivot_detection.swing_L')
    z_hi = config_loader.get('divergence_detection.thresholds.z_hi')
    cooldown = config_loader.get('divergence_detection.denoising.cooldown_secs')
    use_fusion = config_loader.get('divergence_detection.fusion.use_fusion')
    
    print(f"  swing_L: {swing_l} (类型: {type(swing_l)})")
    print(f"  z_hi: {z_hi} (类型: {type(z_hi)})")
    print(f"  cooldown_secs: {cooldown} (类型: {type(cooldown)})")
    print(f"  use_fusion: {use_fusion} (类型: {type(use_fusion)})")
    
    # 检查环境变量是否被处理
    print("\n环境变量处理详情:")
    for env_key, env_value in os.environ.items():
        if env_key.startswith('V13__DIVERGENCE_DETECTION'):
            print(f"  处理 {env_key} = {env_value}")
            # 模拟ConfigLoader的处理逻辑
            parts = [p for p in env_key.split("__") if p]
            while parts and parts[0].upper() in ("V13", "CFG", "CONFIG", "OFI", "CVD"):
                parts.pop(0)
            if parts:
                path = [p.lower() for p in parts]
                print(f"    解析路径: {path}")
                print(f"    最终路径: {' -> '.join(path)}")
    
    # 检查环境变量处理
    print("\n环境变量处理:")
    print(f"  V13__DIVERGENCE_DETECTION__PIVOT_DETECTION__SWING_L = {os.getenv('V13__DIVERGENCE_DETECTION__PIVOT_DETECTION__SWING_L')}")
    print(f"  V13__DIVERGENCE_DETECTION__THRESHOLDS__Z_HI = {os.getenv('V13__DIVERGENCE_DETECTION__THRESHOLDS__Z_HI')}")
    print(f"  V13__DIVERGENCE_DETECTION__DENOISING__COOLDOWN_SECS = {os.getenv('V13__DIVERGENCE_DETECTION__DENOISING__COOLDOWN_SECS')}")
    print(f"  V13__DIVERGENCE_DETECTION__FUSION__USE_FUSION = {os.getenv('V13__DIVERGENCE_DETECTION__FUSION__USE_FUSION')}")
    
    # 手动测试_set_by_path方法
    print("\n手动测试_set_by_path方法:")
    from src.utils.config_loader import load_config
    config = load_config()
    
    # 测试swing_L
    path = ['divergence_detection', 'pivot_detection', 'swing_l']
    print(f"  测试路径: {path}")
    print(f"  原始值: {config.get('divergence_detection', {}).get('pivot_detection', {}).get('swing_L')}")
    
    # 直接修改配置字典
    config_loader = ConfigLoader()
    config_loader._set_by_path(config, path, '15')
    print(f"  覆盖后值: {config.get('divergence_detection', {}).get('pivot_detection', {}).get('swing_L')}")
    
    # 检查路径是否存在
    print(f"  路径检查:")
    print(f"    divergence_detection存在: {'divergence_detection' in config}")
    if 'divergence_detection' in config:
        print(f"    pivot_detection存在: {'pivot_detection' in config['divergence_detection']}")
        if 'pivot_detection' in config['divergence_detection']:
            print(f"    swing_l存在: {'swing_l' in config['divergence_detection']['pivot_detection']}")
            print(f"    swing_L存在: {'swing_L' in config['divergence_detection']['pivot_detection']}")

if __name__ == "__main__":
    debug_env_override()
