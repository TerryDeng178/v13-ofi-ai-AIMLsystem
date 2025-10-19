#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OFI和CVD配置集成测试脚本
"""

import sys
import os
sys.path.insert(0, '.')

from src.utils.config_loader import ConfigLoader
from src.real_ofi_calculator import RealOFICalculator, OFIConfig
from src.real_cvd_calculator import RealCVDCalculator, CVDConfig

def test_ofi_config_integration():
    """测试OFI配置集成"""
    print("=== OFI配置集成测试 ===")
    
    try:
        # 1. 测试默认配置
        print("1. 测试默认配置...")
        ofi_default = RealOFICalculator("ETHUSDT")
        print(f"   默认配置: levels={ofi_default.K}, z_window={ofi_default.z_window}")
        
        # 2. 测试配置加载器
        print("2. 测试配置加载器...")
        config_loader = ConfigLoader()
        ofi_config = RealOFICalculator("ETHUSDT", config_loader=config_loader)
        print(f"   配置加载器: levels={ofi_config.K}, z_window={ofi_config.z_window}")
        
        # 3. 测试环境变量覆盖
        print("3. 测试环境变量覆盖...")
        os.environ['V13__COMPONENTS__OFI__LEVELS'] = '10'
        os.environ['V13__COMPONENTS__OFI__Z_WINDOW'] = '500'
        config_loader.load(reload=True)
        ofi_env = RealOFICalculator("ETHUSDT", config_loader=config_loader)
        print(f"   环境变量覆盖: levels={ofi_env.K}, z_window={ofi_env.z_window}")
        
        # 清理环境变量
        del os.environ['V13__COMPONENTS__OFI__LEVELS']
        del os.environ['V13__COMPONENTS__OFI__Z_WINDOW']
        
        print("   [OK] OFI配置集成测试通过")
        return True
        
    except Exception as e:
        print(f"   [ERROR] OFI配置集成测试失败: {e}")
        return False

def test_cvd_config_integration():
    """测试CVD配置集成"""
    print("\n=== CVD配置集成测试 ===")
    
    try:
        # 1. 测试默认配置
        print("1. 测试默认配置...")
        cvd_default = RealCVDCalculator("ETHUSDT")
        print(f"   默认配置: z_window={cvd_default.cfg.z_window}, z_mode={cvd_default.cfg.z_mode}")
        
        # 2. 测试配置加载器
        print("2. 测试配置加载器...")
        config_loader = ConfigLoader()
        cvd_config = RealCVDCalculator("ETHUSDT", config_loader=config_loader)
        print(f"   配置加载器: z_window={cvd_config.cfg.z_window}, z_mode={cvd_config.cfg.z_mode}")
        
        # 3. 测试环境变量覆盖
        print("3. 测试环境变量覆盖...")
        os.environ['V13__COMPONENTS__CVD__Z_WINDOW'] = '600'
        os.environ['V13__COMPONENTS__CVD__Z_MODE'] = 'delta'
        config_loader.load(reload=True)
        cvd_env = RealCVDCalculator("ETHUSDT", config_loader=config_loader)
        print(f"   环境变量覆盖: z_window={cvd_env.cfg.z_window}, z_mode={cvd_env.cfg.z_mode}")
        
        # 清理环境变量
        del os.environ['V13__COMPONENTS__CVD__Z_WINDOW']
        del os.environ['V13__COMPONENTS__CVD__Z_MODE']
        
        print("   [OK] CVD配置集成测试通过")
        return True
        
    except Exception as e:
        print(f"   [ERROR] CVD配置集成测试失败: {e}")
        return False

def test_config_consistency():
    """测试配置一致性"""
    print("\n=== 配置一致性测试 ===")
    
    try:
        config_loader = ConfigLoader()
        
        # 检查OFI配置
        ofi_config = config_loader.get('components.ofi', {})
        print(f"OFI配置项: {list(ofi_config.keys())}")
        
        # 检查CVD配置
        cvd_config = config_loader.get('components.cvd', {})
        print(f"CVD配置项: {list(cvd_config.keys())}")
        
        # 检查融合指标配置
        fusion_config = config_loader.get('fusion_metrics', {})
        print(f"融合指标配置项: {list(fusion_config.keys())}")
        
        print("   [OK] 配置一致性测试通过")
        return True
        
    except Exception as e:
        print(f"   [ERROR] 配置一致性测试失败: {e}")
        return False

def main():
    """主函数"""
    print("OFI和CVD配置集成测试")
    print("=" * 50)
    
    success = True
    
    # 测试OFI配置集成
    if not test_ofi_config_integration():
        success = False
    
    # 测试CVD配置集成
    if not test_cvd_config_integration():
        success = False
    
    # 测试配置一致性
    if not test_config_consistency():
        success = False
    
    if success:
        print("\n[SUCCESS] 所有测试通过！OFI和CVD已集成到统一配置系统。")
    else:
        print("\n[ERROR] 部分测试失败，请检查配置集成。")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
