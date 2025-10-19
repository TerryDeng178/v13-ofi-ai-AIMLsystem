#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境变量覆盖测试脚本
"""

import sys
import os
sys.path.insert(0, '.')

from src.utils.config_loader import ConfigLoader
from src.ofi_cvd_fusion import OFI_CVD_Fusion

def test_env_override():
    """测试环境变量覆盖"""
    print("=== 环境变量覆盖测试 ===")
    
    try:
        # 设置环境变量
        os.environ['V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_BUY'] = '3.0'
        os.environ['V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_SELL'] = '-3.0'
        
        # 创建配置加载器并重新加载
        config_loader = ConfigLoader()
        config_loader.load(reload=True)
        
        # 创建融合指标实例
        fusion = OFI_CVD_Fusion(config_loader=config_loader)
        
        print("环境变量覆盖后的配置:")
        print(f"  强买入阈值: {fusion.cfg.fuse_strong_buy}")
        print(f"  强卖出阈值: {fusion.cfg.fuse_strong_sell}")
        
        # 验证覆盖是否生效
        if fusion.cfg.fuse_strong_buy == 3.0 and fusion.cfg.fuse_strong_sell == -3.0:
            print("[OK] 环境变量覆盖成功")
            return True
        else:
            print("[ERROR] 环境变量覆盖失败")
            return False
            
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        return False
    finally:
        # 清理环境变量
        if 'V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_BUY' in os.environ:
            del os.environ['V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_BUY']
        if 'V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_SELL' in os.environ:
            del os.environ['V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_SELL']

if __name__ == "__main__":
    success = test_env_override()
    sys.exit(0 if success else 1)
