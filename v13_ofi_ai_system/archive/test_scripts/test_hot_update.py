#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置热更新测试脚本
"""

import sys
import os
sys.path.insert(0, '.')

from src.utils.config_loader import ConfigLoader
from src.ofi_cvd_fusion import OFI_CVD_Fusion
from src.fusion_config_hot_update import create_fusion_hot_updater

def test_hot_update():
    """测试配置热更新"""
    print("=== 配置热更新测试 ===")
    
    try:
        # 创建配置加载器和融合指标实例
        config_loader = ConfigLoader()
        fusion = OFI_CVD_Fusion(config_loader=config_loader)
        
        print("初始配置:")
        print(f"  强买入阈值: {fusion.cfg.fuse_strong_buy}")
        print(f"  强卖出阈值: {fusion.cfg.fuse_strong_sell}")
        
        # 创建热更新器
        hot_updater = create_fusion_hot_updater(
            config_loader=config_loader,
            fusion_instance=fusion
        )
        
        # 添加更新回调
        update_count = [0]
        def on_config_update(new_config):
            update_count[0] += 1
            print(f"配置更新回调 #{update_count[0]}:")
            print(f"  强买入阈值: {new_config.fuse_strong_buy}")
            print(f"  强卖出阈值: {new_config.fuse_strong_sell}")
        
        hot_updater.add_update_callback(on_config_update)
        
        # 模拟配置更新
        print("\n模拟配置更新...")
        success = hot_updater.update_config(force=True)
        
        if success:
            print("[OK] 配置热更新成功")
            
            # 显示更新统计
            stats = hot_updater.get_update_stats()
            print(f"更新统计:")
            print(f"  总更新次数: {stats['total_updates']}")
            print(f"  成功次数: {stats['successful_updates']}")
            print(f"  失败次数: {stats['failed_updates']}")
            
            return True
        else:
            print("[ERROR] 配置热更新失败")
            return False
            
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hot_update()
    sys.exit(0 if success else 1)
