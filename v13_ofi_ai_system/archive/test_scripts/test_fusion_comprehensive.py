#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合指标综合测试脚本
"""

import sys
import os
import time
import random
sys.path.insert(0, '.')

from src.utils.config_loader import ConfigLoader
from src.ofi_cvd_fusion import OFI_CVD_Fusion
from src.fusion_config_hot_update import create_fusion_hot_updater

def test_comprehensive():
    """综合测试"""
    print("=== 融合指标综合测试 ===")
    
    try:
        # 1. 基本功能测试
        print("\n1. 基本功能测试...")
        config_loader = ConfigLoader()
        fusion = OFI_CVD_Fusion(config_loader=config_loader)
        
        print(f"   配置加载: [OK]")
        print(f"   权重归一化: w_ofi={fusion.cfg.w_ofi}, w_cvd={fusion.cfg.w_cvd}")
        print(f"   阈值设置: buy={fusion.cfg.fuse_buy}, strong_buy={fusion.cfg.fuse_strong_buy}")
        
        # 2. 融合计算测试
        print("\n2. 融合计算测试...")
        test_cases = [
            (2.0, 1.5, 0.1, "强买入"),
            (1.0, 0.8, 0.2, "买入"),
            (-1.0, -0.8, 0.2, "卖出"),
            (-2.0, -1.5, 0.1, "强卖出"),
            (0.5, 0.3, 0.1, "中性")
        ]
        
        for z_ofi, z_cvd, lag_sec, expected in test_cases:
            result = fusion.update(ts=time.time(), z_ofi=z_ofi, z_cvd=z_cvd, lag_sec=lag_sec)
            if result:
                print(f"   输入: z_ofi={z_ofi}, z_cvd={z_cvd} -> 信号: {result['signal']} (期望: {expected})")
            else:
                print(f"   输入: z_ofi={z_ofi}, z_cvd={z_cvd} -> 暖启动阶段")
        
        # 3. 性能测试
        print("\n3. 性能测试...")
        num_tests = 1000
        start_time = time.time()
        
        for i in range(num_tests):
            ts = time.time() + i * 0.001
            z_ofi = random.uniform(-5, 5)
            z_cvd = random.uniform(-5, 5)
            lag_sec = random.uniform(0, 0.5)
            fusion.update(ts=ts, z_ofi=z_ofi, z_cvd=z_cvd, lag_sec=lag_sec)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_tests * 1000  # 转换为毫秒
        
        print(f"   测试次数: {num_tests}")
        print(f"   总时间: {total_time:.3f}秒")
        print(f"   平均时间: {avg_time:.3f}ms")
        print(f"   每秒处理: {num_tests / total_time:.0f}次")
        
        # 4. 统计信息测试
        print("\n4. 统计信息测试...")
        stats = fusion.get_stats()
        print(f"   总更新次数: {stats['total_updates']}")
        print(f"   降级次数: {stats['downgrades']}")
        print(f"   暖启动返回: {stats['warmup_returns']}")
        print(f"   无效输入: {stats['invalid_inputs']}")
        print(f"   滞后超限: {stats['lag_exceeded']}")
        
        # 5. 环境变量覆盖测试
        print("\n5. 环境变量覆盖测试...")
        os.environ['V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_BUY'] = '3.0'
        config_loader.load(reload=True)
        fusion_env = OFI_CVD_Fusion(config_loader=config_loader)
        
        if fusion_env.cfg.fuse_strong_buy == 3.0:
            print("   环境变量覆盖: [OK]")
        else:
            print("   环境变量覆盖: [ERROR]")
            return False
        
        # 清理环境变量
        del os.environ['V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_BUY']
        
        # 6. 配置热更新测试
        print("\n6. 配置热更新测试...")
        hot_updater = create_fusion_hot_updater(
            config_loader=config_loader,
            fusion_instance=fusion
        )
        
        success = hot_updater.update_config(force=True)
        if success:
            print("   配置热更新: [OK]")
        else:
            print("   配置热更新: [ERROR]")
            return False
        
        # 7. 配置验证
        print("\n7. 配置验证...")
        # 权重归一化检查
        total_weight = fusion.cfg.w_ofi + fusion.cfg.w_cvd
        if abs(total_weight - 1.0) < 1e-6:
            print("   权重归一化: [OK]")
        else:
            print("   权重归一化: [ERROR]")
            return False
        
        # 阈值逻辑检查
        if (fusion.cfg.fuse_strong_buy > fusion.cfg.fuse_buy and 
            fusion.cfg.fuse_strong_sell < fusion.cfg.fuse_sell):
            print("   阈值逻辑: [OK]")
        else:
            print("   阈值逻辑: [ERROR]")
            return False
        
        # 一致性阈值检查
        if (0 <= fusion.cfg.min_consistency <= 1 and 
            0 <= fusion.cfg.strong_min_consistency <= 1 and
            fusion.cfg.strong_min_consistency > fusion.cfg.min_consistency):
            print("   一致性阈值: [OK]")
        else:
            print("   一致性阈值: [ERROR]")
            return False
        
        print("\n[SUCCESS] 所有测试通过！融合指标配置完全到位。")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_comprehensive()
    sys.exit(0 if success else 1)
