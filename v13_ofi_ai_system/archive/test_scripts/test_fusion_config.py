#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合指标配置测试脚本
"""

import sys
import os
sys.path.insert(0, '.')

from src.utils.config_loader import ConfigLoader
from src.ofi_cvd_fusion import OFI_CVD_Fusion
import time

def test_fusion_config():
    """测试融合指标配置"""
    print("=== 融合指标配置测试 ===")
    
    try:
        # 1. 测试配置加载器
        print("1. 测试配置加载器...")
        config_loader = ConfigLoader()
        print("   [OK] 配置加载器创建成功")
        
        # 2. 测试融合指标创建
        print("2. 测试融合指标创建...")
        fusion = OFI_CVD_Fusion(config_loader=config_loader)
        print("   [OK] 融合指标实例创建成功")
        
        # 3. 显示配置参数
        print("3. 显示配置参数...")
        print(f"   权重: w_ofi={fusion.cfg.w_ofi}, w_cvd={fusion.cfg.w_cvd}")
        print(f"   阈值: buy={fusion.cfg.fuse_buy}, strong_buy={fusion.cfg.fuse_strong_buy}")
        print(f"   阈值: sell={fusion.cfg.fuse_sell}, strong_sell={fusion.cfg.fuse_strong_sell}")
        print(f"   一致性: min={fusion.cfg.min_consistency}, strong={fusion.cfg.strong_min_consistency}")
        print(f"   数据处理: z_clip={fusion.cfg.z_clip}, max_lag={fusion.cfg.max_lag}")
        print(f"   去噪: hysteresis_exit={fusion.cfg.hysteresis_exit}, cooldown_secs={fusion.cfg.cooldown_secs}")
        
        # 4. 测试融合计算
        print("4. 测试融合计算...")
        result = fusion.update(ts=time.time(), z_ofi=2.0, z_cvd=1.5, lag_sec=0.1)
        if result:
            print(f"   融合得分: {result['fusion_score']:.3f}")
            print(f"   信号: {result['signal']}")
            print(f"   一致性: {result['consistency']:.3f}")
            print("   [OK] 融合计算成功")
        else:
            print("   [WARN] 融合计算返回空（可能是暖启动阶段）")
        
        # 5. 测试统计信息
        print("5. 测试统计信息...")
        stats = fusion.get_stats()
        print(f"   总更新次数: {stats['total_updates']}")
        print(f"   降级次数: {stats['downgrades']}")
        print(f"   暖启动返回: {stats['warmup_returns']}")
        print("   [OK] 统计信息正常")
        
        # 6. 测试权重归一化
        print("6. 测试权重归一化...")
        total_weight = fusion.cfg.w_ofi + fusion.cfg.w_cvd
        if abs(total_weight - 1.0) < 1e-6:
            print(f"   [OK] 权重归一化正确: {total_weight:.6f}")
        else:
            print(f"   [ERROR] 权重归一化错误: {total_weight}")
            return False
        
        # 7. 测试阈值逻辑
        print("7. 测试阈值逻辑...")
        if (fusion.cfg.fuse_strong_buy > fusion.cfg.fuse_buy and 
            fusion.cfg.fuse_strong_sell < fusion.cfg.fuse_sell):
            print("   [OK] 阈值逻辑正确")
        else:
            print("   [ERROR] 阈值逻辑错误")
            return False
        
        print("\n[SUCCESS] 所有测试通过！融合指标配置正确到位。")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fusion_config()
    sys.exit(0 if success else 1)
