#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Winsorization 效果
验证 z_raw 和 z_cvd 的截断关系
"""

import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.real_cvd_calculator import RealCVDCalculator, CVDConfig

def main():
    print("测试 Winsorization 效果")
    print("=" * 50)
    
    # 创建计算器，使用较小的winsor_limit便于测试
    config = CVDConfig(
        z_mode="delta",
        scale_mode="hybrid", 
        mad_multiplier=1.47,
        scale_fast_weight=0.35,
        winsor_limit=2.0,  # 较小的截断阈值便于观察
        freeze_min=10      # 较小的warmup阈值
    )
    
    calc = RealCVDCalculator("BTCUSDT", config)
    
    # 模拟更多交易数据，确保超过warmup阈值
    trades = []
    for i in range(20):
        price = 50000 + i
        qty = 1.0 + i * 0.1  # 逐渐增大的数量
        is_buy = i % 2 == 0  # 交替买卖
        trades.append((price, qty, is_buy))
    
    print("处理测试交易...")
    for i, (price, qty, is_buy) in enumerate(trades):
        result = calc.update_with_trade(
            price=price,
            qty=qty, 
            is_buy=is_buy,
            event_time_ms=int(time.time() * 1000)
        )
        
        z_info = calc.get_last_zscores()
        z_stats = calc.get_z_stats()
        
        if i >= 10:  # 只显示warmup后的交易
            print(f"\n交易 {i+1}: price={price}, qty={qty:.1f}, is_buy={is_buy}")
            print(f"  CVD: {result['cvd']:.4f}")
            print(f"  z_cvd: {result['z_cvd']}")
            print(f"  z_raw: {z_info['z_raw']}")
            print(f"  is_warmup: {z_info['is_warmup']}")
            
            # 验证 Winsorization
            if z_info['z_raw'] is not None and z_info['z_cvd'] is not None:
                z_raw = z_info['z_raw']
                z_cvd = z_info['z_cvd']
                winsor_limit = config.winsor_limit
                
                print(f"  验证截断:")
                print(f"    |z_raw| = {abs(z_raw):.4f}")
                print(f"    |z_cvd| = {abs(z_cvd):.4f}")
                print(f"    winsor_limit = {winsor_limit}")
                
                if abs(z_raw) > winsor_limit:
                    expected = winsor_limit if z_raw > 0 else -winsor_limit
                    if abs(z_cvd - expected) < 1e-6:
                        print(f"    截断正确: |z_raw|={abs(z_raw):.4f} > {winsor_limit} -> z_cvd={z_cvd:.4f}")
                    else:
                        print(f"    截断错误: 期望{expected:.4f}, 实际{z_cvd:.4f}")
                else:
                    if abs(z_raw - z_cvd) < 1e-6:
                        print(f"    未截断: |z_raw|={abs(z_raw):.4f} <= {winsor_limit}")
                    else:
                        print(f"    值不同: z_raw={z_raw:.4f}, z_cvd={z_cvd:.4f}")
            
            # 显示尺度诊断
            if 'ewma_fast' in z_stats:
                print(f"  尺度诊断:")
                print(f"    ewma_fast: {z_stats['ewma_fast']:.6f}")
                print(f"    ewma_slow: {z_stats['ewma_slow']:.6f}")
                print(f"    scale: {z_stats['scale']:.6f}")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    
    # 最终检查
    final_z_info = calc.get_last_zscores()
    final_z_stats = calc.get_z_stats()
    
    print(f"\n最终状态:")
    print(f"  z_raw: {final_z_info['z_raw']}")
    print(f"  z_cvd: {final_z_info['z_cvd']}")
    print(f"  is_warmup: {final_z_info['is_warmup']}")
    
    if 'ewma_fast' in final_z_stats:
        print(f"  尺度诊断字段: {list(final_z_stats.keys())}")

if __name__ == "__main__":
    main()
