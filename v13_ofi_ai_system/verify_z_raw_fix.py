#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 z_raw 修复效果
"""

import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.real_cvd_calculator import RealCVDCalculator, CVDConfig

def main():
    print("验证 z_raw 修复效果")
    print("=" * 50)
    
    # 创建计算器
    config = CVDConfig(
        z_mode="delta",
        scale_mode="hybrid", 
        mad_multiplier=1.47,
        scale_fast_weight=0.35,
        winsor_limit=8.0
    )
    
    calc = RealCVDCalculator("BTCUSDT", config)
    
    # 模拟交易数据
    trades = [
        (50000.0, 1.0, True),
        (50001.0, 2.0, False), 
        (50002.0, 3.0, True),
        (50003.0, 4.0, False),
        (50004.0, 5.0, True),
    ]
    
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
        
        print(f"\n交易 {i+1}: price={price}, qty={qty}, is_buy={is_buy}")
        print(f"  CVD: {result['cvd']:.4f}")
        print(f"  z_cvd: {result['z_cvd']}")
        print(f"  z_raw: {z_info['z_raw']}")
        print(f"  is_warmup: {z_info['is_warmup']}")
        
        # 验证 Winsorization
        if z_info['z_raw'] is not None and z_info['z_cvd'] is not None:
            z_raw = z_info['z_raw']
            z_cvd = z_info['z_cvd']
            winsor_limit = config.winsor_limit
            
            if abs(z_raw) > winsor_limit:
                expected = winsor_limit if z_raw > 0 else -winsor_limit
                if abs(z_cvd - expected) < 1e-6:
                    print(f"  ✅ 截断正确: |z_raw|={abs(z_raw):.4f} > {winsor_limit}")
                else:
                    print(f"  ❌ 截断错误: 期望{expected:.4f}, 实际{z_cvd:.4f}")
            else:
                if abs(z_raw - z_cvd) < 1e-6:
                    print(f"  ✅ 未截断: |z_raw|={abs(z_raw):.4f} ≤ {winsor_limit}")
                else:
                    print(f"  ❌ 值不同: z_raw={z_raw:.4f}, z_cvd={z_cvd:.4f}")
        
        # 显示尺度诊断
        if 'ewma_fast' in z_stats:
            print(f"  尺度诊断:")
            print(f"    ewma_fast: {z_stats['ewma_fast']:.6f}")
            print(f"    ewma_slow: {z_stats['ewma_slow']:.6f}")
            print(f"    scale: {z_stats['scale']:.6f}")
    
    print("\n" + "=" * 50)
    print("验证完成！")
    
    # 最终检查
    final_z_info = calc.get_last_zscores()
    final_z_stats = calc.get_z_stats()
    
    print(f"\n最终状态:")
    print(f"  z_raw: {final_z_info['z_raw']}")
    print(f"  z_cvd: {final_z_info['z_cvd']}")
    print(f"  is_warmup: {final_z_info['is_warmup']}")
    
    if 'ewma_fast' in final_z_stats:
        print(f"  尺度诊断字段已添加: {list(final_z_stats.keys())}")

if __name__ == "__main__":
    main()
