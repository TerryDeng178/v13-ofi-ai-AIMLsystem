#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 z_raw 修复效果
验证 z_raw 是真·未截断值且与 z_cvd 同一口径
"""

import sys
import time
from pathlib import Path

# 确保项目根目录在sys.path中
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.real_cvd_calculator import RealCVDCalculator, CVDConfig

def test_z_raw_fix():
    """测试 z_raw 修复效果"""
    print("🧪 测试 z_raw 修复效果")
    print("=" * 50)
    
    # 创建计算器，使用 delta 模式
    config = CVDConfig(
        z_mode="delta",
        scale_mode="hybrid",
        mad_multiplier=1.47,
        scale_fast_weight=0.35,
        winsor_limit=8.0
    )
    
    calc = RealCVDCalculator("BTCUSDT", config)
    
    # 模拟一些交易数据
    test_trades = [
        (50000.0, 1.0, True, int(time.time() * 1000)),   # 买入
        (50001.0, 2.0, False, int(time.time() * 1000)),  # 卖出
        (50002.0, 3.0, True, int(time.time() * 1000)),   # 买入
        (50003.0, 4.0, False, int(time.time() * 1000)),  # 卖出
        (50004.0, 5.0, True, int(time.time() * 1000)),   # 买入
    ]
    
    print("📊 处理测试交易...")
    for i, (price, qty, is_buy, event_time) in enumerate(test_trades):
        result = calc.update_with_trade(
            price=price, 
            qty=qty, 
            is_buy=is_buy, 
            event_time_ms=event_time
        )
        
        # 获取 Z-score 信息
        z_info = calc.get_last_zscores()
        z_stats = calc.get_z_stats()
        
        print(f"\n交易 {i+1}: price={price}, qty={qty}, is_buy={is_buy}")
        print(f"  CVD: {result['cvd']:.4f}")
        print(f"  z_cvd: {result['z_cvd']}")
        print(f"  z_raw: {z_info['z_raw']}")
        print(f"  z_post: {z_info['z_cvd']}")
        print(f"  is_warmup: {z_info['is_warmup']}")
        print(f"  is_flat: {z_info['is_flat']}")
        
        # 验证 z_raw 和 z_cvd 的关系
        if z_info['z_raw'] is not None and z_info['z_cvd'] is not None:
            winsor_limit = config.winsor_limit
            z_raw = z_info['z_raw']
            z_cvd = z_info['z_cvd']
            
            print(f"  ✅ 验证 Winsorization:")
            print(f"    |z_raw| = {abs(z_raw):.4f}")
            print(f"    |z_cvd| = {abs(z_cvd):.4f}")
            print(f"    winsor_limit = {winsor_limit}")
            
            if abs(z_raw) > winsor_limit:
                expected_cvd = winsor_limit if z_raw > 0 else -winsor_limit
                if abs(z_cvd - expected_cvd) < 1e-6:
                    print(f"    ✅ 截断正确: |z_raw|={abs(z_raw):.4f} > {winsor_limit} → z_cvd={z_cvd:.4f}")
                else:
                    print(f"    ❌ 截断错误: 期望 {expected_cvd:.4f}, 实际 {z_cvd:.4f}")
            else:
                if abs(z_raw - z_cvd) < 1e-6:
                    print(f"    ✅ 未截断: |z_raw|={abs(z_raw):.4f} ≤ {winsor_limit} → z_cvd=z_raw")
                else:
                    print(f"    ❌ 未截断但值不同: z_raw={z_raw:.4f}, z_cvd={z_cvd:.4f}")
        
        # 显示尺度诊断信息
        if 'ewma_fast' in z_stats:
            print(f"  📈 尺度诊断:")
            print(f"    ewma_fast: {z_stats.get('ewma_fast', 0):.6f}")
            print(f"    ewma_slow: {z_stats.get('ewma_slow', 0):.6f}")
            print(f"    ewma_mix: {z_stats.get('ewma_mix', 0):.6f}")
            print(f"    sigma_floor: {z_stats.get('sigma_floor', 0):.6f}")
            print(f"    scale: {z_stats.get('scale', 0):.6f}")
    
    print("\n" + "=" * 50)
    print("🎯 测试完成！")
    
    # 最终状态检查
    final_z_info = calc.get_last_zscores()
    final_z_stats = calc.get_z_stats()
    
    print(f"\n📋 最终状态:")
    print(f"  z_raw: {final_z_info['z_raw']}")
    print(f"  z_cvd: {final_z_info['z_cvd']}")
    print(f"  is_warmup: {final_z_info['is_warmup']}")
    print(f"  is_flat: {final_z_info['is_flat']}")
    
    if 'ewma_fast' in final_z_stats:
        print(f"  尺度诊断字段: {list(final_z_stats.keys())}")
        print(f"  ✅ 尺度诊断信息已添加到 get_z_stats()")

if __name__ == "__main__":
    test_z_raw_fix()
