#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_delta_z.py - P1.1 Delta-Z功能测试脚本

功能：
- 测试Delta-Z vs Level-Z两种模式
- 验证Z-score分布改善
- 对比分析结果

使用方法：
    # 测试Delta-Z模式
    python test_delta_z.py --mode delta
    
    # 测试Level-Z模式（基线）
    python test_delta_z.py --mode level
    
    # 对比两种模式
    python test_delta_z.py --compare
"""

import sys
import os
import io
from pathlib import Path

# Windows Unicode支持
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目路径
here = Path(__file__).resolve().parent.parent
src_dir = here / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from real_cvd_calculator import RealCVDCalculator, CVDConfig
import numpy as np
import matplotlib.pyplot as plt

def test_delta_z_mode():
    """测试Delta-Z模式"""
    print("🧪 测试Delta-Z模式...")
    
    # 创建Delta-Z配置
    cfg = CVDConfig(
        z_mode="delta",
        half_life_trades=50,  # 较短半衰期便于测试
        winsor_limit=8.0,
        freeze_min=10,  # 较短暖启动期
        stale_threshold_ms=5000,
    )
    
    calc = RealCVDCalculator("TEST", cfg)
    
    # 模拟交易数据（增加交易数量以触发Z-score计算）
    trades = []
    for i in range(20):  # 20笔交易，超过freeze_min=10
        price = 100.0 + i * 0.1
        qty = 1.0 + i * 0.1
        is_buy = i % 2 == 0  # 交替买卖
        ts = 1000 + i * 100
        trades.append((price, qty, is_buy, ts))
    
    results = []
    for price, qty, is_buy, ts in trades:
        result = calc.update_with_trade(price=price, qty=qty, is_buy=is_buy, event_time_ms=ts)
        results.append(result)
        print(f"  CVD={result['cvd']:.2f}, Z={result['z_cvd']}, Delta={result['meta'].get('delta', 'N/A')}")
    
    # 获取Z统计信息
    z_stats = calc.get_z_stats()
    print(f"  Z统计: {z_stats}")
    
    return results, z_stats

def test_level_z_mode():
    """测试Level-Z模式（基线）"""
    print("🧪 测试Level-Z模式（基线）...")
    
    # 创建Level-Z配置
    cfg = CVDConfig(
        z_mode="level",
        z_window=50,
        warmup_min=10,
    )
    
    calc = RealCVDCalculator("TEST", cfg)
    
    # 模拟相同的交易数据（增加交易数量以触发Z-score计算）
    trades = []
    for i in range(20):  # 20笔交易，超过warmup_min=10
        price = 100.0 + i * 0.1
        qty = 1.0 + i * 0.1
        is_buy = i % 2 == 0  # 交替买卖
        ts = 1000 + i * 100
        trades.append((price, qty, is_buy, ts))
    
    results = []
    for price, qty, is_buy, ts in trades:
        result = calc.update_with_trade(price=price, qty=qty, is_buy=is_buy, event_time_ms=ts)
        results.append(result)
        print(f"  CVD={result['cvd']:.2f}, Z={result['z_cvd']}")
    
    # 获取Z统计信息
    z_stats = calc.get_z_stats()
    print(f"  Z统计: {z_stats}")
    
    return results, z_stats

def compare_modes():
    """对比两种模式"""
    print("🔍 对比Delta-Z vs Level-Z模式...")
    
    # 测试Delta-Z
    delta_results, delta_stats = test_delta_z_mode()
    print()
    
    # 测试Level-Z
    level_results, level_stats = test_level_z_mode()
    print()
    
    # 对比分析
    print("📊 对比分析:")
    print(f"  Delta-Z模式: Z值={[r['z_cvd'] for r in delta_results if r['z_cvd'] is not None]}")
    print(f"  Level-Z模式: Z值={[r['z_cvd'] for r in level_results if r['z_cvd'] is not None]}")
    
    # 计算Z-score统计
    delta_zs = [r['z_cvd'] for r in delta_results if r['z_cvd'] is not None]
    level_zs = [r['z_cvd'] for r in level_results if r['z_cvd'] is not None]
    
    if delta_zs and level_zs:
        print(f"  Delta-Z: median|Z|={np.median(np.abs(delta_zs)):.3f}, P(|Z|>2)={np.mean(np.abs(delta_zs) > 2)*100:.1f}%")
        print(f"  Level-Z: median|Z|={np.median(np.abs(level_zs)):.3f}, P(|Z|>2)={np.mean(np.abs(level_zs) > 2)*100:.1f}%")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="P1.1 Delta-Z功能测试")
    parser.add_argument("--mode", choices=["delta", "level"], help="测试模式")
    parser.add_argument("--compare", action="store_true", help="对比两种模式")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_modes()
    elif args.mode == "delta":
        test_delta_z_mode()
    elif args.mode == "level":
        test_level_z_mode()
    else:
        print("请指定 --mode delta/level 或 --compare")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
