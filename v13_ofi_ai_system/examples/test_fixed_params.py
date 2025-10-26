#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证修复后的OFI参数效果
按照建议的参数基线测试，验证P(|z|>2)是否回到1%-8%区间
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from real_ofi_calculator import RealOFICalculator, OFIConfig

def test_fixed_params():
    """测试修复后的参数"""
    print("=== 验证修复后的OFI参数 ===")
    print("按照建议的参数基线：禁用z_clip + 放宽winsorize + 降低std_floor")
    
    # 测试高流动性交易对（BTCUSDT）
    print("\n--- 高流动性交易对测试 (BTCUSDT) ---")
    config_high = OFIConfig(
        z_window=80,         # 稳+灵组合：较小窗口
        ema_alpha=0.30,      # 稳+灵组合：较高EMA平滑
        z_clip=999,          # 禁用裁剪用于测试
        std_floor=1e-7,      # 降低标准差下限
        winsorize_ofi_delta=3.0  # 放宽MAD软截
    )
    
    calc_high = RealOFICalculator("BTCUSDT", config_high)
    results_high = test_symbol(calc_high, "BTCUSDT", 1500)
    
    # 测试低流动性交易对（XRPUSDT）
    print("\n--- 低流动性交易对测试 (XRPUSDT) ---")
    config_low = OFIConfig(
        z_window=120,        # 稳+灵组合：较大窗口
        ema_alpha=0.20,      # 稳+灵组合：较低EMA平滑
        z_clip=999,          # 禁用裁剪用于测试
        std_floor=1e-7,      # 降低标准差下限
        winsorize_ofi_delta=3.0  # 放宽MAD软截
    )
    
    calc_low = RealOFICalculator("XRPUSDT", config_low)
    results_low = test_symbol(calc_low, "XRPUSDT", 1500)
    
    # 对比分析
    print("\n=== 修复效果对比 ===")
    print(f"高流动性 (z_clip=999): IQR={results_high['iqr']:.3f}, P(|z|>2)={results_high['p_gt_2']:.2f}%, 通过率={results_high['pass_rate']:.1f}%")
    print(f"低流动性 (z_clip=999): IQR={results_low['iqr']:.3f}, P(|z|>2)={results_low['p_gt_2']:.2f}%, 通过率={results_low['pass_rate']:.1f}%")
    
    # 总体评价
    avg_pass_rate = (results_high['pass_rate'] + results_low['pass_rate']) / 2
    avg_p_gt_2 = (results_high['p_gt_2'] + results_low['p_gt_2']) / 2
    
    print(f"\n平均通过率: {avg_pass_rate:.1f}%")
    print(f"平均P(|z|>2): {avg_p_gt_2:.2f}%")
    
    # 修复效果评估
    if avg_p_gt_2 >= 1.0 and avg_p_gt_2 <= 8.0:
        print("SUCCESS 修复成功！P(|z|>2)回到1%-8%验收区间")
    elif avg_pass_rate >= 90:
        print("GOOD 修复有效，通过率显著提升")
    elif avg_pass_rate >= 80:
        print("GOOD 修复有效，通过率保持稳定")
    else:
        print("WARN 修复需要进一步调整")
    
    # 与之前结果对比
    print(f"\n=== 修复前后对比 ===")
    print("修复前: IQR=1.427, P(|z|>2)=0.00%, 通过率=75.0%")
    print(f"修复后: IQR={(results_high['iqr']+results_low['iqr'])/2:.3f}, P(|z|>2)={avg_p_gt_2:.2f}%, 通过率={avg_pass_rate:.1f}%")
    
    # 验收标准检查
    print(f"\n=== 验收标准检查 ===")
    print(f"中位数居中: 高流动性={results_high['median']:.3f}, 低流动性={results_low['median']:.3f}")
    print(f"IQR合理: 高流动性={results_high['iqr']:.3f}, 低流动性={results_low['iqr']:.3f}")
    print(f"P(|z|>2): 高流动性={results_high['p_gt_2']:.2f}%, 低流动性={results_low['p_gt_2']:.2f}%")
    print(f"P(|z|>3): 高流动性={results_high['p_gt_3']:.2f}%, 低流动性={results_low['p_gt_3']:.2f}%")

def test_symbol(calc, symbol, num_points):
    """测试单个交易对"""
    np.random.seed(42)
    results = []
    
    # 根据交易对调整价格波动范围
    if symbol == "BTCUSDT":
        price_volatility = 25  # 高流动性，适中价格波动
    else:
        price_volatility = 35  # 低流动性，较大价格波动
    
    for i in range(num_points):
        # 模拟订单簿快照
        bids = [
            [50000 + np.random.normal(0, price_volatility), np.random.exponential(5)],
            [49999 + np.random.normal(0, price_volatility), np.random.exponential(3)],
            [49998 + np.random.normal(0, price_volatility), np.random.exponential(2)],
            [49997 + np.random.normal(0, price_volatility), np.random.exponential(1)],
            [49996 + np.random.normal(0, price_volatility), np.random.exponential(0.5)]
        ]
        
        asks = [
            [50001 + np.random.normal(0, price_volatility), np.random.exponential(5)],
            [50002 + np.random.normal(0, price_volatility), np.random.exponential(3)],
            [50003 + np.random.normal(0, price_volatility), np.random.exponential(2)],
            [50004 + np.random.normal(0, price_volatility), np.random.exponential(1)],
            [50005 + np.random.normal(0, price_volatility), np.random.exponential(0.5)]
        ]
        
        result = calc.update_with_snapshot(bids, asks, event_time_ms=i*1000)
        results.append(result)
    
    # 分析Z-score分布
    z_scores = [r['z_ofi'] for r in results if r['z_ofi'] is not None]
    
    if len(z_scores) > 0:
        median = np.median(z_scores)
        iqr = np.percentile(z_scores, 75) - np.percentile(z_scores, 25)
        p_gt_2 = np.mean(np.abs(z_scores) > 2) * 100
        p_gt_3 = np.mean(np.abs(z_scores) > 3) * 100
        
        print(f"有效Z-score数据点: {len(z_scores)}")
        print(f"中位数: {median:.6f}")
        print(f"IQR: {iqr:.6f}")
        print(f"P(|z| > 2): {p_gt_2:.2f}%")
        print(f"P(|z| > 3): {p_gt_3:.2f}%")
        
        # 验收标准检查
        print("\n--- 验收标准检查 ---")
        
        # 中位数居中
        median_ok = -0.1 <= median <= 0.1
        print(f"中位数居中: {'OK' if median_ok else 'FAIL'}")
        
        # IQR合理
        iqr_ok = 0.8 <= iqr <= 1.6
        print(f"IQR合理: {'OK' if iqr_ok else 'FAIL'}")
        
        # 尾部占比
        tail_2_ok = 1 <= p_gt_2 <= 8
        print(f"P(|z| > 2): {'OK' if tail_2_ok else 'FAIL'}")
        
        tail_3_ok = p_gt_3 <= 1.5
        print(f"P(|z| > 3): {'OK' if tail_3_ok else 'FAIL'}")
        
        # 计算通过率
        passed = sum([median_ok, iqr_ok, tail_2_ok, tail_3_ok])
        pass_rate = passed / 4 * 100
        
        print(f"通过率: {passed}/4 ({pass_rate:.1f}%)")
        
        # 显示尾部监控数据
        last_result = results[-1]
        if 'meta' in last_result:
            meta = last_result['meta']
            print(f"尾部监控: P(|z|>2)={meta.get('p_gt2_percent', 0):.2f}%, P(|z|>3)={meta.get('p_gt3_percent', 0):.2f}%")
        
        return {
            'median': median,
            'iqr': iqr,
            'p_gt_2': p_gt_2,
            'p_gt_3': p_gt_3,
            'pass_rate': pass_rate
        }
    else:
        print("FAIL 没有有效的Z-score数据")
        return {'pass_rate': 0}

if __name__ == "__main__":
    test_fixed_params()
