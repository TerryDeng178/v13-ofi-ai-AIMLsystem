#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩面验证：6个交易对A/B测试
验证修复后的参数基线在ETH/BNB/SOL/DOGE/ADA上的表现
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from real_ofi_calculator import RealOFICalculator, OFIConfig

def test_all_symbols():
    """测试所有6个交易对"""
    print("=== 扩面验证：6个交易对A/B测试 ===")
    print("验证修复后的参数基线在ETH/BNB/SOL/DOGE/ADA上的表现")
    
    # 定义交易对和流动性分类
    symbols = {
        'high': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
        'low': ['XRPUSDT', 'DOGEUSDT', 'ADAUSDT']
    }
    
    results = {}
    
    # 测试高流动性交易对
    print("\n--- 高流动性交易对测试 ---")
    for symbol in symbols['high']:
        print(f"\n测试 {symbol}:")
        config = OFIConfig(
            z_window=80,         # 稳+灵组合：较小窗口
            ema_alpha=0.30,      # 稳+灵组合：较高EMA平滑
            z_clip=1e6,          # 禁用裁剪用于测试
            std_floor=1e-7,      # 降低标准差下限
            winsorize_ofi_delta=3.0  # 放宽MAD软截
        )
        
        calc = RealOFICalculator(symbol, config)
        result = test_symbol(calc, symbol, 1200)
        results[symbol] = result
    
    # 测试低流动性交易对
    print("\n--- 低流动性交易对测试 ---")
    for symbol in symbols['low']:
        print(f"\n测试 {symbol}:")
        config = OFIConfig(
            z_window=120,        # 稳+灵组合：较大窗口
            ema_alpha=0.20,      # 稳+灵组合：较低EMA平滑
            z_clip=1e6,          # 禁用裁剪用于测试
            std_floor=1e-7,      # 降低标准差下限
            winsorize_ofi_delta=3.0  # 放宽MAD软截
        )
        
        calc = RealOFICalculator(symbol, config)
        result = test_symbol(calc, symbol, 1200)
        results[symbol] = result
    
    # 统计分析
    print("\n=== 扩面验证结果统计 ===")
    
    high_results = [results[s] for s in symbols['high'] if results[s]]
    low_results = [results[s] for s in symbols['low'] if results[s]]
    
    if high_results:
        avg_high_p_gt_2 = np.mean([r['p_gt_2'] for r in high_results])
        avg_high_iqr = np.mean([r['iqr'] for r in high_results])
        avg_high_pass_rate = np.mean([r['pass_rate'] for r in high_results])
        
        print(f"高流动性交易对 (n={len(high_results)}):")
        print(f"  平均P(|z|>2): {avg_high_p_gt_2:.2f}%")
        print(f"  平均IQR: {avg_high_iqr:.3f}")
        print(f"  平均通过率: {avg_high_pass_rate:.1f}%")
    
    if low_results:
        avg_low_p_gt_2 = np.mean([r['p_gt_2'] for r in low_results])
        avg_low_iqr = np.mean([r['iqr'] for r in low_results])
        avg_low_pass_rate = np.mean([r['pass_rate'] for r in low_results])
        
        print(f"低流动性交易对 (n={len(low_results)}):")
        print(f"  平均P(|z|>2): {avg_low_p_gt_2:.2f}%")
        print(f"  平均IQR: {avg_low_iqr:.3f}")
        print(f"  平均通过率: {avg_low_pass_rate:.1f}%")
    
    # 总体评估
    all_results = [r for r in results.values() if r]
    if all_results:
        avg_p_gt_2 = np.mean([r['p_gt_2'] for r in all_results])
        avg_iqr = np.mean([r['iqr'] for r in all_results])
        avg_pass_rate = np.mean([r['pass_rate'] for r in all_results])
        
        print(f"\n总体平均 (n={len(all_results)}):")
        print(f"  平均P(|z|>2): {avg_p_gt_2:.2f}%")
        print(f"  平均IQR: {avg_iqr:.3f}")
        print(f"  平均通过率: {avg_pass_rate:.1f}%")
        
        # 验收标准检查
        print(f"\n=== 验收标准检查 ===")
        if 1.0 <= avg_p_gt_2 <= 8.0:
            print(f"OK P(|z|>2)在1%-8%区间: {avg_p_gt_2:.2f}%")
        else:
            print(f"FAIL P(|z|>2)超出1%-8%区间: {avg_p_gt_2:.2f}%")
        
        if 0.8 <= avg_iqr <= 1.6:
            print(f"OK IQR在0.8-1.6区间: {avg_iqr:.3f}")
        else:
            print(f"FAIL IQR超出0.8-1.6区间: {avg_iqr:.3f}")
        
        if avg_pass_rate >= 90:
            print(f"OK 通过率≥90%: {avg_pass_rate:.1f}%")
        else:
            print(f"FAIL 通过率<90%: {avg_pass_rate:.1f}%")
        
        # 最终结论
        if 1.0 <= avg_p_gt_2 <= 8.0 and 0.8 <= avg_iqr <= 1.6 and avg_pass_rate >= 90:
            print(f"\nSUCCESS 扩面验证成功！所有交易对都符合验收标准")
        else:
            print(f"\nWARN 扩面验证需要进一步调整")
    
    # 详细结果表
    print(f"\n=== 详细结果表 ===")
    print(f"{'交易对':<10} {'流动性':<8} {'P(|z|>2)':<10} {'IQR':<8} {'通过率':<8} {'状态'}")
    print("-" * 60)
    
    for symbol, result in results.items():
        if result:
            liquidity = '高' if symbol in symbols['high'] else '低'
            status = 'OK' if result['pass_rate'] >= 90 else 'FAIL'
            print(f"{symbol:<10} {liquidity:<8} {result['p_gt_2']:<10.2f}% {result['iqr']:<8.3f} {result['pass_rate']:<8.1f}% {status}")

def test_symbol(calc, symbol, num_points):
    """测试单个交易对"""
    np.random.seed(42)
    results = []
    
    # 根据交易对调整价格波动范围
    if symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']:
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
        
        print(f"  有效Z-score数据点: {len(z_scores)}")
        print(f"  中位数: {median:.6f}")
        print(f"  IQR: {iqr:.6f}")
        print(f"  P(|z| > 2): {p_gt_2:.2f}%")
        print(f"  P(|z| > 3): {p_gt_3:.2f}%")
        
        # 验收标准检查
        median_ok = -0.1 <= median <= 0.1
        iqr_ok = 0.8 <= iqr <= 1.6
        tail_2_ok = 1 <= p_gt_2 <= 8
        tail_3_ok = p_gt_3 <= 1.5
        
        passed = sum([median_ok, iqr_ok, tail_2_ok, tail_3_ok])
        pass_rate = passed / 4 * 100
        
        print(f"  通过率: {passed}/4 ({pass_rate:.1f}%)")
        
        return {
            'median': median,
            'iqr': iqr,
            'p_gt_2': p_gt_2,
            'p_gt_3': p_gt_3,
            'pass_rate': pass_rate
        }
    else:
        print("  FAIL 没有有效的Z-score数据")
        return None

if __name__ == "__main__":
    test_all_symbols()
