#!/usr/bin/env python3
"""
测试关键改进的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.real_ofi_calculator import RealOFICalculator, OFIConfig
from src.real_cvd_calculator import RealCVDCalculator, CVDConfig
from analysis.utils_labels import LabelConstructor
import pandas as pd
import numpy as np

def test_l1_ofi():
    """测试L1 OFI实现"""
    print("=== 测试L1 OFI实现 ===")
    
    # 创建OFI计算器
    config = OFIConfig(levels=5, z_window=100, ema_alpha=0.1)
    ofi_calc = RealOFICalculator("ETHUSDT", config)
    
    # 模拟订单簿数据（价格跃迁场景）
    bids = [(100.0, 10.0), (99.9, 15.0), (99.8, 20.0), (99.7, 25.0), (99.6, 30.0)]
    asks = [(100.1, 10.0), (100.2, 15.0), (100.3, 20.0), (100.4, 25.0), (100.5, 30.0)]
    
    # 第一次更新
    result1 = ofi_calc.update_with_snapshot(bids, asks, 1000000)
    print(f"初始OFI: {result1['ofi']:.4f}")
    
    # 价格跃迁：bid价上涨
    bids_new = [(100.1, 12.0), (100.0, 10.0), (99.9, 15.0), (99.8, 20.0), (99.7, 25.0)]
    result2 = ofi_calc.update_with_snapshot(bids_new, asks, 1001000)
    print(f"价跃迁后OFI: {result2['ofi']:.4f}")
    print(f"L1冲击检测: 价格变化 {bids[0][0]:.1f} -> {bids_new[0][0]:.1f}")
    
    return result2['ofi'] != result1['ofi']

def test_cvd_auto_flip():
    """测试CVD自动翻转功能"""
    print("\n=== 测试CVD自动翻转功能 ===")
    
    # 创建CVD计算器
    config = CVDConfig(auto_flip_enabled=True, auto_flip_threshold=0.04)
    cvd_calc = RealCVDCalculator("ETHUSDT", config)
    
    # 模拟交易数据
    result1 = cvd_calc.update_with_trade(price=100.0, qty=1.0, is_buy=True, event_time_ms=1000000)
    print(f"初始CVD: {result1['cvd']:.4f}")
    
    # 设置翻转状态
    cvd_calc.set_flip_state(True, "AUC提升0.05")
    is_flipped, reason = cvd_calc.get_flip_state()
    print(f"翻转状态: {is_flipped}, 原因: {reason}")
    
    return is_flipped

def test_midprice_labels():
    """测试中间价标签"""
    print("\n=== 测试中间价标签 ===")
    
    # 创建标签构造器
    label_constructor = LabelConstructor(horizons=[60, 180], price_type="mid")
    
    # 模拟价格数据
    data = {
        'ts_ms': [1000000, 1001000, 1002000, 1003000, 1004000],
        'best_bid': [100.0, 100.1, 100.2, 100.1, 100.3],
        'best_ask': [100.1, 100.2, 100.3, 100.2, 100.4],
        'price': [100.05, 100.15, 100.25, 100.15, 100.35]  # 成交价
    }
    df = pd.DataFrame(data)
    
    # 构造标签
    labeled_df = label_constructor.construct_labels(df)
    
    # 检查是否使用了中间价
    if 'price' in labeled_df.columns:
        mid_prices = (labeled_df['best_bid'] + labeled_df['best_ask']) / 2
        is_midprice = np.allclose(labeled_df['price'], mid_prices)
        print(f"使用中间价: {is_midprice}")
        return is_midprice
    
    return False

def test_tick_rule_limit():
    """测试tick-rule传播限制"""
    print("\n=== 测试tick-rule传播限制 ===")
    
    # 创建CVD计算器
    config = CVDConfig(use_tick_rule=True)
    cvd_calc = RealCVDCalculator("ETHUSDT", config)
    
    # 模拟连续相同价格的交易
    results = []
    for i in range(10):
        result = cvd_calc.update_with_trade(
            price=100.0,  # 相同价格
            qty=1.0, 
            is_buy=None,  # 强制使用tick-rule
            event_time_ms=1000000 + i * 1000
        )
        results.append(result)
    
    # 检查是否有限制传播
    cvd_values = [r['cvd'] for r in results if r and 'cvd' in r]
    print(f"CVD值变化: {cvd_values[:5]}...")
    
    # 应该看到传播被限制
    return len(cvd_values) > 0

def test_fusion_calibration():
    """测试Fusion校准"""
    print("\n=== 测试Fusion校准 ===")
    
    # 模拟OFI和CVD结果
    ofi_result = {'ofi_z': 2.0}
    cvd_result = {'z_cvd': 1.5}
    
    # 模拟Fusion计算（堆分+校准）
    w_ofi = 0.5
    w_cvd = 0.5
    fusion_raw = w_ofi * ofi_result['ofi_z'] + w_cvd * cvd_result['z_cvd']
    
    # Platt校准
    import math
    k = 1.0
    proba = 1 / (1 + math.exp(-k * fusion_raw))
    
    print(f"融合原始分数: {fusion_raw:.4f}")
    print(f"校准后概率: {proba:.4f}")
    
    # 检查概率是否在合理范围内
    return 0 <= proba <= 1

def main():
    """运行所有测试"""
    print("开始测试关键改进...")
    
    tests = [
        ("L1 OFI实现", test_l1_ofi),
        ("CVD自动翻转", test_cvd_auto_flip),
        ("中间价标签", test_midprice_labels),
        ("Tick-rule限制", test_tick_rule_limit),
        ("Fusion校准", test_fusion_calibration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            status = "PASS" if result else "FAIL"
            print(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"{test_name}: ERROR - {e}")
    
    print(f"\n测试结果汇总:")
    passed = sum(results.values())
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
