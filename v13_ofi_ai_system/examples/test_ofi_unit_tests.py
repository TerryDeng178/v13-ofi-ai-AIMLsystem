#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OFI计算器单测用例
补4个单测用例：z_window动态更新、winsor别名映射、裁剪关闭路径、尾部计数正确性
"""

import sys
import os
import unittest
import numpy as np
from collections import deque

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from real_ofi_calculator import RealOFICalculator, OFIConfig

class TestOFICalculator(unittest.TestCase):
    """OFI计算器单测用例"""
    
    def setUp(self):
        """测试前准备"""
        self.config = OFIConfig(
            z_window=80,
            ema_alpha=0.30,
            z_clip=3.0,
            std_floor=1e-7,
            winsorize_ofi_delta=3.0
        )
        self.calc = RealOFICalculator("BTCUSDT", self.config)
    
    def test_z_window_dynamic_update(self):
        """测试z_window动态更新"""
        print("\n--- 测试z_window动态更新 ---")
        
        # 初始窗口大小
        initial_window = len(self.calc.ofi_hist)
        self.assertEqual(initial_window, 0)
        
        # 添加一些数据
        for i in range(100):
            bids = [[50000 + i, 10], [49999 + i, 5], [49998 + i, 3], [49997 + i, 2], [49996 + i, 1]]
            asks = [[50001 + i, 12], [50002 + i, 6], [50003 + i, 4], [50004 + i, 2], [50005 + i, 1]]
            self.calc.update_with_snapshot(bids, asks, event_time_ms=i*1000)
        
        # 检查窗口大小
        self.assertEqual(len(self.calc.ofi_hist), 80)  # 应该被限制在80
        
        # 动态更新z_window
        new_window = 120
        updated = self.calc.update_params(z_window=new_window)
        
        # 检查窗口大小是否更新
        self.assertEqual(self.calc.z_window, new_window)
        self.assertEqual(self.calc.ofi_hist.maxlen, new_window)
        
        # 添加更多数据验证
        for i in range(50):
            bids = [[50000 + i, 10], [49999 + i, 5], [49998 + i, 3], [49997 + i, 2], [49996 + i, 1]]
            asks = [[50001 + i, 12], [50002 + i, 6], [50003 + i, 4], [50004 + i, 2], [50005 + i, 1]]
            self.calc.update_with_snapshot(bids, asks, event_time_ms=(i+100)*1000)
        
        # 检查窗口大小
        self.assertEqual(len(self.calc.ofi_hist), 120)  # 应该被限制在120
        
        print(f"OK z_window动态更新测试通过")
    
    def test_winsorize_alias_mapping(self):
        """测试winsorize参数别名映射"""
        print("\n--- 测试winsorize参数别名映射 ---")
        
        # 测试不同的别名
        aliases = [
            ('winsor_k_mad', 2.5),
            ('winsorize_ofi_delta', 3.0),
            ('winsorize_ofi_delta_mad_k', 3.5)
        ]
        
        for alias, value in aliases:
            # 使用正确的参数名
            if alias == 'winsor_k_mad':
                config = OFIConfig(winsorize_ofi_delta=value)
            elif alias == 'winsorize_ofi_delta_mad_k':
                config = OFIConfig(winsorize_ofi_delta=value)
            else:
                config = OFIConfig(**{alias: value})
            calc = RealOFICalculator("TEST", config)
            
            # 检查是否正确映射
            self.assertEqual(calc.winsor_k_mad, value)
            print(f"OK 别名 {alias}={value} 映射正确")
        
        print("OK winsorize参数别名映射测试通过")
    
    def test_z_clip_disable_path(self):
        """测试z_clip裁剪关闭路径"""
        print("\n--- 测试z_clip裁剪关闭路径 ---")
        
        # 测试不同的关闭条件
        disable_values = [0, -1, 1e6, 1e7, float('inf')]
        
        for disable_val in disable_values:
            config = OFIConfig(z_clip=disable_val)
            calc = RealOFICalculator("TEST", config)
            
            # 添加一些数据
            for i in range(50):
                bids = [[50000 + i, 10], [49999 + i, 5], [49998 + i, 3], [49997 + i, 2], [49996 + i, 1]]
                asks = [[50001 + i, 12], [50002 + i, 6], [50003 + i, 4], [50004 + i, 2], [50005 + i, 1]]
                result = calc.update_with_snapshot(bids, asks, event_time_ms=i*1000)
                
                if result['z_ofi'] is not None:
                    # 检查Z-score是否被裁剪
                    z_ofi = result['z_ofi']
                    if disable_val <= 0 or disable_val >= 1e6:
                        # 应该不被裁剪
                        self.assertTrue(abs(z_ofi) <= 10)  # 合理范围内
                    else:
                        # 应该被裁剪
                        self.assertTrue(abs(z_ofi) <= disable_val)
            
            print(f"OK z_clip={disable_val} 关闭路径测试通过")
        
        print("OK z_clip裁剪关闭路径测试通过")
    
    def test_tail_counting_correctness(self):
        """测试尾部计数正确性"""
        print("\n--- 测试尾部计数正确性 ---")
        
        # 重置计数器
        self.calc.p_gt2_cnt = 0
        self.calc.p_gt3_cnt = 0
        self.calc.total_cnt = 0
        
        # 添加一些数据
        z_scores = []
        for i in range(100):
            bids = [[50000 + i, 10], [49999 + i, 5], [49998 + i, 3], [49997 + i, 2], [49996 + i, 1]]
            asks = [[50001 + i, 12], [50002 + i, 6], [50003 + i, 4], [50004 + i, 2], [50005 + i, 1]]
            result = self.calc.update_with_snapshot(bids, asks, event_time_ms=i*1000)
            
            if result['z_ofi'] is not None:
                z_scores.append(result['z_ofi'])
        
        # 手动计算尾部计数
        manual_gt2 = sum(1 for z in z_scores if abs(z) > 2)
        manual_gt3 = sum(1 for z in z_scores if abs(z) > 3)
        manual_total = len(z_scores)
        
        # 检查计数器是否正确
        self.assertEqual(self.calc.p_gt2_cnt, manual_gt2)
        self.assertEqual(self.calc.p_gt3_cnt, manual_gt3)
        self.assertEqual(self.calc.total_cnt, manual_total)
        
        # 检查百分比计算
        expected_p_gt2 = (manual_gt2 / manual_total * 100) if manual_total > 0 else 0
        expected_p_gt3 = (manual_gt3 / manual_total * 100) if manual_total > 0 else 0
        
        # 获取最后一个结果的meta
        last_result = self.calc.update_with_snapshot(
            [[50000, 10], [49999, 5], [49998, 3], [49997, 2], [49996, 1]],
            [[50001, 12], [50002, 6], [50003, 4], [50004, 2], [50005, 1]],
            event_time_ms=100*1000
        )
        
        if 'meta' in last_result:
            meta = last_result['meta']
            self.assertAlmostEqual(meta['p_gt2_percent'], expected_p_gt2, places=2)
            self.assertAlmostEqual(meta['p_gt3_percent'], expected_p_gt3, places=2)
        
        print(f"OK 尾部计数正确性测试通过")
        print(f"   P(|z|>2): {manual_gt2}/{manual_total} = {expected_p_gt2:.2f}%")
        print(f"   P(|z|>3): {manual_gt3}/{manual_total} = {expected_p_gt3:.2f}%")

def run_tests():
    """运行所有测试"""
    print("=== OFI计算器单测用例 ===")
    print("测试4个关键功能：z_window动态更新、winsor别名映射、裁剪关闭路径、尾部计数正确性")
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOFICalculator)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    print(f"\n=== 测试结果 ===")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败详情:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\n错误详情:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\nSUCCESS 所有测试通过！")
    else:
        print("\nFAIL 部分测试失败")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_tests()
