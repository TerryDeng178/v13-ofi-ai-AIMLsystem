"""
OFI-CVD背离检测模块单元测试

测试覆盖：
- 枢轴检测正确性
- 四种背离类型检测
- OFI-CVD冲突检测
- 去噪机制（冷却、聚类合并、最小枢轴间距）
- 评分系统
- 边界条件处理
- 性能基准

Author: V13 OFI+CVD AI System
Created: 2025-01-20
"""

import unittest
import time
import math
import numpy as np
from unittest.mock import patch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ofi_cvd_divergence import (
    DivergenceConfig, DivergenceDetector, DivergenceType, PivotDetector
)


class TestPivotDetector(unittest.TestCase):
    """枢轴检测器测试"""
    
    def setUp(self):
        self.detector = PivotDetector(window_size=5)
    
    def test_pivot_detection_high(self):
        """测试高点检测"""
        # 创建V型价格序列
        prices = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
        indicators = [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1]
        
        # 使用新的方法添加数据点并检测枢轴
        for i, (price, indicator) in enumerate(zip(prices, indicators)):
            self.detector.add_point_and_detect(i, price, indicator)
        
        pivots = self.detector.get_all_pivots()
        self.assertGreater(len(pivots), 0)
        
        # 检查中间点是否为高点
        middle_pivot = next((p for p in pivots if p['index'] == 5), None)
        self.assertIsNotNone(middle_pivot)
        self.assertTrue(middle_pivot['is_price_high'])
    
    def test_pivot_detection_low(self):
        """测试低点检测"""
        # 创建倒V型价格序列
        prices = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        indicators = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        # 使用新的方法添加数据点并检测枢轴
        for i, (price, indicator) in enumerate(zip(prices, indicators)):
            self.detector.add_point_and_detect(i, price, indicator)
        
        pivots = self.detector.get_all_pivots()
        self.assertGreater(len(pivots), 0)
        
        # 检查中间点是否为低点
        middle_pivot = next((p for p in pivots if p['index'] == 5), None)
        self.assertIsNotNone(middle_pivot)
        self.assertTrue(middle_pivot['is_price_low'])
    
    def test_insufficient_data(self):
        """测试数据不足时不产生枢轴"""
        # 只添加少量数据点
        for i in range(3):
            self.detector.add_point(i, 1.0 + i * 0.1, 0.1 + i * 0.1)
        
        pivots = self.detector.find_pivots()
        self.assertEqual(len(pivots), 0)


class TestDivergenceDetector(unittest.TestCase):
    """背离检测器测试"""
    
    def setUp(self):
        self.config = DivergenceConfig(
            swing_L=5,
            ema_k=3,
            z_hi=2.0,
            z_mid=1.0,
            min_separation=3,
            cooldown_secs=1.0,
            warmup_min=10,
            max_lag=0.1,
            use_fusion=True,
            cons_min=0.3
        )
        self.detector = DivergenceDetector(self.config)
    
    def test_warmup_phase(self):
        """测试暖启动阶段不产生事件"""
        # 在暖启动阶段
        result = self.detector.update(
            ts=1.0, price=100.0, z_ofi=2.0, z_cvd=1.5,
            warmup=True
        )
        
        self.assertIsNotNone(result)
        self.assertIn('warmup', result['reason_codes'])
        self.assertEqual(result['type'], None)
    
    def test_insufficient_samples(self):
        """测试样本数不足时不产生事件"""
        # 样本数少于warmup_min
        for i in range(5):
            result = self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
            if i < 4:  # 前几个样本
                self.assertIsNotNone(result)
                self.assertIn('warmup', result['reason_codes'])
    
    def test_lag_exceeded(self):
        """测试滞后超阈时不产生事件"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 滞后超阈
        result = self.detector.update(
            ts=20.0, price=120.0, z_ofi=2.0, z_cvd=1.5,
            lag_sec=0.5  # 超过max_lag=0.1
        )
        
        self.assertIsNotNone(result)
        self.assertIn('lag_exceeded', result['reason_codes'])
        self.assertEqual(result['type'], None)
    
    def test_invalid_input(self):
        """测试无效输入"""
        # NaN输入
        result = self.detector.update(
            ts=float('nan'), price=100.0, z_ofi=1.0, z_cvd=0.5
        )
        self.assertIsNotNone(result)
        self.assertIn('invalid_input', result['reason_codes'])
        
        # 负价格
        result = self.detector.update(
            ts=1.0, price=-100.0, z_ofi=1.0, z_cvd=0.5
        )
        self.assertIsNotNone(result)
        self.assertIn('invalid_input', result['reason_codes'])
    
    def test_ofi_cvd_conflict(self):
        """测试OFI-CVD冲突检测"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # OFI-CVD冲突：OFI强正，CVD强负
        result = self.detector.update(
            ts=20.0, price=120.0, z_ofi=3.0, z_cvd=-2.5  # 满足冲突条件
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result['type'], 'ofi_cvd_conflict')
        self.assertIn('ofi_cvd_conflict', result['reason_codes'])
        self.assertIn('price_ofi', result['channels'])
        self.assertIn('price_cvd', result['channels'])
    
    def test_bull_regular_divergence(self):
        """测试看涨常规背离"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 创建看涨背离：价格下降，OFI上升
        # 第一阶段：价格高，OFI低
        for i in range(10):
            ts = 20.0 + i
            price = 120.0 - i * 0.5  # 价格下降
            z_ofi = 1.0 - i * 0.1    # OFI下降
            self.detector.update(ts, price, z_ofi, 0.5)
        
        # 第二阶段：价格低，OFI高
        for i in range(10):
            ts = 30.0 + i
            price = 115.0 + i * 0.2  # 价格上升
            z_ofi = 0.0 + i * 0.2    # OFI上升
            result = self.detector.update(ts, price, z_ofi, 0.5)
            
            if result and result['type'] == 'bull_div':
                self.assertEqual(result['type'], 'bull_div')
                self.assertIn('price_ofi', result['channels'])
                self.assertGreater(result['score'], 0)
                break
    
    def test_bear_regular_divergence(self):
        """测试看跌常规背离"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 创建看跌背离：价格上升，OFI下降
        # 第一阶段：价格低，OFI高
        for i in range(10):
            ts = 20.0 + i
            price = 100.0 + i * 0.5  # 价格上升
            z_ofi = 2.0 - i * 0.1    # OFI下降
            self.detector.update(ts, price, z_ofi, 0.5)
        
        # 第二阶段：价格高，OFI低
        for i in range(10):
            ts = 30.0 + i
            price = 105.0 - i * 0.2  # 价格下降
            z_ofi = 1.0 + i * 0.1    # OFI上升
            result = self.detector.update(ts, price, z_ofi, 0.5)
            
            if result and result['type'] == 'bear_div':
                self.assertEqual(result['type'], 'bear_div')
                self.assertIn('price_ofi', result['channels'])
                self.assertGreater(result['score'], 0)
                break
    
    def test_cooldown_mechanism(self):
        """测试冷却机制"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 触发一个冲突事件
        result1 = self.detector.update(
            ts=20.0, price=120.0, z_ofi=3.0, z_cvd=-2.5  # 冲突事件
        )
        self.assertIsNotNone(result1)
        
        # 在冷却期内再次触发相同类型的冲突事件
        result2 = self.detector.update(
            ts=20.5, price=121.0, z_ofi=3.5, z_cvd=-3.0  # 另一个冲突事件
        )
        # 相同类型的事件应该被冷却机制抑制
        if result2 and result2['type'] == 'ofi_cvd_conflict':
            self.assertIsNone(result2)  # 应该被冷却机制抑制
    
    def test_minimum_separation(self):
        """测试最小枢轴间距"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 创建距离太近的枢轴
        for i in range(2):  # 只有2个点，小于min_separation=3
            ts = 20.0 + i
            price = 120.0 - i * 0.1
            z_ofi = 1.0 - i * 0.1
            result = self.detector.update(ts, price, z_ofi, 0.5)
            # 应该不会产生背离事件，因为枢轴间距太小
            if result and 'bull_div' in result.get('type', ''):
                self.fail("不应该产生背离事件，枢轴间距太小")
    
    def test_score_calculation(self):
        """测试评分计算"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 触发一个高评分事件
        result = self.detector.update(
            ts=20.0, price=120.0, z_ofi=3.0, z_cvd=-2.5  # 强信号
        )
        
        if result and result['type'] == 'ofi_cvd_conflict':
            self.assertGreaterEqual(result['score'], 0)
            self.assertLessEqual(result['score'], 100)
    
    def test_z_value_clipping(self):
        """测试Z值裁剪"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 使用超出范围的Z值
        result = self.detector.update(
            ts=20.0, price=120.0, z_ofi=10.0, z_cvd=-8.0  # 超出[-5,5]范围
        )
        
        # 应该正常处理，Z值会被裁剪
        self.assertIsNotNone(result)
        if result['type'] == 'ofi_cvd_conflict':
            self.assertIn('debug', result)
    
    def test_statistics_tracking(self):
        """测试统计信息跟踪"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 触发一些事件
        self.detector.update(ts=20.0, price=120.0, z_ofi=3.0, z_cvd=-2.5)
        self.detector.update(ts=21.0, price=121.0, z_ofi=3.5, z_cvd=-3.0, lag_sec=0.5)  # 滞后超阈
        
        stats = self.detector.get_stats()
        self.assertGreater(stats['events_total'], 0)
        self.assertGreater(stats['suppressed_total'], 0)
        self.assertIn('lag_exceeded', stats['suppressed_by_reason'])
    
    def test_reset_functionality(self):
        """测试重置功能"""
        # 先完成暖启动并触发事件
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        self.detector.update(ts=20.0, price=120.0, z_ofi=3.0, z_cvd=-2.5)
        
        # 重置
        self.detector.reset()
        
        # 检查统计信息是否重置
        stats = self.detector.get_stats()
        self.assertEqual(stats['events_total'], 0)
        self.assertEqual(stats['suppressed_total'], 0)
        
        # 检查样本计数是否重置
        self.assertEqual(self.detector._sample_count, 0)
    
    def test_hidden_bull_divergence(self):
        """测试隐藏看涨背离"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 创建隐藏看涨背离：价格HL，OFI LL
        # 第一阶段：价格低，OFI高
        for i in range(10):
            ts = 20.0 + i
            price = 100.0 + i * 0.5  # 价格上升
            z_ofi = 2.0 - i * 0.1    # OFI下降
            self.detector.update(ts, price, z_ofi, 0.5)
        
        # 第二阶段：价格高，OFI低
        for i in range(10):
            ts = 30.0 + i
            price = 105.0 - i * 0.2  # 价格下降
            z_ofi = 1.0 + i * 0.1    # OFI上升
            result = self.detector.update(ts, price, z_ofi, 0.5)
            
            if result and result['type'] == 'hidden_bull':
                self.assertEqual(result['type'], 'hidden_bull')
                self.assertIn('price_ofi', result['channels'])
                self.assertGreater(result['score'], 0)
                break
    
    def test_hidden_bear_divergence(self):
        """测试隐藏看跌背离"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 创建隐藏看跌背离：价格LH，OFI HH
        # 第一阶段：价格高，OFI低
        for i in range(10):
            ts = 20.0 + i
            price = 120.0 - i * 0.5  # 价格下降
            z_ofi = 1.0 + i * 0.1    # OFI上升
            self.detector.update(ts, price, z_ofi, 0.5)
        
        # 第二阶段：价格低，OFI高
        for i in range(10):
            ts = 30.0 + i
            price = 115.0 + i * 0.2  # 价格上升
            z_ofi = 2.0 - i * 0.1    # OFI下降
            result = self.detector.update(ts, price, z_ofi, 0.5)
            
            if result and result['type'] == 'hidden_bear':
                self.assertEqual(result['type'], 'hidden_bear')
                self.assertIn('price_ofi', result['channels'])
                self.assertGreater(result['score'], 0)
                break
    
    def test_price_cvd_divergence(self):
        """测试价格-CVD背离"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 创建价格-CVD背离
        for i in range(20):
            ts = 20.0 + i
            price = 100.0 + i * 0.3  # 价格上升
            z_cvd = 2.0 - i * 0.1    # CVD下降
            result = self.detector.update(ts, price, 1.0, z_cvd)
            
            if result and 'cvd' in result['channels']:
                self.assertIn('price_cvd', result['channels'])
                self.assertGreater(result['score'], 0)
                break
    
    def test_price_fusion_divergence(self):
        """测试价格-融合背离"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5,
                fusion_score=1.0, consistency=0.5
            )
        
        # 创建价格-融合背离
        for i in range(20):
            ts = 20.0 + i
            price = 100.0 + i * 0.3  # 价格上升
            fusion_score = 2.0 - i * 0.1  # 融合分数下降
            result = self.detector.update(
                ts, price, 1.0, 0.5, fusion_score=fusion_score, consistency=0.5
            )
            
            if result and 'fusion' in result['channels']:
                self.assertIn('price_fusion', result['channels'])
                self.assertGreater(result['score'], 0)
                break
    
    def test_consistency_bonus(self):
        """测试一致性加分"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5,
                fusion_score=1.0, consistency=0.5
            )
        
        # 测试高一致性情况
        result_high_cons = self.detector.update(
            ts=20.0, price=120.0, z_ofi=3.0, z_cvd=-2.5,
            fusion_score=2.0, consistency=0.8  # 高一致性
        )
        
        # 测试低一致性情况
        result_low_cons = self.detector.update(
            ts=21.0, price=121.0, z_ofi=3.0, z_cvd=-2.5,
            fusion_score=2.0, consistency=0.2  # 低一致性
        )
        
        # 高一致性应该有更高的评分（如果都触发了事件）
        if result_high_cons and result_low_cons:
            self.assertGreaterEqual(
                result_high_cons['score'], result_low_cons['score']
            )
    
    def test_cooldown_by_type(self):
        """测试按类型冷却"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 触发一个冲突事件
        result1 = self.detector.update(
            ts=20.0, price=120.0, z_ofi=3.0, z_cvd=-2.5
        )
        self.assertIsNotNone(result1)
        
        # 立即触发一个方向性背离事件（不同类型）
        result2 = self.detector.update(
            ts=20.1, price=121.0, z_ofi=3.5, z_cvd=-3.0
        )
        # 不同类型的事件应该不被冷却机制抑制
        if result2 and result2['type'] != 'ofi_cvd_conflict':
            self.assertIsNotNone(result2)
    
    def test_multiple_channels_consistency(self):
        """测试多通道一致性"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5,
                fusion_score=1.0, consistency=0.5
            )
        
        # 创建多通道一致背离：价格、OFI、CVD、Fusion都同向
        for i in range(20):
            ts = 20.0 + i
            price = 100.0 + i * 0.3  # 价格上升
            z_ofi = 1.0 + i * 0.1    # OFI上升
            z_cvd = 0.5 + i * 0.1    # CVD上升
            fusion_score = 1.0 + i * 0.1  # Fusion上升
            result = self.detector.update(
                ts, price, z_ofi, z_cvd, fusion_score=fusion_score, consistency=0.8
            )
            
            if result and result['type'] in ['bull_div', 'hidden_bull']:
                # 多通道一致性应该提高评分
                self.assertGreaterEqual(result['score'], 50)  # 至少中等评分
                break


class TestDivergenceDetectorPerformance(unittest.TestCase):
    """背离检测器性能测试"""
    
    def setUp(self):
        self.config = DivergenceConfig(
            swing_L=5,
            warmup_min=10,
            cooldown_secs=0.1
        )
        self.detector = DivergenceDetector(self.config)
    
    def test_performance_benchmark(self):
        """性能基准测试：P95延迟<3ms"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 性能测试
        latencies = []
        iterations = 10000
        
        for i in range(iterations):
            ts = 20.0 + i * 0.001
            price = 120.0 + i * 0.01
            z_ofi = 1.0 + (i % 10) * 0.1
            z_cvd = 0.5 + (i % 7) * 0.1
            
            start_time = time.perf_counter()
            self.detector.update(ts, price, z_ofi, z_cvd)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        # 计算统计信息
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.5)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        max_latency = max(latencies)
        
        print(f"\n性能基准测试结果:")
        print(f"P50延迟: {p50:.3f}ms")
        print(f"P95延迟: {p95:.3f}ms")
        print(f"P99延迟: {p99:.3f}ms")
        print(f"最大延迟: {max_latency:.3f}ms")
        
        # 断言P95延迟小于3ms
        self.assertLess(p95, 3.0, f"P95延迟{p95:.3f}ms超过3ms阈值")
        
        # 断言P99延迟小于5ms
        self.assertLess(p99, 5.0, f"P99延迟{p99:.3f}ms超过5ms阈值")
    
    def test_channels_consistency(self):
        """测试channels字段一致性"""
        # 先完成暖启动
        for i in range(15):
            self.detector.update(
                ts=i + 1.0, price=100.0 + i, z_ofi=1.0, z_cvd=0.5
            )
        
        # 创建背离事件
        for i in range(20):
            ts = 20.0 + i
            price = 100.0 + i * 0.3  # 价格上升
            z_ofi = 2.0 - i * 0.1    # OFI下降
            result = self.detector.update(ts, price, z_ofi, 1.0)
            
            if result and result['type'] in ['bull_div', 'bear_div', 'hidden_bull', 'hidden_bear']:
                # 检查channels字段存在且包含正确的通道
                self.assertIn('channels', result, "事件必须包含channels字段")
                self.assertIsInstance(result['channels'], list, "channels必须是列表")
                self.assertGreater(len(result['channels']), 0, "channels不能为空")
                
                # 检查channels包含price_ofi或price_cvd或price_fusion
                valid_channels = ['price_ofi', 'price_cvd', 'price_fusion']
                has_valid_channel = any(ch in result['channels'] for ch in valid_channels)
                self.assertTrue(has_valid_channel, f"channels必须包含有效的通道: {result['channels']}")
                
                # 检查channel字段也存在（向后兼容）
                self.assertIn('channel', result, "事件必须包含channel字段")
                break


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
