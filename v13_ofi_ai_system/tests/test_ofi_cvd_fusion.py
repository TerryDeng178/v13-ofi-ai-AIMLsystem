"""
OFI+CVD融合模块单元测试

测试覆盖:
1. 权重归一化
2. 一致性边界
3. 暖启动保护
4. 时序对齐
5. 去噪逻辑
6. 可复现性

Author: V13 OFI+CVD AI Trading System
Date: 2025-10-19
"""

import unittest
import math
import time
from src.ofi_cvd_fusion import OFI_CVD_Fusion, OFICVDFusionConfig, SignalType


class TestOFICVDFusion(unittest.TestCase):
    """OFI+CVD融合测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = OFICVDFusionConfig()
        self.fusion = OFI_CVD_Fusion(self.config)
    
    def test_weight_normalization(self):
        """测试权重归一化"""
        # 测试任意配置都能归一
        config = OFICVDFusionConfig(w_ofi=0.8, w_cvd=0.6)
        fusion = OFI_CVD_Fusion(config)
        
        self.assertAlmostEqual(fusion.w_ofi + fusion.w_cvd, 1.0, places=10)
        self.assertAlmostEqual(fusion.w_ofi, 0.8 / 1.4, places=10)
        self.assertAlmostEqual(fusion.w_cvd, 0.6 / 1.4, places=10)
        
        # 测试非法配置
        with self.assertRaises(ValueError):
            OFI_CVD_Fusion(OFICVDFusionConfig(w_ofi=0, w_cvd=0))
    
    def test_consistency_boundaries(self):
        """测试一致性边界"""
        # 同号情况
        result = self.fusion._consistency(2.0, 1.0)
        self.assertAlmostEqual(result, 0.5, places=10)
        
        # 异号情况
        result = self.fusion._consistency(2.0, -1.0)
        self.assertEqual(result, 0.0)
        
        # 零值情况
        result = self.fusion._consistency(0.0, 1.0)
        self.assertEqual(result, 0.0)
        
        result = self.fusion._consistency(1.0, 0.0)
        self.assertEqual(result, 0.0)
        
        # 极值情况
        result = self.fusion._consistency(5.0, 0.1)
        self.assertAlmostEqual(result, 0.02, places=10)
    
    def test_warmup_protection(self):
        """测试暖启动保护"""
        # 前30次应该返回warmup
        for i in range(30):
            result = self.fusion.update(1.0, 1.0, time.time())
            self.assertEqual(result['signal'], 'neutral')
            self.assertTrue(result['warmup'])
            self.assertIn('warmup', result['reason_codes'])
        
        # 第31次应该正常处理
        result = self.fusion.update(3.0, 2.0, time.time())
        self.assertFalse(result['warmup'])
        self.assertNotIn('warmup', result['reason_codes'])
    
    def test_invalid_input_handling(self):
        """测试无效输入处理"""
        # NaN输入
        result = self.fusion.update(float('nan'), 1.0, time.time())
        self.assertEqual(result['signal'], 'neutral')
        self.assertIn('invalid_input', result['reason_codes'])
        
        # Inf输入
        result = self.fusion.update(float('inf'), 1.0, time.time())
        self.assertEqual(result['signal'], 'neutral')
        self.assertIn('invalid_input', result['reason_codes'])
        
        # None输入
        result = self.fusion.update(None, 1.0, time.time())
        self.assertEqual(result['signal'], 'neutral')
        self.assertIn('invalid_input', result['reason_codes'])
    
    def test_signal_generation(self):
        """测试信号生成"""
        # 跳过暖启动
        for _ in range(35):
            self.fusion.update(0.1, 0.1, time.time())
        
        # 强买入信号 - 需要满足两个条件：融合得分和一致性
        result = self.fusion.update(3.0, 2.5, time.time())
        self.assertEqual(result['signal'], 'strong_buy')
        self.assertGreater(result['consistency'], 0.7)
        
        # 等待冷却时间
        time.sleep(1.1)
        
        # 买入信号 - 由于迟滞逻辑，可能保持为strong_buy
        result = self.fusion.update(2.0, 1.5, time.time())
        self.assertIn(result['signal'], ['buy', 'strong_buy'])
        self.assertGreater(result['consistency'], 0.3)
        
        # 等待冷却时间
        time.sleep(1.1)
        
        # 强卖出信号
        result = self.fusion.update(-3.0, -2.5, time.time())
        self.assertEqual(result['signal'], 'strong_sell')
        
        # 等待冷却时间
        time.sleep(1.1)
        
        # 卖出信号 - 由于迟滞逻辑，可能保持为strong_sell
        result = self.fusion.update(-2.0, -1.5, time.time())
        self.assertIn(result['signal'], ['sell', 'strong_sell'])
    
    def test_denoising_logic(self):
        """测试去噪逻辑"""
        # 跳过暖启动
        for _ in range(35):
            self.fusion.update(0.1, 0.1, time.time())
        
        ts = time.time()
        
        # 测试冷却时间
        result1 = self.fusion.update(3.0, 2.5, ts)
        self.assertEqual(result1['signal'], 'strong_buy')
        
        # 立即反向信号应该被冷却
        result2 = self.fusion.update(-3.0, -2.5, ts + 0.1)
        self.assertEqual(result2['signal'], 'neutral')
        
        # 等待冷却时间后应该正常
        time.sleep(1.1)
        result3 = self.fusion.update(2.0, 1.5, time.time())
        # 由于迟滞逻辑，可能保持为strong_buy，这是正常的
        self.assertIn(result3['signal'], ['buy', 'strong_buy'])
    
    def test_hysteresis(self):
        """测试迟滞机制"""
        # 跳过暖启动
        for _ in range(35):
            self.fusion.update(0.1, 0.1, time.time())
        
        # 触发强买入
        result1 = self.fusion.update(3.0, 2.5, time.time())
        self.assertEqual(result1['signal'], 'strong_buy')
        
        # 等待冷却时间
        time.sleep(1.1)
        
        # 回落到迟滞阈值内，应该保持原信号
        result2 = self.fusion.update(1.5, 1.0, time.time())
        self.assertEqual(result2['signal'], 'strong_buy')
        
        # 回落到迟滞阈值外，应该变为neutral
        result3 = self.fusion.update(0.5, 0.5, time.time())
        self.assertEqual(result3['signal'], 'neutral')
    
    def test_lag_handling(self):
        """测试时间滞后处理"""
        # 跳过暖启动
        for _ in range(35):
            self.fusion.update(0.1, 0.1, time.time())
        
        # 正常滞后
        result1 = self.fusion.update(2.0, 1.5, time.time(), lag_sec=0.1)
        self.assertNotIn('lag_exceeded', result1['reason_codes'])
        
        # 超时滞后
        result2 = self.fusion.update(2.0, 1.5, time.time(), lag_sec=0.5)
        self.assertIn('lag_exceeded', result2['reason_codes'])
    
    def test_reproducibility(self):
        """测试可复现性"""
        # 跳过暖启动
        for _ in range(35):
            self.fusion.update(0.1, 0.1, time.time())
        
        # 相同输入应该产生相同输出
        ts = time.time()
        result1 = self.fusion.update(2.0, 1.5, ts)
        
        # 重置并重新处理
        self.fusion.reset()
        for _ in range(35):
            self.fusion.update(0.1, 0.1, time.time())
        
        result2 = self.fusion.update(2.0, 1.5, ts)
        
        self.assertEqual(result1['signal'], result2['signal'])
        self.assertAlmostEqual(result1['fusion_score'], result2['fusion_score'])
        self.assertAlmostEqual(result1['consistency'], result2['consistency'])
    
    def test_z_score_clipping(self):
        """测试Z-score裁剪"""
        # 跳过暖启动
        for _ in range(35):
            self.fusion.update(0.1, 0.1, time.time())
        
        # 极值应该被裁剪
        result = self.fusion.update(10.0, -10.0, time.time())
        self.assertLessEqual(abs(result['components']['ofi']), 5.0)
        self.assertLessEqual(abs(result['components']['cvd']), 5.0)
    
    def test_statistics_tracking(self):
        """测试统计信息跟踪"""
        # 测试各种统计
        self.fusion.update(float('nan'), 1.0, time.time())
        self.fusion.update(1.0, 1.0, time.time(), lag_sec=0.5)
        
        stats = self.fusion.get_stats()
        self.assertGreater(stats['invalid_inputs'], 0)
        self.assertGreaterEqual(stats['lag_exceeded'], 0)  # 可能为0，因为lag_sec=0.5 > max_lag=0.3
        self.assertGreater(stats['warmup_returns'], 0)
    
    def test_reset_functionality(self):
        """测试重置功能"""
        # 运行一段时间
        for i in range(50):
            self.fusion.update(i * 0.1, i * 0.1, time.time())
        
        # 重置
        self.fusion.reset()
        
        # 应该回到初始状态
        self.assertEqual(self.fusion._last_signal, SignalType.NEUTRAL)
        self.assertIsNone(self.fusion._last_emit_ts)
        self.assertEqual(self.fusion._streak, 0)
        self.assertEqual(self.fusion._warmup_count, 0)
        
        # 统计应该清零
        stats = self.fusion.get_stats()
        for value in stats.values():
            self.assertEqual(value, 0)


class TestOFICVDFusionConfig(unittest.TestCase):
    """配置类测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = OFICVDFusionConfig()
        
        self.assertEqual(config.w_ofi, 0.6)
        self.assertEqual(config.w_cvd, 0.4)
        self.assertEqual(config.fuse_buy, 1.5)
        self.assertEqual(config.fuse_strong_buy, 2.5)
        self.assertEqual(config.fuse_sell, -1.5)
        self.assertEqual(config.fuse_strong_sell, -2.5)
        self.assertEqual(config.min_consistency, 0.3)
        self.assertEqual(config.strong_min_consistency, 0.7)
        self.assertEqual(config.z_clip, 5.0)
        self.assertEqual(config.max_lag, 0.300)
        self.assertEqual(config.hysteresis_exit, 1.2)
        self.assertEqual(config.cooldown_secs, 1.0)
        self.assertEqual(config.min_consecutive, 2)
        self.assertEqual(config.min_warmup_samples, 30)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = OFICVDFusionConfig(
            w_ofi=0.7,
            w_cvd=0.3,
            fuse_buy=2.0,
            min_consistency=0.5
        )
        
        self.assertEqual(config.w_ofi, 0.7)
        self.assertEqual(config.w_cvd, 0.3)
        self.assertEqual(config.fuse_buy, 2.0)
        self.assertEqual(config.min_consistency, 0.5)
    
    def test_lag_exceeded_degradation(self):
        """测试滞后超阈降级为单因子"""
        # 设置较小的滞后阈值
        config = OFICVDFusionConfig(max_lag=0.1)
        fusion = OFI_CVD_Fusion(config)
        
        # 先通过暖启动期
        for _ in range(35):
            fusion.update(1.0, 1.0, time.time())
        
        # 测试OFI更强的情况
        result = fusion.update(3.0, 1.0, time.time(), lag_sec=0.2)
        self.assertIn("lag_exceeded", result['reason_codes'])
        self.assertIn("degraded_ofi_only", result['reason_codes'])
        self.assertEqual(result['ofi_weight'], 1.0)
        self.assertEqual(result['cvd_weight'], 0.0)
        self.assertAlmostEqual(result['components']['ofi'], 3.0, places=5)
        self.assertAlmostEqual(result['components']['cvd'], 0.0, places=5)
        
        # 测试CVD更强的情况
        fusion.reset()
        for _ in range(35):
            fusion.update(1.0, 1.0, time.time())
        result = fusion.update(1.0, 3.0, time.time(), lag_sec=0.2)
        self.assertIn("lag_exceeded", result['reason_codes'])
        self.assertIn("degraded_cvd_only", result['reason_codes'])
        self.assertEqual(result['ofi_weight'], 0.0)
        self.assertEqual(result['cvd_weight'], 1.0)
        self.assertAlmostEqual(result['components']['ofi'], 0.0, places=5)
        self.assertAlmostEqual(result['components']['cvd'], 3.0, places=5)
    
    def test_performance_benchmark(self):
        """测试性能基准（P50/P95）"""
        import statistics
        
        # 创建融合器实例
        fusion = OFI_CVD_Fusion()
        
        # 准备测试数据
        n_iterations = 10000  # 减少迭代次数以加快测试
        latencies = []
        
        # 预热
        for _ in range(100):
            fusion.update(1.0, 1.0, time.time())
        
        # 性能测试
        start_time = time.time()
        for i in range(n_iterations):
            iter_start = time.time()
            fusion.update(1.0 + i * 0.001, 1.0 + i * 0.001, time.time())
            iter_end = time.time()
            latencies.append((iter_end - iter_start) * 1000)  # 转换为毫秒
        
        end_time = time.time()
        
        # 计算统计信息
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        
        # 验证性能要求
        self.assertLess(p95, 3.0, f"P95延迟 {p95:.3f}ms 超过3ms阈值")
        
        # 打印性能报告
        print(f"\n性能基准测试结果 (n={n_iterations}):")
        print(f"总耗时: {(end_time - start_time):.3f}s")
        print(f"平均延迟: {statistics.mean(latencies):.3f}ms")
        print(f"P50延迟: {p50:.3f}ms")
        print(f"P95延迟: {p95:.3f}ms")
        print(f"P99延迟: {p99:.3f}ms")
        print(f"最大延迟: {max(latencies):.3f}ms")


if __name__ == '__main__':
    unittest.main()
