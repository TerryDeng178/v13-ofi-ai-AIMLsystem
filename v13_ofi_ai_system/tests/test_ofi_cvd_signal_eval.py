#!/usr/bin/env python3
"""
OFI+CVD信号分析工具单元测试
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from analysis.ofi_cvd_signal_eval import OFICVDSignalEvaluator
from analysis.utils_labels import LabelConstructor, SliceAnalyzer, DataValidator
from analysis.plots import PlotGenerator


class TestLabelConstructor(unittest.TestCase):
    """测试标签构造器"""
    
    def setUp(self):
        self.horizons = [60, 180, 300]
        self.constructor = LabelConstructor(self.horizons)
        
        # 创建测试数据
        timestamps = pd.date_range('2025-10-21', periods=1000, freq='1s')
        prices = 100 + np.cumsum(np.random.normal(0, 0.01, 1000))
        
        self.test_data = pd.DataFrame({
            'ts_ms': [int(ts.timestamp() * 1000) for ts in timestamps],
            'price': prices
        })
    
    def test_construct_labels(self):
        """测试标签构造"""
        labeled_data = self.constructor.construct_labels(self.test_data)
        
        # 检查标签列是否存在
        for horizon in self.horizons:
            self.assertIn(f'label_{horizon}s', labeled_data.columns)
            self.assertIn(f'return_{horizon}s', labeled_data.columns)
        
        # 检查标签值
        for horizon in self.horizons:
            labels = labeled_data[f'label_{horizon}s'].dropna()
            self.assertTrue(all(labels.isin([0, 1])))
    
    def test_validate_labels(self):
        """测试标签验证"""
        labeled_data = self.constructor.construct_labels(self.test_data)
        validation_results = self.constructor.validate_labels(labeled_data)
        
        # 检查验证结果结构
        for horizon in self.horizons:
            self.assertIn(f'{horizon}s_label_dist', validation_results)
            self.assertIn(f'{horizon}s_return_stats', validation_results)


class TestSliceAnalyzer(unittest.TestCase):
    """测试切片分析器"""
    
    def setUp(self):
        self.slice_config = {
            'regime': ['Active', 'Quiet'],
            'tod': ['Tokyo', 'London', 'NY']
        }
        self.analyzer = SliceAnalyzer(self.slice_config)
        
        # 创建测试数据
        timestamps = pd.date_range('2025-10-21', periods=1000, freq='1s')
        prices = 100 + np.cumsum(np.random.normal(0, 0.01, 1000))
        
        self.test_data = pd.DataFrame({
            'ts_ms': [int(ts.timestamp() * 1000) for ts in timestamps],
            'price': prices
        })
    
    def test_create_regime_slices(self):
        """测试活跃度切片创建"""
        sliced_data = self.analyzer.create_regime_slices(self.test_data)
        
        self.assertIn('regime', sliced_data.columns)
        self.assertTrue(all(sliced_data['regime'].isin(['Active', 'Quiet', 'Medium'])))
    
    def test_create_time_slices(self):
        """测试时段切片创建"""
        sliced_data = self.analyzer.create_time_slices(self.test_data)
        
        self.assertIn('tod', sliced_data.columns)
        self.assertTrue(all(sliced_data['tod'].isin(['Tokyo', 'London', 'NY', 'Other'])))
    
    def test_create_volatility_slices(self):
        """测试波动率切片创建"""
        sliced_data = self.analyzer.create_volatility_slices(self.test_data)
        
        self.assertIn('vol_regime', sliced_data.columns)
        self.assertTrue(all(sliced_data['vol_regime'].isin(['low', 'mid', 'high'])))
    
    def test_analyze_slices(self):
        """测试切片分析"""
        # 添加切片列
        self.test_data['regime'] = np.random.choice(['Active', 'Quiet'], len(self.test_data))
        self.test_data['tod'] = np.random.choice(['Tokyo', 'London', 'NY'], len(self.test_data))
        
        # 添加标签数据
        for horizon in [60, 180]:
            self.test_data[f'label_{horizon}s'] = np.random.choice([0, 1], len(self.test_data))
        
        metrics = {'test_metric': 0.5}
        slice_results = self.analyzer.analyze_slices(self.test_data, metrics)
        
        self.assertIn('regime', slice_results)
        self.assertIn('tod', slice_results)


class TestDataValidator(unittest.TestCase):
    """测试数据验证器"""
    
    def setUp(self):
        self.validator = DataValidator()
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'ts_ms': [int(datetime.now().timestamp() * 1000) + i for i in range(100)],
            'price': 100 + np.random.normal(0, 1, 100),
            'signal': np.random.normal(0, 1, 100)
        })
    
    def test_check_data_quality(self):
        """测试数据质量检查"""
        quality_report = self.validator.check_data_quality(self.test_data, 'test')
        
        self.assertIn('data_type', quality_report)
        self.assertIn('total_rows', quality_report)
        self.assertIn('missing_data', quality_report)
        self.assertIn('data_quality_score', quality_report)
        
        self.assertEqual(quality_report['data_type'], 'test')
        self.assertEqual(quality_report['total_rows'], 100)
        self.assertGreaterEqual(quality_report['data_quality_score'], 0.0)
        self.assertLessEqual(quality_report['data_quality_score'], 1.0)
    
    def test_check_temporal_consistency(self):
        """测试时间一致性检查"""
        temporal_report = self.validator.check_temporal_consistency(self.test_data)
        
        self.assertIn('temporal_consistency', temporal_report)
        self.assertIn('time_stats', temporal_report)
        self.assertIn('gaps_detected', temporal_report)
    
    def test_check_signal_quality(self):
        """测试信号质量检查"""
        signal_quality = self.validator.check_signal_quality(self.test_data, 'test')
        
        self.assertIn('signal_type', signal_quality)
        self.assertIn('total_signals', signal_quality)
        self.assertIn('valid_signals', signal_quality)
        self.assertIn('signal_quality_score', signal_quality)


class TestOFICVDSignalEvaluator(unittest.TestCase):
    """测试主评估器"""
    
    def setUp(self):
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / 'data' / 'ofi_cvd'
        self.output_dir = Path(self.temp_dir) / 'output'
        
        # 创建测试数据文件
        self._create_test_data()
        
        # 创建评估器
        self.evaluator = OFICVDSignalEvaluator(
            data_root=str(self.data_dir),
            symbols=['ETHUSDT'],
            date_from='2025-10-21',
            date_to='2025-10-21',
            horizons=[60, 180],
            fusion_weights={'w_ofi': 0.6, 'w_cvd': 0.4},
            slices={'regime': ['Active', 'Quiet']},
            output_dir=str(self.output_dir),
            run_tag='test_run'
        )
    
    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """创建测试数据文件"""
        # 创建目录结构
        (self.data_dir / 'date=2025-10-21' / 'symbol=ETHUSDT' / 'kind=prices').mkdir(parents=True)
        (self.data_dir / 'date=2025-10-21' / 'symbol=ETHUSDT' / 'kind=ofi').mkdir(parents=True)
        (self.data_dir / 'date=2025-10-21' / 'symbol=ETHUSDT' / 'kind=cvd').mkdir(parents=True)
        (self.data_dir / 'date=2025-10-21' / 'symbol=ETHUSDT' / 'kind=fusion').mkdir(parents=True)
        (self.data_dir / 'date=2025-10-21' / 'symbol=ETHUSDT' / 'kind=events').mkdir(parents=True)
        
        # 创建测试数据
        timestamps = pd.date_range('2025-10-21', periods=100, freq='1s')
        
        # 价格数据
        prices_data = pd.DataFrame({
            'ts_ms': [int(ts.timestamp() * 1000) for ts in timestamps],
            'event_ts_ms': [int(ts.timestamp() * 1000) for ts in timestamps],
            'symbol': 'ETHUSDT',
            'price': 4000 + np.random.normal(0, 10, 100),
            'qty': np.random.uniform(0.1, 1.0, 100),
            'agg_trade_id': range(100),
            'latency_ms': np.random.uniform(1, 10, 100),
            'recv_rate_tps': np.random.uniform(10, 50, 100)
        })
        prices_data.to_parquet(self.data_dir / 'date=2025-10-21' / 'symbol=ETHUSDT' / 'kind=prices' / 'test.parquet')
        
        # OFI数据
        ofi_data = pd.DataFrame({
            'ts_ms': [int(ts.timestamp() * 1000) for ts in timestamps],
            'symbol': 'ETHUSDT',
            'ofi_value': np.random.normal(0, 1, 100),
            'ofi_z': np.random.normal(0, 1, 100),
            'scale': np.random.uniform(0.5, 2.0, 100),
            'regime': np.random.choice(['Active', 'Quiet'], 100)
        })
        ofi_data.to_parquet(self.data_dir / 'date=2025-10-21' / 'symbol=ETHUSDT' / 'kind=ofi' / 'test.parquet')
        
        # CVD数据
        cvd_data = pd.DataFrame({
            'ts_ms': [int(ts.timestamp() * 1000) for ts in timestamps],
            'symbol': 'ETHUSDT',
            'cvd': np.cumsum(np.random.normal(0, 1, 100)),
            'delta': np.random.normal(0, 1, 100),
            'z_raw': np.random.normal(0, 1, 100),
            'z_cvd': np.random.normal(0, 1, 100),
            'scale': np.random.uniform(0.5, 2.0, 100),
            'sigma_floor': np.random.uniform(0.1, 0.5, 100),
            'floor_used': np.random.choice([True, False], 100),
            'regime': np.random.choice(['Active', 'Quiet'], 100)
        })
        cvd_data.to_parquet(self.data_dir / 'date=2025-10-21' / 'symbol=ETHUSDT' / 'kind=cvd' / 'test.parquet')
        
        # Fusion数据
        fusion_data = pd.DataFrame({
            'ts_ms': [int(ts.timestamp() * 1000) for ts in timestamps],
            'symbol': 'ETHUSDT',
            'score': np.random.normal(0, 1, 100),
            'score_z': np.random.normal(0, 1, 100),
            'regime': np.random.choice(['Active', 'Quiet'], 100)
        })
        fusion_data.to_parquet(self.data_dir / 'date=2025-10-21' / 'symbol=ETHUSDT' / 'kind=fusion' / 'test.parquet')
        
        # Events数据
        events_data = pd.DataFrame({
            'ts_ms': [int(ts.timestamp() * 1000) for ts in timestamps[:10]],
            'symbol': 'ETHUSDT',
            'event_type': np.random.choice(['anomaly', 'divergence'], 10),
            'meta_json': ['{"test": "data"}'] * 10
        })
        events_data.to_parquet(self.data_dir / 'date=2025-10-21' / 'symbol=ETHUSDT' / 'kind=events' / 'test.parquet')
    
    def test_load_data(self):
        """测试数据加载"""
        data = self.evaluator.load_data()
        
        self.assertIn('ETHUSDT', data)
        self.assertIn('prices', data['ETHUSDT'])
        self.assertIn('ofi', data['ETHUSDT'])
        self.assertIn('cvd', data['ETHUSDT'])
        self.assertIn('fusion', data['ETHUSDT'])
        self.assertIn('events', data['ETHUSDT'])
    
    def test_validate_schema(self):
        """测试schema验证"""
        self.evaluator.load_data()
        schema_checks = self.evaluator.validate_schema()
        
        self.assertIn('ETHUSDT', schema_checks)
        self.assertIn('prices', schema_checks['ETHUSDT'])
        self.assertIn('ofi', schema_checks['ETHUSDT'])
        self.assertIn('cvd', schema_checks['ETHUSDT'])
        self.assertIn('fusion', schema_checks['ETHUSDT'])
        self.assertIn('events', schema_checks['ETHUSDT'])
    
    def test_construct_labels(self):
        """测试标签构造"""
        self.evaluator.load_data()
        labeled_data = self.evaluator.construct_labels()
        
        self.assertIn('ETHUSDT', labeled_data)
        for horizon in self.evaluator.horizons:
            self.assertIn(f'label_{horizon}s', labeled_data['ETHUSDT'].columns)
    
    def test_extract_signals(self):
        """测试信号提取"""
        self.evaluator.load_data()
        signals = self.evaluator.extract_signals()
        
        self.assertIn('ETHUSDT', signals)
        self.assertIn('ofi', signals['ETHUSDT'])
        self.assertIn('cvd', signals['ETHUSDT'])
        self.assertIn('fusion', signals['ETHUSDT'])
        self.assertIn('events', signals['ETHUSDT'])
    
    def test_run_analysis(self):
        """测试完整分析流程"""
        results = self.evaluator.run_analysis()
        
        self.assertIsInstance(results, dict)
        self.assertIn('run_tag', results)
        self.assertIn('timestamp', results)
        self.assertIn('best_thresholds', results)


class TestPlotGenerator(unittest.TestCase):
    """测试图表生成器"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.plot_generator = PlotGenerator(Path(self.temp_dir))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_plot_generator_initialization(self):
        """测试图表生成器初始化"""
        self.assertTrue(self.plot_generator.output_dir.exists())
    
    def test_generate_all_plots(self):
        """测试生成所有图表"""
        # 创建示例数据
        metrics_data = {'test': 'data'}
        signal_data = {'test': 'data'}
        slice_data = {'test': 'data'}
        events_data = {'test': 'data'}
        
        # 生成图表
        self.plot_generator.generate_all_plots(metrics_data, signal_data, slice_data, events_data)
        
        # 检查图表文件是否生成
        chart_files = list(self.plot_generator.output_dir.glob('*.png'))
        self.assertGreater(len(chart_files), 0)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
