#!/usr/bin/env python3
"""
OFI+CVD数据采集系统测试脚本

验证系统组件是否正常工作，包括：
- 配置加载
- 组件初始化
- 数据格式验证
- 监控指标
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

def test_config_loading():
    """测试配置加载"""
    print("测试配置加载...")
    
    try:
        from src.utils import ConfigLoader
        config_loader = ConfigLoader()
        config = config_loader.load_config()
        
        # 检查必要的配置项
        required_sections = ['data_harvest', 'divergence_detection', 'strategy']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"缺少配置节: {section}")
        
        print("✅ 配置加载成功")
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_component_initialization():
    """测试组件初始化"""
    print("测试组件初始化...")
    
    try:
        from src.features import OFICalculator, CVDCalculator
        from src.regimes import RegimeClassifier
        from src.signals import DivergenceDetector
        
        # 初始化OFI计算器
        ofi_calculator = OFICalculator(
            levels=5,
            weights=[0.4, 0.3, 0.2, 0.08, 0.02],
            z_mode='delta',
            scale_mode='hybrid',
            mad_multiplier=1.8,
            scale_fast_weight=0.20,
            half_life_trades=1200,
            winsor_limit=8.0
        )
        
        # 初始化CVD计算器
        cvd_calculator = CVDCalculator(
            z_mode='delta',
            scale_mode='hybrid',
            mad_multiplier=1.8,
            scale_fast_weight=0.20,
            half_life_trades=1200,
            winsor_limit=8.0
        )
        
        # 初始化其他组件
        regime_classifier = RegimeClassifier()
        divergence_detector = DivergenceDetector()
        
        print("✅ 组件初始化成功")
        return True
    except Exception as e:
        print(f"❌ 组件初始化失败: {e}")
        return False

def test_data_processing():
    """测试数据处理"""
    print("测试数据处理...")
    
    try:
        from src.features import OFICalculator, CVDCalculator
        from src.regimes import RegimeClassifier
        
        # 创建测试数据
        test_data = {
            'ts_ms': 1640995200000,
            'price': 50000.0,
            'qty': 0.001,
            'is_buyer_maker': False,
            'agg_trade_id': 'test_123'
        }
        
        # 初始化组件
        ofi_calculator = OFICalculator(
            levels=5,
            weights=[0.4, 0.3, 0.2, 0.08, 0.02],
            z_mode='delta',
            scale_mode='hybrid',
            mad_multiplier=1.8,
            scale_fast_weight=0.20,
            half_life_trades=1200,
            winsor_limit=8.0
        )
        
        cvd_calculator = CVDCalculator(
            z_mode='delta',
            scale_mode='hybrid',
            mad_multiplier=1.8,
            scale_fast_weight=0.20,
            half_life_trades=1200,
            winsor_limit=8.0
        )
        
        regime_classifier = RegimeClassifier()
        
        # 处理数据
        ofi_result = ofi_calculator.update(test_data)
        cvd_result = cvd_calculator.update(test_data)
        regime = regime_classifier.classify(test_data)
        
        print("✅ 数据处理成功")
        return True
    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        return False

def test_parquet_schema():
    """测试Parquet模式"""
    print("测试Parquet模式...")
    
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        import tempfile
        
        # 创建测试数据
        test_data = {
            'ts_ms': [1640995200000, 1640995201000],
            'event_ts_ms': [1640995200000, 1640995201000],
            'symbol': ['BTCUSDT', 'BTCUSDT'],
            'price': [50000.0, 50001.0],
            'qty': [0.001, 0.002],
            'agg_trade_id': ['test_1', 'test_2'],
            'latency_ms': [10.5, 12.3],
            'recv_rate_tps': [1.5, 1.6]
        }
        
        df = pd.DataFrame(test_data)
        
        # 测试Parquet写入和读取
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            df.to_parquet(tmp.name, compression='snappy', index=False)
            
            # 读取验证
            df_read = pd.read_parquet(tmp.name)
            
            # 验证数据完整性
            assert len(df_read) == len(df)
            assert list(df_read.columns) == list(df.columns)
            
            # 清理临时文件
            os.unlink(tmp.name)
        
        print("✅ Parquet模式测试成功")
        return True
    except Exception as e:
        print(f"❌ Parquet模式测试失败: {e}")
        return False

def test_monitoring_metrics():
    """测试监控指标"""
    print("测试监控指标...")
    
    try:
        from prometheus_client import Counter, Histogram, Gauge
        
        # 创建测试指标
        test_counter = Counter('test_counter', 'Test counter')
        test_histogram = Histogram('test_histogram', 'Test histogram')
        test_gauge = Gauge('test_gauge', 'Test gauge')
        
        # 测试指标操作
        test_counter.inc(5)
        test_histogram.observe(1.5)
        test_gauge.set(42.0)
        
        print("✅ 监控指标测试成功")
        return True
    except Exception as e:
        print(f"❌ 监控指标测试失败: {e}")
        return False

def test_validation_script():
    """测试验证脚本"""
    print("测试验证脚本...")
    
    try:
        import subprocess
        import tempfile
        import os
        
        # 创建临时测试数据
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试Parquet文件
            import pandas as pd
            import pyarrow.parquet as pq
            
            test_data = {
                'ts_ms': [1640995200000, 1640995201000, 1640995202000],
                'event_ts_ms': [1640995200000, 1640995201000, 1640995202000],
                'symbol': ['BTCUSDT', 'BTCUSDT', 'BTCUSDT'],
                'price': [50000.0, 50001.0, 50002.0],
                'qty': [0.001, 0.002, 0.003],
                'agg_trade_id': ['test_1', 'test_2', 'test_3'],
                'latency_ms': [10.5, 12.3, 8.7],
                'recv_rate_tps': [1.5, 1.6, 1.4]
            }
            
            df = pd.DataFrame(test_data)
            
            # 创建目录结构
            os.makedirs(f"{temp_dir}/date=2025-01-20/symbol=BTCUSDT/kind=prices", exist_ok=True)
            
            # 写入测试文件
            df.to_parquet(f"{temp_dir}/date=2025-01-20/symbol=BTCUSDT/kind=prices/part-1.parquet", 
                         compression='snappy', index=False)
            
            # 运行验证脚本
            result = subprocess.run([
                sys.executable, 
                'scripts/validate_ofi_cvd_harvest.py',
                '--base-dir', temp_dir,
                '--output-dir', f"{temp_dir}/reports"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            if result.returncode == 0:
                print("✅ 验证脚本测试成功")
                return True
            else:
                print(f"❌ 验证脚本测试失败: {result.stderr}")
                return False
    except Exception as e:
        print(f"❌ 验证脚本测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("OFI+CVD数据采集系统测试")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_component_initialization,
        test_data_processing,
        test_parquet_schema,
        test_monitoring_metrics,
        test_validation_script
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统准备就绪。")
        return 0
    else:
        print("❌ 部分测试失败，请检查系统配置。")
        return 1

if __name__ == '__main__':
    exit(main())

