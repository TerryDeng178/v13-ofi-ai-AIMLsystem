#!/usr/bin/env python3
"""
融合指标收集器配置集成测试脚本

测试融合指标收集器与统一配置系统的集成功能
包括配置加载、参数验证、环境变量覆盖等
"""

import sys
import os
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.fusion_metrics_config_loader import FusionMetricsCollectorConfigLoader, FusionMetricsCollectorConfig
from src.fusion_metrics import FusionMetricsCollector
from src.ofi_cvd_fusion import OFI_CVD_Fusion

def test_config_loading():
    """测试配置加载功能"""
    print("=== 测试融合指标收集器配置加载功能 ===")
    
    try:
        # 创建配置加载器
        config_loader = ConfigLoader()
        
        # 创建融合指标收集器配置加载器
        fusion_loader = FusionMetricsCollectorConfigLoader(config_loader)
        
        # 加载配置
        config = fusion_loader.load_config()
        
        # 验证基础配置
        assert hasattr(config, 'enabled'), "缺少enabled属性"
        assert hasattr(config, 'history'), "缺少history属性"
        assert hasattr(config, 'collection'), "缺少collection属性"
        assert hasattr(config, 'performance'), "缺少performance属性"
        assert hasattr(config, 'monitoring'), "缺少monitoring属性"
        assert hasattr(config, 'hot_reload'), "缺少hot_reload属性"
        
        print("配置加载成功")
        print(f"  - 启用状态: {config.enabled}")
        print(f"  - 最大记录数: {config.history.max_records}")
        print(f"  - 更新间隔: {config.collection.update_interval}s")
        print(f"  - 批处理大小: {config.collection.batch_size}")
        print(f"  - 最大收集率: {config.performance.max_collection_rate}/s")
        print(f"  - 监控端口: {config.monitoring.prometheus_port}")
        print(f"  - 热更新启用: {config.hot_reload.enabled}")
        
        return True
        
    except Exception as e:
        print(f"配置加载失败: {e}")
        return False

def test_fusion_metrics_collector_creation():
    """测试融合指标收集器创建"""
    print("\n=== 测试融合指标收集器创建 ===")
    
    try:
        # 创建配置加载器
        config_loader = ConfigLoader()
        
        # 创建融合器
        fusion = OFI_CVD_Fusion()
        
        # 使用配置加载器创建融合指标收集器
        collector = FusionMetricsCollector(fusion, config_loader=config_loader)
        
        # 验证配置
        assert collector.config is not None, "配置对象为空"
        assert hasattr(collector, 'max_history'), "缺少max_history属性"
        assert hasattr(collector, 'metrics_history'), "缺少metrics_history属性"
        assert hasattr(collector, 'fusion'), "缺少fusion属性"
        
        print("融合指标收集器创建成功")
        print(f"  - 最大历史记录: {collector.max_history}")
        print(f"  - 历史记录数量: {len(collector.metrics_history)}")
        print(f"  - 融合器类型: {type(collector.fusion).__name__}")
        print(f"  - 配置来源: 统一配置系统")
        
        return True
        
    except Exception as e:
        print(f"融合指标收集器创建失败: {e}")
        return False

def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n=== 测试向后兼容性 ===")
    
    try:
        # 创建融合器
        fusion = OFI_CVD_Fusion()
        
        # 测试使用默认配置创建
        collector1 = FusionMetricsCollector(fusion)
        assert collector1.max_history == 1000, "默认配置未正确应用"
        assert collector1.config is not None, "默认配置未正确应用"
        
        print("向后兼容性测试成功")
        print("  - 默认配置: 支持")
        print("  - 统一配置系统: 支持")
        
        return True
        
    except Exception as e:
        print(f"向后兼容性测试失败: {e}")
        return False

def test_environment_override():
    """测试环境变量覆盖功能"""
    print("\n=== 测试环境变量覆盖功能 ===")
    
    try:
        # 设置环境变量
        os.environ['V13__FUSION_METRICS_COLLECTOR__ENABLED'] = 'False'
        os.environ['V13__FUSION_METRICS_COLLECTOR__HISTORY__MAX_RECORDS'] = '2000'
        os.environ['V13__FUSION_METRICS_COLLECTOR__COLLECTION__UPDATE_INTERVAL'] = '0.5'
        os.environ['V13__FUSION_METRICS_COLLECTOR__PERFORMANCE__MAX_COLLECTION_RATE'] = '200'
        os.environ['V13__FUSION_METRICS_COLLECTOR__MONITORING__PROMETHEUS__PORT'] = '8007'
        
        # 重新加载配置
        config_loader = ConfigLoader()
        fusion = OFI_CVD_Fusion()
        collector = FusionMetricsCollector(fusion, config_loader=config_loader)
        
        # 验证环境变量覆盖
        if collector.config.enabled == False:
            print("Enabled环境变量覆盖成功")
        else:
            print(f"Enabled环境变量覆盖失败，期望: False，实际: {collector.config.enabled}")
            return False
        
        if collector.max_history == 2000:
            print("Max Records环境变量覆盖成功")
        else:
            print(f"Max Records环境变量覆盖失败，期望: 2000，实际: {collector.max_history}")
            return False
        
        if collector.config.collection.update_interval == 0.5:
            print("Update Interval环境变量覆盖成功")
        else:
            print(f"Update Interval环境变量覆盖失败，期望: 0.5，实际: {collector.config.collection.update_interval}")
            return False
        
        if collector.config.performance.max_collection_rate == 200:
            print("Max Collection Rate环境变量覆盖成功")
        else:
            print(f"Max Collection Rate环境变量覆盖失败，期望: 200，实际: {collector.config.performance.max_collection_rate}")
            return False
        
        if collector.config.monitoring.prometheus_port == 8007:
            print("Prometheus Port环境变量覆盖成功")
        else:
            print(f"Prometheus Port环境变量覆盖失败，期望: 8007，实际: {collector.config.monitoring.prometheus_port}")
            return False
        
        # 清理环境变量
        del os.environ['V13__FUSION_METRICS_COLLECTOR__ENABLED']
        del os.environ['V13__FUSION_METRICS_COLLECTOR__HISTORY__MAX_RECORDS']
        del os.environ['V13__FUSION_METRICS_COLLECTOR__COLLECTION__UPDATE_INTERVAL']
        del os.environ['V13__FUSION_METRICS_COLLECTOR__PERFORMANCE__MAX_COLLECTION_RATE']
        del os.environ['V13__FUSION_METRICS_COLLECTOR__MONITORING__PROMETHEUS__PORT']
        
        return True
        
    except Exception as e:
        print(f"环境变量覆盖测试失败: {e}")
        return False

def test_config_methods():
    """测试配置方法"""
    print("\n=== 测试配置方法 ===")
    
    try:
        config_loader = ConfigLoader()
        fusion_loader = FusionMetricsCollectorConfigLoader(config_loader)
        
        # 测试历史记录配置
        history = fusion_loader.get_history_config()
        assert hasattr(history, 'max_records'), "历史记录配置缺少max_records"
        assert hasattr(history, 'cleanup_interval'), "历史记录配置缺少cleanup_interval"
        print("历史记录配置方法正常")
        
        # 测试收集配置
        collection = fusion_loader.get_collection_config()
        assert hasattr(collection, 'update_interval'), "收集配置缺少update_interval"
        assert hasattr(collection, 'batch_size'), "收集配置缺少batch_size"
        print("收集配置方法正常")
        
        # 测试性能配置
        performance = fusion_loader.get_performance_config()
        assert hasattr(performance, 'max_collection_rate'), "性能配置缺少max_collection_rate"
        assert hasattr(performance, 'memory_limit_mb'), "性能配置缺少memory_limit_mb"
        print("性能配置方法正常")
        
        # 测试监控配置
        monitoring = fusion_loader.get_monitoring_config()
        assert hasattr(monitoring, 'prometheus_port'), "监控配置缺少prometheus_port"
        assert hasattr(monitoring, 'alerts_enabled'), "监控配置缺少alerts_enabled"
        print("监控配置方法正常")
        
        # 测试热更新配置
        hot_reload = fusion_loader.get_hot_reload_config()
        assert hasattr(hot_reload, 'enabled'), "热更新配置缺少enabled"
        assert hasattr(hot_reload, 'reload_delay'), "热更新配置缺少reload_delay"
        print("热更新配置方法正常")
        
        return True
        
    except Exception as e:
        print(f"配置方法测试失败: {e}")
        return False

def test_fusion_metrics_functionality():
    """测试融合指标收集器功能"""
    print("\n=== 测试融合指标收集器功能 ===")
    
    try:
        config_loader = ConfigLoader()
        fusion = OFI_CVD_Fusion()
        collector = FusionMetricsCollector(fusion, config_loader=config_loader)
        
        # 检查基础状态
        assert hasattr(collector, 'fusion'), "缺少fusion属性"
        assert hasattr(collector, 'metrics_history'), "缺少metrics_history属性"
        assert hasattr(collector, 'max_history'), "缺少max_history属性"
        assert hasattr(collector, 'config'), "缺少config属性"
        
        # 测试收集指标
        import time
        current_time = time.time()
        
        metrics = collector.collect_metrics(
            z_ofi=1.5,
            z_cvd=2.0,
            ts=current_time,
            price=50000.0,
            lag_sec=0.1
        )
        
        assert metrics is not None, "指标收集失败"
        assert hasattr(metrics, 'fusion_score'), "指标缺少fusion_score"
        assert hasattr(metrics, 'signal'), "指标缺少signal"
        assert hasattr(metrics, 'consistency'), "指标缺少consistency"
        
        # 检查历史记录
        assert len(collector.metrics_history) == 1, f"历史记录数量错误: {len(collector.metrics_history)}"
        
        print("融合指标收集器功能正常")
        print(f"  - 融合器类型: {type(collector.fusion).__name__}")
        print(f"  - 历史记录数量: {len(collector.metrics_history)}")
        print(f"  - 最大历史记录: {collector.max_history}")
        print(f"  - 融合得分: {metrics.fusion_score}")
        print(f"  - 信号类型: {metrics.signal}")
        print(f"  - 一致性: {metrics.consistency}")
        
        return True
        
    except Exception as e:
        print(f"融合指标收集器功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("融合指标收集器配置集成测试开始")
    print("=" * 60)
    
    tests = [
        ("配置加载功能", test_config_loading),
        ("融合指标收集器创建", test_fusion_metrics_collector_creation),
        ("向后兼容性", test_backward_compatibility),
        ("环境变量覆盖", test_environment_override),
        ("配置方法", test_config_methods),
        ("融合指标收集器功能", test_fusion_metrics_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"[FAIL] {test_name} 测试失败")
        except Exception as e:
            print(f"[FAIL] {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("[SUCCESS] 所有测试通过！融合指标收集器配置集成功能正常")
        return True
    else:
        print("[ERROR] 部分测试失败，请检查配置和代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
