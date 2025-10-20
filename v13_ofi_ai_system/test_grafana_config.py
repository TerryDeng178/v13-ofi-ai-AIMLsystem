#!/usr/bin/env python3
"""
Grafana配置集成测试脚本

测试Grafana组件与统一配置系统的集成功能
包括配置加载、仪表盘生成、环境变量覆盖等
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.grafana_config import GrafanaConfigLoader, GrafanaConfig
from src.grafana_dashboard_generator import GrafanaDashboardGenerator

def test_config_loading():
    """测试配置加载功能"""
    print("=== 测试配置加载功能 ===")
    
    try:
        # 创建配置加载器
        config_loader = ConfigLoader()
        
        # 创建Grafana配置加载器
        grafana_loader = GrafanaConfigLoader(config_loader)
        
        # 加载配置
        config = grafana_loader.load_config()
        
        # 验证基础配置
        assert hasattr(config, 'dashboard_uid'), "缺少dashboard_uid属性"
        assert hasattr(config, 'refresh_interval'), "缺少refresh_interval属性"
        assert hasattr(config, 'server'), "缺少server属性"
        assert hasattr(config, 'dashboards'), "缺少dashboards属性"
        assert hasattr(config, 'datasources'), "缺少datasources属性"
        assert hasattr(config, 'variables'), "缺少variables属性"
        assert hasattr(config, 'alerting'), "缺少alerting属性"
        
        print("配置加载成功")
        print(f"  - Dashboard UID: {config.dashboard_uid}")
        print(f"  - Refresh Interval: {config.refresh_interval}")
        print(f"  - Server Host: {config.server.host}")
        print(f"  - Server Port: {config.server.port}")
        print(f"  - Dashboards Count: {len(config.dashboards)}")
        print(f"  - Datasources Count: {len(config.datasources)}")
        print(f"  - Variables Count: {len(config.variables)}")
        
        return True
        
    except Exception as e:
        print(f"配置加载失败: {e}")
        return False

def test_dashboard_configs():
    """测试仪表盘配置"""
    print("\n=== 测试仪表盘配置 ===")
    
    try:
        config_loader = ConfigLoader()
        grafana_loader = GrafanaConfigLoader(config_loader)
        
        # 测试策略模式仪表盘配置
        strategy_config = grafana_loader.get_dashboard_config('strategy_mode')
        if strategy_config:
            print("[OK] 策略模式仪表盘配置加载成功")
            print(f"  - UID: {strategy_config.uid}")
            print(f"  - Title: {strategy_config.title}")
            print(f"  - Timezone: {strategy_config.timezone}")
            print(f"  - Tags: {strategy_config.tags}")
        else:
            print("[WARN] 策略模式仪表盘配置未找到，使用默认配置")
        
        # 测试性能仪表盘配置
        performance_config = grafana_loader.get_dashboard_config('performance')
        if performance_config:
            print("[OK] 性能仪表盘配置加载成功")
            print(f"  - UID: {performance_config.uid}")
            print(f"  - Title: {performance_config.title}")
        else:
            print("[WARN] 性能仪表盘配置未找到，使用默认配置")
        
        # 测试告警仪表盘配置
        alerts_config = grafana_loader.get_dashboard_config('alerts')
        if alerts_config:
            print("[OK] 告警仪表盘配置加载成功")
            print(f"  - UID: {alerts_config.uid}")
            print(f"  - Title: {alerts_config.title}")
        else:
            print("[WARN] 告警仪表盘配置未找到，使用默认配置")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 仪表盘配置测试失败: {e}")
        return False

def test_datasource_configs():
    """测试数据源配置"""
    print("\n=== 测试数据源配置 ===")
    
    try:
        config_loader = ConfigLoader()
        grafana_loader = GrafanaConfigLoader(config_loader)
        
        # 测试Prometheus数据源配置
        prometheus_config = grafana_loader.get_datasource_config('prometheus')
        if prometheus_config:
            print("[OK] Prometheus数据源配置加载成功")
            print(f"  - Name: {prometheus_config.name}")
            print(f"  - Type: {prometheus_config.type}")
            print(f"  - URL: {prometheus_config.url}")
            print(f"  - Is Default: {prometheus_config.is_default}")
        else:
            print("[WARN] Prometheus数据源配置未找到")
        
        # 测试Loki数据源配置
        loki_config = grafana_loader.get_datasource_config('loki')
        if loki_config:
            print("[OK] Loki数据源配置加载成功")
            print(f"  - Name: {loki_config.name}")
            print(f"  - Type: {loki_config.type}")
            print(f"  - URL: {loki_config.url}")
        else:
            print("[WARN] Loki数据源配置未找到")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 数据源配置测试失败: {e}")
        return False

def test_variable_configs():
    """测试变量配置"""
    print("\n=== 测试变量配置 ===")
    
    try:
        config_loader = ConfigLoader()
        grafana_loader = GrafanaConfigLoader(config_loader)
        
        # 测试环境变量配置
        env_config = grafana_loader.get_variable_config('env')
        if env_config:
            print("[OK] 环境变量配置加载成功")
            print(f"  - Name: {env_config.name}")
            print(f"  - Type: {env_config.type}")
            print(f"  - Query: {env_config.query}")
            print(f"  - Default Value: {env_config.default_value}")
        else:
            print("[WARN] 环境变量配置未找到")
        
        # 测试交易对变量配置
        symbol_config = grafana_loader.get_variable_config('symbol')
        if symbol_config:
            print("[OK] 交易对变量配置加载成功")
            print(f"  - Name: {symbol_config.name}")
            print(f"  - Type: {symbol_config.type}")
            print(f"  - Query: {symbol_config.query}")
            print(f"  - Multi: {symbol_config.multi}")
            print(f"  - Default Value: {symbol_config.default_value}")
        else:
            print("[WARN] 交易对变量配置未找到")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 变量配置测试失败: {e}")
        return False

def test_alert_rules():
    """测试告警规则配置"""
    print("\n=== 测试告警规则配置 ===")
    
    try:
        config_loader = ConfigLoader()
        grafana_loader = GrafanaConfigLoader(config_loader)
        
        # 获取告警规则
        alert_rules = grafana_loader.get_alert_rules()
        
        if alert_rules:
            print(f"[OK] 告警规则配置加载成功，共{len(alert_rules)}条规则")
            for i, rule in enumerate(alert_rules, 1):
                print(f"  {i}. {rule.alert}")
                print(f"     - 表达式: {rule.expr}")
                print(f"     - 持续时间: {rule.for_duration}")
                print(f"     - 严重级别: {rule.severity}")
                print(f"     - 摘要: {rule.summary}")
        else:
            print("[WARN] 告警规则配置未找到或未启用")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 告警规则配置测试失败: {e}")
        return False

def test_dashboard_generation():
    """测试仪表盘生成功能"""
    print("\n=== 测试仪表盘生成功能 ===")
    
    try:
        config_loader = ConfigLoader()
        generator = GrafanaDashboardGenerator(config_loader)
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 生成策略模式仪表盘
            strategy_dashboard = generator.generate_strategy_mode_dashboard()
            if strategy_dashboard and 'dashboard' in strategy_dashboard:
                print("[OK] 策略模式仪表盘生成成功")
                dashboard = strategy_dashboard['dashboard']
                print(f"  - UID: {dashboard.get('uid', 'N/A')}")
                print(f"  - Title: {dashboard.get('title', 'N/A')}")
                print(f"  - Panels Count: {len(dashboard.get('panels', []))}")
                print(f"  - Variables Count: {len(dashboard.get('templating', {}).get('list', []))}")
            else:
                print("[FAIL] 策略模式仪表盘生成失败")
                return False
            
            # 生成性能仪表盘
            performance_dashboard = generator.generate_performance_dashboard()
            if performance_dashboard and 'dashboard' in performance_dashboard:
                print("[OK] 性能仪表盘生成成功")
                dashboard = performance_dashboard['dashboard']
                print(f"  - UID: {dashboard.get('uid', 'N/A')}")
                print(f"  - Title: {dashboard.get('title', 'N/A')}")
                print(f"  - Panels Count: {len(dashboard.get('panels', []))}")
            else:
                print("[FAIL] 性能仪表盘生成失败")
                return False
            
            # 生成告警仪表盘
            alerts_dashboard = generator.generate_alerts_dashboard()
            if alerts_dashboard and 'dashboard' in alerts_dashboard:
                print("[OK] 告警仪表盘生成成功")
                dashboard = alerts_dashboard['dashboard']
                print(f"  - UID: {dashboard.get('uid', 'N/A')}")
                print(f"  - Title: {dashboard.get('title', 'N/A')}")
                print(f"  - Panels Count: {len(dashboard.get('panels', []))}")
            else:
                print("[FAIL] 告警仪表盘生成失败")
                return False
            
            # 测试保存功能
            if generator.save_dashboard('strategy_mode', str(temp_path / 'test_strategy.json')):
                print("[OK] 仪表盘保存功能正常")
            else:
                print("[FAIL] 仪表盘保存功能失败")
                return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 仪表盘生成测试失败: {e}")
        return False

def test_environment_override():
    """测试环境变量覆盖功能"""
    print("\n=== 测试环境变量覆盖功能 ===")
    
    try:
        # 设置环境变量
        os.environ['V13__MONITORING__GRAFANA__DASHBOARD_UID'] = 'test-override'
        os.environ['V13__MONITORING__GRAFANA__SERVER__HOST'] = 'test-host'
        os.environ['V13__MONITORING__GRAFANA__SERVER__PORT'] = '9999'
        
        # 重新加载配置
        config_loader = ConfigLoader()
        grafana_loader = GrafanaConfigLoader(config_loader)
        config = grafana_loader.load_config()
        
        # 验证环境变量覆盖
        if config.dashboard_uid == 'test-override':
            print("[OK] Dashboard UID环境变量覆盖成功")
        else:
            print(f"[FAIL] Dashboard UID环境变量覆盖失败，期望: test-override，实际: {config.dashboard_uid}")
            return False
        
        if config.server.host == 'test-host':
            print("[OK] Server Host环境变量覆盖成功")
        else:
            print(f"[FAIL] Server Host环境变量覆盖失败，期望: test-host，实际: {config.server.host}")
            return False
        
        if config.server.port == 9999:
            print("[OK] Server Port环境变量覆盖成功")
        else:
            print(f"[FAIL] Server Port环境变量覆盖失败，期望: 9999，实际: {config.server.port}")
            return False
        
        # 清理环境变量
        del os.environ['V13__MONITORING__GRAFANA__DASHBOARD_UID']
        del os.environ['V13__MONITORING__GRAFANA__SERVER__HOST']
        del os.environ['V13__MONITORING__GRAFANA__SERVER__PORT']
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 环境变量覆盖测试失败: {e}")
        return False

def test_json_generation():
    """测试JSON配置生成"""
    print("\n=== 测试JSON配置生成 ===")
    
    try:
        config_loader = ConfigLoader()
        grafana_loader = GrafanaConfigLoader(config_loader)
        
        # 测试数据源JSON生成
        prometheus_json = grafana_loader.generate_datasource_json('prometheus')
        if prometheus_json:
            print("[OK] Prometheus数据源JSON生成成功")
            print(f"  - Name: {prometheus_json.get('name', 'N/A')}")
            print(f"  - Type: {prometheus_json.get('type', 'N/A')}")
            print(f"  - URL: {prometheus_json.get('url', 'N/A')}")
        else:
            print("[WARN] Prometheus数据源JSON生成失败")
        
        # 测试变量JSON生成
        variables_json = grafana_loader.generate_variables_json()
        if variables_json:
            print(f"[OK] 变量JSON生成成功，共{len(variables_json)}个变量")
            for var in variables_json:
                print(f"  - {var.get('name', 'N/A')}: {var.get('type', 'N/A')}")
        else:
            print("[WARN] 变量JSON生成失败")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] JSON配置生成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("Grafana配置集成测试开始")
    print("=" * 50)
    
    tests = [
        ("配置加载功能", test_config_loading),
        ("仪表盘配置", test_dashboard_configs),
        ("数据源配置", test_datasource_configs),
        ("变量配置", test_variable_configs),
        ("告警规则配置", test_alert_rules),
        ("仪表盘生成功能", test_dashboard_generation),
        ("环境变量覆盖", test_environment_override),
        ("JSON配置生成", test_json_generation)
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
    
    print("\n" + "=" * 50)
    print(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("[SUCCESS] 所有测试通过！Grafana配置集成功能正常")
        return True
    else:
        print("[ERROR] 部分测试失败，请检查配置和代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
