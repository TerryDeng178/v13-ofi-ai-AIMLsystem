#!/usr/bin/env python3
"""
策略模式管理器配置集成测试脚本

测试策略模式管理器与统一配置系统的集成功能
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
from src.strategy_mode_config_loader import StrategyModeConfigLoader, StrategyModeConfig
from src.utils.strategy_mode_manager import StrategyModeManager

def test_config_loading():
    """测试配置加载功能"""
    print("=== 测试策略模式管理器配置加载功能 ===")
    
    try:
        # 创建配置加载器
        config_loader = ConfigLoader()
        
        # 创建策略模式配置加载器
        strategy_loader = StrategyModeConfigLoader(config_loader)
        
        # 加载配置
        config = strategy_loader.load_config()
        
        # 验证基础配置
        assert hasattr(config, 'default_mode'), "缺少default_mode属性"
        assert hasattr(config, 'hysteresis'), "缺少hysteresis属性"
        assert hasattr(config, 'schedule'), "缺少schedule属性"
        assert hasattr(config, 'market'), "缺少market属性"
        assert hasattr(config, 'features'), "缺少features属性"
        assert hasattr(config, 'monitoring'), "缺少monitoring属性"
        assert hasattr(config, 'hot_reload'), "缺少hot_reload属性"
        
        print("配置加载成功")
        print(f"  - 默认模式: {config.default_mode}")
        print(f"  - 迟滞窗口: {config.hysteresis.window_secs}s")
        print(f"  - 时间表启用: {config.schedule.enabled}")
        print(f"  - 市场触发器启用: {config.market.enabled}")
        print(f"  - 动态模式启用: {config.features.dynamic_mode_enabled}")
        print(f"  - 监控端口: {config.monitoring.prometheus_port}")
        print(f"  - 热更新启用: {config.hot_reload.enabled}")
        
        return True
        
    except Exception as e:
        print(f"配置加载失败: {e}")
        return False

def test_strategy_mode_manager_creation():
    """测试策略模式管理器创建"""
    print("\n=== 测试策略模式管理器创建 ===")
    
    try:
        # 创建配置加载器
        config_loader = ConfigLoader()
        
        # 使用配置加载器创建策略模式管理器
        manager = StrategyModeManager(config_loader=config_loader)
        
        # 验证配置
        assert manager.config is not None, "配置对象为空"
        assert hasattr(manager, 'mode_setting'), "缺少mode_setting属性"
        assert hasattr(manager, 'schedule_enabled'), "缺少schedule_enabled属性"
        assert hasattr(manager, 'market_enabled'), "缺少market_enabled属性"
        assert hasattr(manager, 'window_secs'), "缺少window_secs属性"
        
        print("策略模式管理器创建成功")
        print(f"  - 模式设置: {manager.mode_setting}")
        print(f"  - 时间表启用: {manager.schedule_enabled}")
        print(f"  - 市场触发器启用: {manager.market_enabled}")
        print(f"  - 迟滞窗口: {manager.window_secs}s")
        print(f"  - 配置来源: 统一配置系统")
        
        return True
        
    except Exception as e:
        print(f"策略模式管理器创建失败: {e}")
        return False

def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n=== 测试向后兼容性 ===")
    
    try:
        # 测试使用原始配置对象创建
        original_config = {
            'strategy': {
                'mode': 'active',
                'hysteresis': {
                    'window_secs': 30,
                    'min_active_windows': 2,
                    'min_quiet_windows': 4
                },
                'triggers': {
                    'schedule': {
                        'enabled': False,
                        'timezone': 'UTC'
                    },
                    'market': {
                        'enabled': True,
                        'min_trades_per_min': 1000
                    }
                }
            },
            'features': {
                'strategy': {
                    'dynamic_mode_enabled': False,
                    'dry_run': True
                }
            }
        }
        
        manager1 = StrategyModeManager(config=original_config)
        assert manager1.mode_setting == 'active', "原始配置未正确传递"
        assert manager1.window_secs == 30, "原始配置未正确传递"
        assert manager1.schedule_enabled == False, "原始配置未正确传递"
        assert manager1.min_trades_per_min == 1000, "原始配置未正确传递"
        
        # 测试使用默认配置创建
        manager2 = StrategyModeManager()
        assert manager2.mode_setting == 'auto', "默认配置未正确应用"
        assert manager2.window_secs == 60, "默认配置未正确应用"
        
        print("向后兼容性测试成功")
        print("  - 原始配置对象: 支持")
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
        os.environ['V13__STRATEGY_MODE__DEFAULT_MODE'] = 'active'
        os.environ['V13__STRATEGY_MODE__HYSTERESIS__WINDOW_SECS'] = '30'
        os.environ['V13__STRATEGY_MODE__TRIGGERS__SCHEDULE__ENABLED'] = 'False'
        os.environ['V13__STRATEGY_MODE__TRIGGERS__MARKET__MIN_TRADES_PER_MIN'] = '1000'
        os.environ['V13__STRATEGY_MODE__FEATURES__DRY_RUN'] = 'True'
        
        # 重新加载配置
        config_loader = ConfigLoader()
        manager = StrategyModeManager(config_loader=config_loader)
        
        # 验证环境变量覆盖
        if manager.mode_setting == 'active':
            print("Default Mode环境变量覆盖成功")
        else:
            print(f"Default Mode环境变量覆盖失败，期望: active，实际: {manager.mode_setting}")
            return False
        
        if manager.window_secs == 30:
            print("Window Secs环境变量覆盖成功")
        else:
            print(f"Window Secs环境变量覆盖失败，期望: 30，实际: {manager.window_secs}")
            return False
        
        if manager.schedule_enabled == False:
            print("Schedule Enabled环境变量覆盖成功")
        else:
            print(f"Schedule Enabled环境变量覆盖失败，期望: False，实际: {manager.schedule_enabled}")
            return False
        
        if manager.min_trades_per_min == 1000:
            print("Min Trades Per Min环境变量覆盖成功")
        else:
            print(f"Min Trades Per Min环境变量覆盖失败，期望: 1000，实际: {manager.min_trades_per_min}")
            return False
        
        if manager.dry_run == True:
            print("Dry Run环境变量覆盖成功")
        else:
            print(f"Dry Run环境变量覆盖失败，期望: True，实际: {manager.dry_run}")
            return False
        
        # 清理环境变量
        del os.environ['V13__STRATEGY_MODE__DEFAULT_MODE']
        del os.environ['V13__STRATEGY_MODE__HYSTERESIS__WINDOW_SECS']
        del os.environ['V13__STRATEGY_MODE__TRIGGERS__SCHEDULE__ENABLED']
        del os.environ['V13__STRATEGY_MODE__TRIGGERS__MARKET__MIN_TRADES_PER_MIN']
        del os.environ['V13__STRATEGY_MODE__FEATURES__DRY_RUN']
        
        return True
        
    except Exception as e:
        print(f"环境变量覆盖测试失败: {e}")
        return False

def test_config_methods():
    """测试配置方法"""
    print("\n=== 测试配置方法 ===")
    
    try:
        config_loader = ConfigLoader()
        strategy_loader = StrategyModeConfigLoader(config_loader)
        
        # 测试迟滞配置
        hysteresis = strategy_loader.get_hysteresis_config()
        assert hasattr(hysteresis, 'window_secs'), "迟滞配置缺少window_secs"
        assert hasattr(hysteresis, 'min_active_windows'), "迟滞配置缺少min_active_windows"
        print("迟滞配置方法正常")
        
        # 测试时间表配置
        schedule = strategy_loader.get_schedule_config()
        assert hasattr(schedule, 'enabled'), "时间表配置缺少enabled"
        assert hasattr(schedule, 'timezone'), "时间表配置缺少timezone"
        print("时间表配置方法正常")
        
        # 测试市场配置
        market = strategy_loader.get_market_config()
        assert hasattr(market, 'enabled'), "市场配置缺少enabled"
        assert hasattr(market, 'min_trades_per_min'), "市场配置缺少min_trades_per_min"
        print("市场配置方法正常")
        
        # 测试特性配置
        features = strategy_loader.get_features_config()
        assert hasattr(features, 'dynamic_mode_enabled'), "特性配置缺少dynamic_mode_enabled"
        assert hasattr(features, 'dry_run'), "特性配置缺少dry_run"
        print("特性配置方法正常")
        
        # 测试监控配置
        monitoring = strategy_loader.get_monitoring_config()
        assert hasattr(monitoring, 'prometheus_port'), "监控配置缺少prometheus_port"
        assert hasattr(monitoring, 'alerts_enabled'), "监控配置缺少alerts_enabled"
        print("监控配置方法正常")
        
        # 测试热更新配置
        hot_reload = strategy_loader.get_hot_reload_config()
        assert hasattr(hot_reload, 'enabled'), "热更新配置缺少enabled"
        assert hasattr(hot_reload, 'reload_delay'), "热更新配置缺少reload_delay"
        print("热更新配置方法正常")
        
        return True
        
    except Exception as e:
        print(f"配置方法测试失败: {e}")
        return False

def test_strategy_mode_functionality():
    """测试策略模式管理器功能"""
    print("\n=== 测试策略模式管理器功能 ===")
    
    try:
        config_loader = ConfigLoader()
        manager = StrategyModeManager(config_loader=config_loader)
        
        # 检查基础状态
        assert hasattr(manager, 'current_mode'), "缺少current_mode属性"
        assert hasattr(manager, 'mode_setting'), "缺少mode_setting属性"
        assert hasattr(manager, 'schedule_enabled'), "缺少schedule_enabled属性"
        assert hasattr(manager, 'market_enabled'), "缺少market_enabled属性"
        
        # 检查迟滞配置
        assert hasattr(manager, 'window_secs'), "缺少window_secs属性"
        assert hasattr(manager, 'min_active_windows'), "缺少min_active_windows属性"
        assert hasattr(manager, 'min_quiet_windows'), "缺少min_quiet_windows属性"
        
        # 检查市场配置
        assert hasattr(manager, 'min_trades_per_min'), "缺少min_trades_per_min属性"
        assert hasattr(manager, 'min_quote_updates_per_sec'), "缺少min_quote_updates_per_sec属性"
        assert hasattr(manager, 'max_spread_bps'), "缺少max_spread_bps属性"
        
        # 检查特性配置
        assert hasattr(manager, 'dynamic_mode_enabled'), "缺少dynamic_mode_enabled属性"
        assert hasattr(manager, 'dry_run'), "缺少dry_run属性"
        
        print("策略模式管理器功能正常")
        print(f"  - 当前模式: {manager.current_mode}")
        print(f"  - 模式设置: {manager.mode_setting}")
        print(f"  - 时间表启用: {manager.schedule_enabled}")
        print(f"  - 市场触发器启用: {manager.market_enabled}")
        print(f"  - 迟滞窗口: {manager.window_secs}s")
        print(f"  - 动态模式: {'启用' if manager.dynamic_mode_enabled else '禁用'}")
        print(f"  - 干运行: {'是' if manager.dry_run else '否'}")
        
        return True
        
    except Exception as e:
        print(f"策略模式管理器功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("策略模式管理器配置集成测试开始")
    print("=" * 60)
    
    tests = [
        ("配置加载功能", test_config_loading),
        ("策略模式管理器创建", test_strategy_mode_manager_creation),
        ("向后兼容性", test_backward_compatibility),
        ("环境变量覆盖", test_environment_override),
        ("配置方法", test_config_methods),
        ("策略模式管理器功能", test_strategy_mode_functionality)
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
        print("[SUCCESS] 所有测试通过！策略模式管理器配置集成功能正常")
        return True
    else:
        print("[ERROR] 部分测试失败，请检查配置和代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
