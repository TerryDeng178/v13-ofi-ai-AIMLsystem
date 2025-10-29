# -*- coding: utf-8 -*-
"""
策略模式管理器手动走查脚本 (Manual Mode Walkthrough)

用于手动验证策略模式管理器的行为，包括：
- 模式切换逻辑
- 事件生成
- 指标更新
- 配置参数影响

运行方式：
python scripts/manual_mode_walkthrough.py

作者: V13 Team
创建日期: 2025-01-27
"""

import sys
import os
import time
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.strategy_mode_manager import StrategyModeManager, MarketActivity, StrategyMode, TriggerReason


def print_separator(title: str):
    """打印分隔符"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_config_info(manager: StrategyModeManager):
    """打印配置信息"""
    print_separator("配置信息")
    print(f"组合逻辑: {manager.combine_logic}")
    print(f"迟滞窗口: {manager.hysteresis_window_secs}秒")
    print(f"Active确认窗口: {manager.min_active_windows}")
    print(f"Quiet确认窗口: {manager.min_quiet_windows}")
    print(f"市场窗口: {manager.market_window_secs}秒")
    print(f"市场门槛: trades>={manager.min_trades_per_min}, quotes>={manager.min_quote_updates_per_sec}")


def print_current_state(manager: StrategyModeManager):
    """打印当前状态"""
    print_separator("当前状态")
    print(f"当前模式: {manager.current_mode.value}")
    print(f"历史记录长度: {len(manager.activity_history)}")
    print(f"市场样本数: {len(manager.market_samples)}")
    
    if manager.activity_history:
        print("最近历史:")
        for i, (ts, active) in enumerate(list(manager.activity_history)[-5:]):
            print(f"  [{i+1}] {datetime.fromtimestamp(ts).strftime('%H:%M:%S')} -> {'ACTIVE' if active else 'QUIET'}")


def print_metrics_keys():
    """打印关键指标键名"""
    print_separator("关键指标键名")
    print("模式相关:")
    print("  - strategy_mode_current: 当前模式 (0=quiet, 1=active)")
    print("  - strategy_mode_last_change_timestamp: 最后切换时间戳")
    print("  - strategy_time_in_mode_seconds_total: 各模式累计时长")
    print("  - strategy_mode_transitions_total: 模式切换次数")
    
    print("\n市场门控:")
    print("  - strategy_market_gate_basic_pass: 基础门槛 (0/1)")
    print("  - strategy_market_gate_quality_pass: 质量过滤 (0/1)")
    print("  - strategy_market_gate_window_pass: 窗口门槛 (0/1)")
    print("  - strategy_market_samples_window_size: 窗口样本数")
    print("  - strategy_market_samples_coverage_seconds: 窗口覆盖时长")
    
    print("\n触发器:")
    print("  - strategy_trigger_schedule_active: 时间表活跃 (0/1)")
    print("  - strategy_trigger_market_active: 市场活跃 (0/1)")


def create_test_scenarios():
    """创建测试场景"""
    scenarios = {
        'active_market': {
            'name': '活跃市场',
            'trades_per_min': 200.0,
            'quote_updates_per_sec': 50.0,
            'spread_bps': 3.0,
            'volatility_bps': 5.0,
            'volume_usd': 500000.0
        },
        'quiet_market': {
            'name': '不活跃市场',
            'trades_per_min': 50.0,
            'quote_updates_per_sec': 10.0,
            'spread_bps': 8.0,
            'volatility_bps': 1.0,
            'volume_usd': 50000.0
        },
        'borderline_market': {
            'name': '边界市场',
            'trades_per_min': 100.0,  # 刚好达到门槛
            'quote_updates_per_sec': 20.0,  # 刚好达到门槛
            'spread_bps': 5.0,  # 刚好达到门槛
            'volatility_bps': 2.0,  # 刚好达到门槛
            'volume_usd': 100000.0  # 刚好达到门槛
        }
    }
    return scenarios


def run_scenario(manager: StrategyModeManager, scenario_name: str, scenario_data: dict, 
                schedule_active: bool = True, iterations: int = 3):
    """运行测试场景"""
    print_separator(f"场景: {scenario_data['name']} (Schedule: {'ON' if schedule_active else 'OFF'})")
    
    with manager._samples_lock:  # 确保线程安全
        for i in range(iterations):
            print(f"\n--- 第 {i+1} 次更新 ---")
            
            # 创建市场数据
            activity = MarketActivity()
            activity.trades_per_min = scenario_data['trades_per_min']
            activity.quote_updates_per_sec = scenario_data['quote_updates_per_sec']
            activity.spread_bps = scenario_data['spread_bps']
            activity.volatility_bps = scenario_data['volatility_bps']
            activity.volume_usd = scenario_data['volume_usd']
            
            # Mock schedule 状态
            with manager._samples_lock:
                original_check = manager.check_schedule_active
                manager.check_schedule_active = lambda: schedule_active
                
                # 执行更新
                start_time = time.time()
                result = manager.update_mode(activity)
                duration = (time.time() - start_time) * 1000
                
                # 恢复原始方法
                manager.check_schedule_active = original_check
            
            # 打印结果
            print(f"目标模式: {manager.current_mode.value}")
            print(f"处理耗时: {duration:.2f}ms")
            
            if result:
                print("事件:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print("无事件 (未切换)")
            
            print_current_state(manager)
            
            # 短暂延迟模拟时序
            time.sleep(0.02)


def test_or_vs_and_logic():
    """测试 OR vs AND 逻辑"""
    print_separator("OR vs AND 逻辑对比")
    
    base_config = {
        'strategy': {
            'mode': 'auto',
            'hysteresis': {
                'window_secs': 60,
                'min_active_windows': 1,  # 简化测试
                'min_quiet_windows': 1
            },
            'triggers': {
                'schedule': {
                    'enabled': True,
                    'timezone': 'UTC',  # 使用 UTC 避免时区问题
                    'calendar': 'CRYPTO',
                    'enabled_weekdays': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    'holidays': [],
                    'active_windows': [
                        {'start': '00:00', 'end': '23:59', 'timezone': 'UTC'}
                    ],
                    'wrap_midnight': True
                },
                'market': {
                    'enabled': True,
                    'window_secs': 60,
                    'min_trades_per_min': 100,
                    'min_quote_updates_per_sec': 20,
                    'max_spread_bps': 5,
                    'min_volatility_bps': 2,
                    'min_volume_usd': 100000,
                    'use_median': False,
                    'winsorize_percentile': 95
                }
            }
        },
        'features': {
            'strategy': {
                'dynamic_mode_enabled': True,
                'dry_run': False
            }
        }
    }
    
    scenarios = create_test_scenarios()
    
    # 测试 OR 逻辑
    print("\n>>> OR 逻辑测试")
    config_or = base_config.copy()
    config_or['strategy']['triggers']['combine_logic'] = 'OR'
    manager_or = StrategyModeManager(config_or)
    print_config_info(manager_or)
    
    # Schedule ON, Market QUIET -> 应该 ACTIVE (OR)
    run_scenario(manager_or, 'quiet_market', scenarios['quiet_market'], 
                schedule_active=True, iterations=2)
    
    # 测试 AND 逻辑
    print("\n>>> AND 逻辑测试")
    config_and = base_config.copy()
    config_and['strategy']['triggers']['combine_logic'] = 'AND'
    manager_and = StrategyModeManager(config_and)
    print_config_info(manager_and)
    
    # Schedule ON, Market QUIET -> 应该 QUIET (AND)
    run_scenario(manager_and, 'quiet_market', scenarios['quiet_market'], 
                schedule_active=True, iterations=2)


def main():
    """主函数"""
    print_separator("策略模式管理器手动走查")
    print("本脚本用于验证策略模式管理器的行为")
    print("包括模式切换、事件生成、指标更新等")
    
    # 打印指标键名
    print_metrics_keys()
    
    # 测试 OR vs AND 逻辑
    test_or_vs_and_logic()
    
    # 测试迟滞机制
    print_separator("迟滞机制测试")
    config = {
        'strategy': {
            'mode': 'auto',
            'hysteresis': {
                'window_secs': 60,
                'min_active_windows': 2,  # 需要2次确认
                'min_quiet_windows': 3    # 需要3次确认
            },
            'triggers': {
                'combine_logic': 'OR',
                'schedule': {
                    'enabled': True,
                    'timezone': 'UTC',  # 使用 UTC 避免时区问题
                    'calendar': 'CRYPTO',
                    'enabled_weekdays': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    'holidays': [],
                    'active_windows': [
                        {'start': '00:00', 'end': '23:59', 'timezone': 'UTC'}
                    ],
                    'wrap_midnight': True
                },
                'market': {
                    'enabled': True,
                    'window_secs': 60,
                    'min_trades_per_min': 100,
                    'min_quote_updates_per_sec': 20,
                    'max_spread_bps': 5,
                    'min_volatility_bps': 2,
                    'min_volume_usd': 100000,
                    'use_median': False,
                    'winsorize_percentile': 95
                }
            }
        },
        'features': {
            'strategy': {
                'dynamic_mode_enabled': True,
                'dry_run': False
            }
        }
    }
    
    manager = StrategyModeManager(config)
    print_config_info(manager)
    
    scenarios = create_test_scenarios()
    
    # 测试切换到 ACTIVE (需要2次确认)
    print("\n>>> 切换到 ACTIVE (需要2次确认)")
    run_scenario(manager, 'active_market', scenarios['active_market'], 
                schedule_active=True, iterations=3)
    
    # 测试切换到 QUIET (需要3次确认)
    print("\n>>> 切换到 QUIET (需要3次确认)")
    run_scenario(manager, 'quiet_market', scenarios['quiet_market'], 
                schedule_active=False, iterations=4)
    
    print_separator("走查完成")
    print("关键观察点:")
    print("1. OR逻辑: schedule OR market 任一满足即可")
    print("2. AND逻辑: schedule AND market 必须同时满足")
    print("3. 迟滞机制: 需要连续确认才切换")
    print("4. 事件结构: 包含完整的触发信息")
    print("5. 指标更新: 实时反映当前状态")


if __name__ == '__main__':
    main()
