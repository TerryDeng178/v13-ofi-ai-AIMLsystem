#!/usr/bin/env python3
"""
调试策略模式管理器配置加载
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.strategy_mode_config_loader import StrategyModeConfigLoader

def debug_strategy_config():
    """调试策略模式配置加载"""
    print("=== 调试策略模式管理器配置加载 ===")
    
    try:
        # 创建配置加载器
        config_loader = ConfigLoader()
        print("配置加载器创建成功")
        
        # 测试转换为原始配置格式
        print("\n=== 测试转换为原始配置格式 ===")
        
        # 模拟StrategyModeManager的_load_from_config_loader方法
        try:
            
            strategy_config_loader = StrategyModeConfigLoader(config_loader)
            config = strategy_config_loader.load_config()
            
            # 转换为原始配置格式
            result = {
                'strategy': {
                    'mode': config.default_mode,
                    'hysteresis': {
                        'window_secs': config.hysteresis.window_secs,
                        'min_active_windows': config.hysteresis.min_active_windows,
                        'min_quiet_windows': config.hysteresis.min_quiet_windows
                    },
                    'triggers': {
                        'schedule': {
                            'enabled': config.schedule.enabled,
                            'timezone': config.schedule.timezone,
                            'calendar': config.schedule.calendar,
                            'enabled_weekdays': config.schedule.enabled_weekdays,
                            'holidays': config.schedule.holidays,
                            'active_windows': [
                                {
                                    'start': w.start,
                                    'end': w.end,
                                    'timezone': w.timezone
                                } for w in config.schedule.active_windows
                            ],
                            'wrap_midnight': config.schedule.wrap_midnight
                        },
                        'market': {
                            'enabled': config.market.enabled,
                            'window_secs': config.market.window_secs,
                            'min_trades_per_min': config.market.min_trades_per_min,
                            'min_quote_updates_per_sec': config.market.min_quote_updates_per_sec,
                            'max_spread_bps': config.market.max_spread_bps,
                            'min_volatility_bps': config.market.min_volatility_bps,
                            'min_volume_usd': config.market.min_volume_usd,
                            'use_median': config.market.use_median,
                            'winsorize_percentile': config.market.winsorize_percentile
                        }
                    }
                },
                'features': {
                    'strategy': {
                        'dynamic_mode_enabled': config.features.dynamic_mode_enabled,
                        'dry_run': config.features.dry_run
                    }
                }
            }
            
            print("配置转换成功")
            print(f"结果类型: {type(result)}")
            print(f"结果内容: {result}")
            
            # 测试访问配置
            strategy_config = result.get('strategy', {})
            print(f"策略配置: {strategy_config}")
            
        except Exception as e:
            print(f"配置转换失败: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_strategy_config()
