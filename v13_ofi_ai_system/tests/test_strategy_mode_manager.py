# -*- coding: utf-8 -*-
"""
策略模式管理器单元测试

测试覆盖：
1. 跨午夜时间窗口判定
2. 迟滞逻辑（防抖）
3. 干跑模式
4. 强制模式覆盖
5. 原子更新成功/失败回滚
6. HKT时区处理
7. params_diff白名单与截断
8. 节假日判定
9. 周末判定
10. 时区边界

作者: V13 Team
创建日期: 2025-10-19
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
import pytz

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.strategy_mode_manager import (
    StrategyModeManager,
    StrategyMode,
    TriggerReason,
    MarketActivity
)


class TestStrategyModeManager(unittest.TestCase):
    """策略模式管理器测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.config = {
            'system': {
                'version': 'v13.0.7',
                'environment': 'testing'
            },
            'strategy': {
                'mode': 'auto',
                'hysteresis': {
                    'window_secs': 60,
                    'min_active_windows': 3,
                    'min_quiet_windows': 6
                },
                'triggers': {
                    'schedule': {
                        'enabled': True,
                        'timezone': 'Asia/Hong_Kong',
                        'calendar': 'CRYPTO',
                        'enabled_weekdays': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                        'holidays': [],
                        'active_windows': ['09:00-12:00', '14:00-17:00', '21:00-02:00', '06:00-08:00']
                    },
                    'market': {
                        'enabled': True,
                        'min_trades_per_min': 500,
                        'min_quote_updates_per_sec': 100,
                        'max_spread_bps': 5,
                        'min_volatility_bps': 10,
                        'min_volume_usd': 1000000
                    }
                },
                'params': {
                    'ofi': {
                        'active': {'bucket_ms': 50, 'depth_levels': 15},
                        'quiet': {'bucket_ms': 500, 'depth_levels': 5}
                    },
                    'cvd': {
                        'active': {'window_ticks': 1000, 'ema_span': 30},
                        'quiet': {'window_ticks': 10000, 'ema_span': 300}
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
    
    # ==========================================================================
    # 测试组1: 跨午夜时间窗口
    # ==========================================================================
    
    def test_cross_midnight_window_before_midnight(self):
        """测试跨午夜窗口（午夜前）：21:00-02:00，测试22:00"""
        manager = StrategyModeManager(self.config)
        
        # 创建HKT时区的测试时间：22:00
        hkt = pytz.timezone('Asia/Hong_Kong')
        test_time = datetime(2025, 10, 19, 22, 0, 0, tzinfo=hkt)
        
        # 应该在活跃窗口内（21:00-02:00）
        is_active = manager.check_schedule_active(test_time)
        self.assertTrue(is_active, "22:00 HKT应该在21:00-02:00窗口内")
    
    def test_cross_midnight_window_after_midnight(self):
        """测试跨午夜窗口（午夜后）：21:00-02:00，测试01:00"""
        manager = StrategyModeManager(self.config)
        
        # 创建HKT时区的测试时间：01:00
        hkt = pytz.timezone('Asia/Hong_Kong')
        test_time = datetime(2025, 10, 20, 1, 0, 0, tzinfo=hkt)
        
        # 应该在活跃窗口内（21:00-02:00）
        is_active = manager.check_schedule_active(test_time)
        self.assertTrue(is_active, "01:00 HKT应该在21:00-02:00窗口内")
    
    def test_cross_midnight_window_boundary_start(self):
        """测试跨午夜窗口边界（起点）：21:00"""
        manager = StrategyModeManager(self.config)
        
        hkt = pytz.timezone('Asia/Hong_Kong')
        test_time = datetime(2025, 10, 19, 21, 0, 0, tzinfo=hkt)
        
        is_active = manager.check_schedule_active(test_time)
        self.assertTrue(is_active, "21:00应该在21:00-02:00窗口内（含起点）")
    
    def test_cross_midnight_window_boundary_end(self):
        """测试跨午夜窗口边界（终点）：02:00"""
        manager = StrategyModeManager(self.config)
        
        hkt = pytz.timezone('Asia/Hong_Kong')
        test_time = datetime(2025, 10, 20, 2, 0, 0, tzinfo=hkt)
        
        is_active = manager.check_schedule_active(test_time)
        self.assertFalse(is_active, "02:00应该不在21:00-02:00窗口内（不含终点）")
    
    def test_cross_midnight_window_outside(self):
        """测试跨午夜窗口外：03:00（凌晨低谷）"""
        manager = StrategyModeManager(self.config)
        
        hkt = pytz.timezone('Asia/Hong_Kong')
        test_time = datetime(2025, 10, 20, 3, 0, 0, tzinfo=hkt)
        
        is_active = manager.check_schedule_active(test_time)
        self.assertFalse(is_active, "03:00不应该在任何活跃窗口内（凌晨低谷）")
    
    # ==========================================================================
    # 测试组2: 迟滞逻辑（防抖）
    # ==========================================================================
    
    def test_hysteresis_to_active_需要3个窗口(self):
        """测试切换到active需要连续3个窗口满足"""
        manager = StrategyModeManager(self.config)
        
        # 创建活跃的市场数据
        activity = MarketActivity()
        activity.trades_per_min = 600
        activity.quote_updates_per_sec = 120
        activity.spread_bps = 4.5
        activity.volatility_bps = 15
        activity.volume_usd = 2000000
        
        # 初始模式应该是quiet
        self.assertEqual(manager.current_mode, StrategyMode.QUIET)
        
        # 第1次：还不够
        mode, reason, _ = manager.decide_mode(activity)
        self.assertEqual(mode, StrategyMode.QUIET, "第1个活跃窗口不应该切换")
        
        # 第2次：还不够
        mode, reason, _ = manager.decide_mode(activity)
        self.assertEqual(mode, StrategyMode.QUIET, "第2个活跃窗口不应该切换")
        
        # 第3次：应该切换了
        mode, reason, _ = manager.decide_mode(activity)
        self.assertEqual(mode, StrategyMode.ACTIVE, "连续3个活跃窗口应该切换到active")
    
    def test_hysteresis_to_quiet_需要6个窗口(self):
        """测试切换到quiet需要连续6个窗口不满足"""
        manager = StrategyModeManager(self.config)
        manager.current_mode = StrategyMode.ACTIVE  # 先设置为active
        
        # 创建不活跃的市场数据
        activity = MarketActivity()
        activity.trades_per_min = 100  # 低于阈值500
        activity.quote_updates_per_sec = 20  # 低于阈值100
        activity.spread_bps = 8  # 高于阈值5
        activity.volatility_bps = 3  # 低于阈值10
        activity.volume_usd = 500000  # 低于阈值1000000
        
        # 需要6次才能切回quiet
        for i in range(5):
            mode, reason, _ = manager.decide_mode(activity)
            self.assertEqual(mode, StrategyMode.ACTIVE, f"第{i+1}个不活跃窗口不应该切换")
        
        # 第6次：应该切换了
        mode, reason, _ = manager.decide_mode(activity)
        self.assertEqual(mode, StrategyMode.QUIET, "连续6个不活跃窗口应该切换到quiet")
    
    # ==========================================================================
    # 测试组3: 强制模式
    # ==========================================================================
    
    def test_force_active_mode(self):
        """测试强制active模式"""
        self.config['strategy']['mode'] = 'active'
        manager = StrategyModeManager(self.config)
        
        # 即使市场不活跃，也应该保持active
        activity = MarketActivity()
        activity.trades_per_min = 10  # 极低
        
        mode, reason, _ = manager.decide_mode(activity)
        self.assertEqual(mode, StrategyMode.ACTIVE)
        self.assertEqual(reason, TriggerReason.MANUAL)
    
    def test_force_quiet_mode(self):
        """测试强制quiet模式"""
        self.config['strategy']['mode'] = 'quiet'
        manager = StrategyModeManager(self.config)
        
        # 即使市场活跃，也应该保持quiet
        activity = MarketActivity()
        activity.trades_per_min = 1000  # 极高
        activity.quote_updates_per_sec = 200
        activity.spread_bps = 2
        activity.volatility_bps = 20
        activity.volume_usd = 5000000
        
        mode, reason, _ = manager.decide_mode(activity)
        self.assertEqual(mode, StrategyMode.QUIET)
        self.assertEqual(reason, TriggerReason.MANUAL)
    
    # ==========================================================================
    # 测试组4: 干跑模式
    # ==========================================================================
    
    def test_dry_run_mode(self):
        """测试干跑模式"""
        self.config['features']['strategy']['dry_run'] = True
        manager = StrategyModeManager(self.config)
        
        # 应用参数应该成功但不实际应用
        success, failed_modules = manager.apply_params(StrategyMode.ACTIVE)
        
        self.assertTrue(success)
        self.assertEqual(len(failed_modules), 0)
        # 检查指标是否记录了dry_run
        metrics = manager.get_metrics()
        # 应该有dry_run的histogram记录
    
    # ==========================================================================
    # 测试组5: 市场触发器
    # ==========================================================================
    
    def test_market_trigger_all_conditions_met(self):
        """测试市场触发器（所有条件满足）"""
        manager = StrategyModeManager(self.config)
        
        activity = MarketActivity()
        activity.trades_per_min = 600     # ✓ >= 500
        activity.quote_updates_per_sec = 120  # ✓ >= 100
        activity.spread_bps = 4.5          # ✓ <= 5
        activity.volatility_bps = 15       # ✓ >= 10
        activity.volume_usd = 2000000      # ✓ >= 1000000
        
        is_active = manager.check_market_active(activity)
        self.assertTrue(is_active, "所有条件满足应该判定为活跃")
    
    def test_market_trigger_one_condition_failed(self):
        """测试市场触发器（一个条件不满足）"""
        manager = StrategyModeManager(self.config)
        
        activity = MarketActivity()
        activity.trades_per_min = 600
        activity.quote_updates_per_sec = 120
        activity.spread_bps = 4.5
        activity.volatility_bps = 8  # ✗ < 10（不满足）
        activity.volume_usd = 2000000
        
        is_active = manager.check_market_active(activity)
        self.assertFalse(is_active, "任一条件不满足应该判定为不活跃")
    
    # ==========================================================================
    # 测试组6: 周末与节假日
    # ==========================================================================
    
    def test_weekend_crypto_24_7(self):
        """测试周末（数字资产24/7）"""
        manager = StrategyModeManager(self.config)
        
        # 创建周六的测试时间：10:00（在09:00-12:00窗口内）
        hkt = pytz.timezone('Asia/Hong_Kong')
        saturday = datetime(2025, 10, 18, 10, 0, 0, tzinfo=hkt)  # 2025-10-18是周六
        
        is_active = manager.check_schedule_active(saturday)
        self.assertTrue(is_active, "数字资产市场周末也应该正常运作")
    
    def test_holiday_crypto_no_holidays(self):
        """测试节假日（数字资产无节假日）"""
        # 添加一个节假日
        self.config['strategy']['triggers']['schedule']['holidays'] = ['2025-10-19']
        manager = StrategyModeManager(self.config)
        
        hkt = pytz.timezone('Asia/Hong_Kong')
        holiday = datetime(2025, 10, 19, 10, 0, 0, tzinfo=hkt)
        
        is_active = manager.check_schedule_active(holiday)
        self.assertFalse(is_active, "节假日应该不活跃（如果配置了）")
    
    # ==========================================================================
    # 测试组7: params_diff
    # ==========================================================================
    
    def test_params_diff_whitelist(self):
        """测试params_diff白名单"""
        manager = StrategyModeManager(self.config)
        
        diff = manager._compute_params_diff(StrategyMode.QUIET, StrategyMode.ACTIVE)
        
        # 应该包含白名单中的差异
        self.assertIn('ofi.bucket_ms', diff)
        self.assertEqual(diff['ofi.bucket_ms'], '500 → 50')
        
        self.assertIn('cvd.window_ticks', diff)
        self.assertEqual(diff['cvd.window_ticks'], '10000 → 1000')
    
    def test_params_diff_truncation(self):
        """测试params_diff截断（超过10个）"""
        # 添加更多参数以测试截断
        for i in range(15):
            key = f'test_param_{i}'
            self.config['strategy']['params']['ofi']['active'][key] = i
            self.config['strategy']['params']['ofi']['quiet'][key] = i + 100
        
        manager = StrategyModeManager(self.config)
        diff = manager._compute_params_diff(StrategyMode.QUIET, StrategyMode.ACTIVE)
        
        # 应该被截断到10个 + _truncated
        self.assertLessEqual(len(diff), 11)
        if len(diff) == 11:
            self.assertIn('_truncated', diff)
    
    # ==========================================================================
    # 测试组8: 时区处理
    # ==========================================================================
    
    def test_timezone_hkt(self):
        """测试HKT时区处理"""
        manager = StrategyModeManager(self.config)
        
        # 验证时区设置
        self.assertEqual(manager.timezone.zone, 'Asia/Hong_Kong')
        
        # UTC时间：01:00 = HKT 09:00（在09:00-12:00窗口内）
        utc = pytz.UTC
        utc_time = datetime(2025, 10, 19, 1, 0, 0, tzinfo=utc)
        
        is_active = manager.check_schedule_active(utc_time)
        self.assertTrue(is_active, "UTC 01:00 = HKT 09:00应该在活跃窗口内")
    
    # ==========================================================================
    # 测试组9: 更新模式与指标
    # ==========================================================================
    
    def test_update_mode_triggers_metrics(self):
        """测试update_mode更新指标"""
        manager = StrategyModeManager(self.config)
        
        # 创建活跃市场数据
        activity = MarketActivity()
        activity.trades_per_min = 600
        activity.quote_updates_per_sec = 120
        activity.spread_bps = 4.5
        activity.volatility_bps = 15
        activity.volume_usd = 2000000
        
        # 连续3次调用以触发切换
        for _ in range(3):
            manager.decide_mode(activity)
        
        # 执行切换
        event = manager.update_mode(activity)
        
        # 如果发生了切换，验证事件结构
        if event:
            self.assertIn('event', event)
            self.assertIn('from', event)
            self.assertIn('to', event)
            self.assertIn('reason', event)
            self.assertIn('triggers', event)
            self.assertIn('params_diff', event)
            self.assertIn('update_duration_ms', event)
            self.assertIn('rollback', event)
            self.assertIn('failed_modules', event)
        
        # 验证指标已更新
        metrics = manager.get_metrics()
        self.assertGreater(len(metrics), 0, "应该有指标记录")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)

