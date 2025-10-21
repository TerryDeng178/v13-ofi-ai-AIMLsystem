#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVD统一配置测试脚本
"""

import sys
import os
import time
import random
sys.path.insert(0, '.')

from src.utils.config_loader import ConfigLoader
from src.real_cvd_calculator import RealCVDCalculator

def test_cvd_unified_config():
    """测试CVD在统一配置下的运行"""
    print("=== CVD统一配置测试 ===")
    
    try:
        # 1. 创建配置加载器
        print("1. 创建配置加载器...")
        config_loader = ConfigLoader()
        print("   [OK] 配置加载器创建成功")
        
        # 2. 从统一配置创建CVD计算器
        print("2. 从统一配置创建CVD计算器...")
        cvd_calc = RealCVDCalculator("ETHUSDT", config_loader=config_loader)
        print(f"   配置参数: z_window={cvd_calc.cfg.z_window}, z_mode={cvd_calc.cfg.z_mode}")
        print(f"   half_life_trades={cvd_calc.cfg.half_life_trades}")
        print("   [OK] CVD计算器创建成功")
        
        # 3. 测试CVD计算
        print("3. 测试CVD计算...")
        
        # 模拟成交数据
        trades = [
            {"price": 50000.0, "qty": 1.5, "is_buy": True, "event_time_ms": int(time.time() * 1000)},
            {"price": 50001.0, "qty": 2.0, "is_buy": False, "event_time_ms": int(time.time() * 1000) + 100},
            {"price": 49999.0, "qty": 1.8, "is_buy": True, "event_time_ms": int(time.time() * 1000) + 200},
            {"price": 50002.0, "qty": 1.2, "is_buy": False, "event_time_ms": int(time.time() * 1000) + 300},
            {"price": 50000.5, "qty": 0.8, "is_buy": True, "event_time_ms": int(time.time() * 1000) + 400},
        ]
        
        # 逐笔处理成交数据
        for i, trade in enumerate(trades):
            result = cvd_calc.update_with_trade(
                price=trade["price"],
                qty=trade["qty"],
                is_buy=trade["is_buy"],
                event_time_ms=trade["event_time_ms"]
            )
            print(f"   第{i+1}笔: 价格={trade['price']}, 数量={trade['qty']}, 方向={'买入' if trade['is_buy'] else '卖出'}")
            print(f"           CVD={result['cvd']:.6f}, Z-score={result['z_cvd']}, EMA={result['ema_cvd']}")
        
        # 4. 测试多次计算 (建立Z-score基线)
        print("4. 测试多次计算建立Z-score基线...")
        for i in range(10):
            # 生成随机成交数据
            price = 50000 + random.uniform(-100, 100)
            qty = random.uniform(0.1, 3.0)
            is_buy = random.choice([True, False])
            event_time = int((time.time() + i) * 1000)
            
            result = cvd_calc.update_with_trade(
                price=price,
                qty=qty,
                is_buy=is_buy,
                event_time_ms=event_time
            )
            
            if result['z_cvd'] is not None:
                print(f"   第{i+1}次: CVD={result['cvd']:.3f}, Z-score={result['z_cvd']:.3f}")
            else:
                print(f"   第{i+1}次: CVD={result['cvd']:.3f}, Z-score=warmup")
        
        # 5. 测试统计信息
        print("5. 测试统计信息...")
        print(f"   坏数据点数: {cvd_calc.bad_points}")
        print(f"   历史数据长度: {len(cvd_calc._hist)}")
        print(f"   当前EMA: {cvd_calc.ema_cvd}")
        print(f"   最后价格: {cvd_calc._last_price}")
        
        # 6. 测试配置参数验证
        print("6. 测试配置参数验证...")
        print(f"   Z-score窗口: {cvd_calc.cfg.z_window}")
        print(f"   Z-score模式: {cvd_calc.cfg.z_mode}")
        print(f"   半衰期: {cvd_calc.cfg.half_life_trades}")
        print(f"   EMA系数: {cvd_calc.cfg.ema_alpha}")
        print(f"   使用Tick Rule: {cvd_calc.cfg.use_tick_rule}")
        
        # 验证配置参数
        if (cvd_calc.cfg.z_window > 0 and 
            cvd_calc.cfg.half_life_trades > 0 and
            cvd_calc.cfg.ema_alpha > 0):
            print("   [OK] 配置参数正确")
        else:
            print("   [ERROR] 配置参数错误")
            return False
        
        print("\n[SUCCESS] CVD在统一配置下运行正常！")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] CVD测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cvd_performance():
    """测试CVD性能"""
    print("\n=== CVD性能测试 ===")
    
    try:
        config_loader = ConfigLoader()
        cvd_calc = RealCVDCalculator("ETHUSDT", config_loader=config_loader)
        
        # 性能测试
        num_tests = 1000
        start_time = time.time()
        
        for i in range(num_tests):
            # 生成随机成交数据
            price = 50000 + random.uniform(-100, 100)
            qty = random.uniform(0.1, 3.0)
            is_buy = random.choice([True, False])
            event_time = int((time.time() + i) * 1000)
            
            cvd_calc.update_with_trade(
                price=price,
                qty=qty,
                is_buy=is_buy,
                event_time_ms=event_time
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_tests * 1000  # 转换为毫秒
        
        print(f"性能测试结果:")
        print(f"  测试次数: {num_tests}")
        print(f"  总时间: {total_time:.3f}秒")
        print(f"  平均时间: {avg_time:.3f}ms")
        print(f"  每秒处理: {num_tests / total_time:.0f}次")
        
        if avg_time < 1.0:  # 小于1ms
            print("  [OK] 性能优秀")
        else:
            print("  [WARN] 性能需要优化")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 性能测试失败: {e}")
        return False

def test_cvd_config_override():
    """测试CVD配置覆盖"""
    print("\n=== CVD配置覆盖测试 ===")
    
    try:
        # 设置环境变量
        os.environ['V13__COMPONENTS__CVD__Z_WINDOW'] = '600'
        os.environ['V13__COMPONENTS__CVD__Z_MODE'] = 'delta'
        os.environ['V13__COMPONENTS__CVD__HALF_LIFE_TRADES'] = '500'
        
        # 重新加载配置
        config_loader = ConfigLoader()
        config_loader.load(reload=True)
        
        # 创建CVD计算器
        cvd_calc = RealCVDCalculator("ETHUSDT", config_loader=config_loader)
        
        print(f"环境变量覆盖后的配置:")
        print(f"  Z-score窗口: {cvd_calc.cfg.z_window} (应该是600)")
        print(f"  Z-score模式: {cvd_calc.cfg.z_mode} (应该是delta)")
        print(f"  半衰期: {cvd_calc.cfg.half_life_trades} (应该是500)")
        
        # 验证覆盖是否生效
        if (cvd_calc.cfg.z_window == 600 and 
            cvd_calc.cfg.z_mode == 'delta' and
            cvd_calc.cfg.half_life_trades == 500):
            print("  [OK] 环境变量覆盖成功")
        else:
            print("  [ERROR] 环境变量覆盖失败")
            return False
        
        # 清理环境变量
        del os.environ['V13__COMPONENTS__CVD__Z_WINDOW']
        del os.environ['V13__COMPONENTS__CVD__Z_MODE']
        del os.environ['V13__COMPONENTS__CVD__HALF_LIFE_TRADES']
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 配置覆盖测试失败: {e}")
        return False

def main():
    """主函数"""
    print("CVD统一配置测试")
    print("=" * 50)
    
    success = True
    
    # 基本功能测试
    if not test_cvd_unified_config():
        success = False
    
    # 性能测试
    if not test_cvd_performance():
        success = False
    
    # 配置覆盖测试
    if not test_cvd_config_override():
        success = False
    
    if success:
        print("\n[SUCCESS] 所有测试通过！CVD在统一配置下运行完全正常。")
    else:
        print("\n[ERROR] 部分测试失败，请检查配置。")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
