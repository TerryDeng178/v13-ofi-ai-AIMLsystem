#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OFI统一配置测试脚本
"""

import sys
import os
import time
import random
sys.path.insert(0, '.')

from src.utils.config_loader import ConfigLoader
from src.real_ofi_calculator import RealOFICalculator

def test_ofi_unified_config():
    """测试OFI在统一配置下的运行"""
    print("=== OFI统一配置测试 ===")
    
    try:
        # 1. 创建配置加载器
        print("1. 创建配置加载器...")
        config_loader = ConfigLoader()
        print("   [OK] 配置加载器创建成功")
        
        # 2. 从统一配置创建OFI计算器
        print("2. 从统一配置创建OFI计算器...")
        ofi_calc = RealOFICalculator("ETHUSDT", config_loader=config_loader)
        print(f"   配置参数: levels={ofi_calc.K}, z_window={ofi_calc.z_window}")
        print(f"   权重: {ofi_calc.w}")
        print("   [OK] OFI计算器创建成功")
        
        # 3. 测试OFI计算
        print("3. 测试OFI计算...")
        
        # 模拟订单簿数据
        bids = [[50000.0, 1.5], [49999.0, 2.0], [49998.0, 1.8], [49997.0, 1.2], [49996.0, 0.8]]
        asks = [[50001.0, 1.2], [50002.0, 1.8], [50003.0, 2.1], [50004.0, 1.5], [50005.0, 1.0]]
        
        # 第一次计算 (建立基线)
        result1 = ofi_calc.update_with_snapshot(bids, asks, event_time_ms=int(time.time() * 1000))
        print(f"   第一次计算: OFI={result1['ofi']:.6f}, Z-score={result1['z_ofi']}")
        
        # 第二次计算 (模拟价格变化)
        bids2 = [[50010.0, 1.5], [50009.0, 2.0], [50008.0, 1.8], [50007.0, 1.2], [50006.0, 0.8]]
        asks2 = [[50011.0, 1.2], [50012.0, 1.8], [50013.0, 2.1], [50014.0, 1.5], [50015.0, 1.0]]
        
        result2 = ofi_calc.update_with_snapshot(bids2, asks2, event_time_ms=int((time.time() + 1) * 1000))
        print(f"   第二次计算: OFI={result2['ofi']:.6f}, Z-score={result2['z_ofi']}")
        
        # 4. 测试多次计算 (建立Z-score基线)
        print("4. 测试多次计算建立Z-score基线...")
        for i in range(10):
            # 生成随机订单簿数据
            base_price = 50000 + random.uniform(-100, 100)
            bids_random = [[base_price + j, random.uniform(0.5, 3.0)] for j in range(5)]
            asks_random = [[base_price + 5 + j, random.uniform(0.5, 3.0)] for j in range(5)]
            
            result = ofi_calc.update_with_snapshot(bids_random, asks_random, event_time_ms=int((time.time() + i) * 1000))
            if result['z_ofi'] is not None:
                print(f"   第{i+1}次: OFI={result['ofi']:.3f}, Z-score={result['z_ofi']:.3f}")
            else:
                print(f"   第{i+1}次: OFI={result['ofi']:.3f}, Z-score=warmup")
        
        # 5. 测试统计信息
        print("5. 测试统计信息...")
        print(f"   坏数据点数: {ofi_calc.bad_points}")
        print(f"   历史数据长度: {len(ofi_calc.ofi_hist)}")
        print(f"   当前EMA: {ofi_calc.ema_ofi}")
        
        # 6. 测试配置参数验证
        print("6. 测试配置参数验证...")
        print(f"   档位数: {ofi_calc.K}")
        print(f"   权重和: {sum(ofi_calc.w):.6f} (应该接近1.0)")
        print(f"   Z-score窗口: {ofi_calc.z_window}")
        print(f"   EMA系数: {ofi_calc.ema_alpha}")
        
        # 验证权重归一化
        if abs(sum(ofi_calc.w) - 1.0) < 1e-6:
            print("   [OK] 权重归一化正确")
        else:
            print("   [ERROR] 权重归一化错误")
            return False
        
        # 验证配置参数
        if ofi_calc.K > 0 and ofi_calc.z_window > 0:
            print("   [OK] 配置参数正确")
        else:
            print("   [ERROR] 配置参数错误")
            return False
        
        print("\n[SUCCESS] OFI在统一配置下运行正常！")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] OFI测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ofi_performance():
    """测试OFI性能"""
    print("\n=== OFI性能测试 ===")
    
    try:
        config_loader = ConfigLoader()
        ofi_calc = RealOFICalculator("ETHUSDT", config_loader=config_loader)
        
        # 性能测试
        num_tests = 1000
        start_time = time.time()
        
        for i in range(num_tests):
            # 生成随机订单簿数据
            base_price = 50000 + random.uniform(-100, 100)
            bids = [[base_price + j, random.uniform(0.5, 3.0)] for j in range(5)]
            asks = [[base_price + 5 + j, random.uniform(0.5, 3.0)] for j in range(5)]
            
            ofi_calc.update_with_snapshot(bids, asks, event_time_ms=int((time.time() + i) * 1000))
        
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

def main():
    """主函数"""
    print("OFI统一配置测试")
    print("=" * 50)
    
    success = True
    
    # 基本功能测试
    if not test_ofi_unified_config():
        success = False
    
    # 性能测试
    if not test_ofi_performance():
        success = False
    
    if success:
        print("\n[SUCCESS] 所有测试通过！OFI在统一配置下运行完全正常。")
    else:
        print("\n[ERROR] 部分测试失败，请检查配置。")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
