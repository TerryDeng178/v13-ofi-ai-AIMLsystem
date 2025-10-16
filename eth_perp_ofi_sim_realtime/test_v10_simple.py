#!/usr/bin/env python3
"""
V10.0 简化测试脚本
测试3级加权OFI和深度学习功能
"""

import sys
import os
import numpy as np
import pandas as pd
import time
from datetime import datetime

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_v10_enhanced_ofi():
    """测试V10增强OFI计算器"""
    print("="*60)
    print("测试V10.0增强OFI计算器")
    print("="*60)
    
    try:
        from ofi_v10_enhanced import V10EnhancedOFI
        
        # 创建OFI计算器
        ofi_calc = V10EnhancedOFI(
            micro_window_ms=100,
            z_window_seconds=900,
            levels=3,
            weights=[0.5, 0.3, 0.2]
        )
        
        print("OFI计算器创建成功")
        print(f"微窗口: {ofi_calc.w}ms")
        print(f"Z窗口: {ofi_calc.zn}个桶")
        print(f"层级数: {ofi_calc.levels}")
        print(f"权重: {ofi_calc.weights}")
        
        # 模拟数据
        print("\n模拟市场数据...")
        for i in range(100):
            t = i * 100  # 100ms间隔
            
            # 模拟最优买卖价
            bid = 2500.0 + np.random.normal(0, 0.1)
            ask = bid + 0.2 + np.random.normal(0, 0.05)
            bid_sz = np.random.uniform(10, 50)
            ask_sz = np.random.uniform(10, 50)
            
            ofi_calc.on_best(t, bid, bid_sz, ask, ask_sz)
            
            # 模拟L2更新
            if np.random.random() < 0.3:
                side = 'bid' if np.random.random() < 0.5 else 'ask'
                price = bid if side == 'bid' else ask
                qty = np.random.uniform(1, 20)
                typ = 'l2_add' if np.random.random() < 0.7 else 'l2_cancel'
                
                ofi_calc.on_l2(t, typ, side, price, qty)
            
            # 每10次更新读取一次OFI
            if i % 10 == 0:
                ofi_data = ofi_calc.read()
                if ofi_data:
                    print(f"时间: {t}ms")
                    print(f"  OFI: {ofi_data['ofi']:.2f}")
                    print(f"  OFI_Z: {ofi_data['ofi_z']:.3f}")
                    print(f"  加权OFI: {ofi_data['weighted_ofi']:.2f}")
                    print(f"  加权OFI_Z: {ofi_data['weighted_ofi_z']:.3f}")
                    print(f"  各级OFI: {ofi_data['level_ofis']}")
                    print(f"  各级Z: {ofi_data['level_zs']}")
                    print()
        
        # 测试特征创建
        print("测试特征创建...")
        ofi_data = ofi_calc.read()
        if ofi_data:
            market_data = {
                "bid": 2500.0,
                "ask": 2500.2,
                "bid_sz": 25.0,
                "ask_sz": 30.0,
                "spread": 0.2,
                "mid_price": 2500.1
            }
            
            features = ofi_calc.create_features(ofi_data, market_data)
            print(f"特征维度: {len(features)}")
            print(f"特征值: {features[:10]}...")
            
            # 测试信号预测
            signal_result = ofi_calc.predict_signal(features)
            print(f"信号预测: {signal_result}")
        
        # 获取统计信息
        stats = ofi_calc.get_statistics()
        print(f"统计信息: {stats}")
        
        print("V10增强OFI测试完成!")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_market_simulation():
    """测试市场模拟"""
    print("\n" + "="*60)
    print("测试市场模拟器")
    print("="*60)
    
    try:
        from sim import MarketSimulator
        
        # 创建配置
        config = {
            "sim": {
                "seed": 42,
                "seconds": 10,  # 10秒模拟
                "init_mid": 2500.0,
                "tick_size": 0.1,
                "base_spread_ticks": 2,
                "base_depth": 30.0,
                "depth_jitter": 0.6,
                "levels": 5,
                "regimes": [
                    {"name": "trend_up", "prob": 0.25, "mu": 0.20, "sigma": 1.8, "dur_mean_s": 30},
                    {"name": "trend_down", "prob": 0.25, "mu": -0.20, "sigma": 1.8, "dur_mean_s": 30},
                    {"name": "mean_rev", "prob": 0.40, "mu": 0.00, "sigma": 1.2, "dur_mean_s": 45},
                    {"name": "burst", "prob": 0.10, "mu": 0.00, "sigma": 4.0, "dur_mean_s": 6}
                ],
                "rates": {
                    "limit_add": 120,
                    "limit_cancel": 80,
                    "market_sweep": 35
                },
                "spoof_ratio": 0.08,
                "sweep_levels_mean": 1.4,
                "minute_drift_bps": 0.0
            }
        }
        
        # 创建模拟器
        simulator = MarketSimulator(config)
        print("市场模拟器创建成功")
        
        # 运行模拟
        print("运行市场模拟...")
        events = []
        for evts in simulator.stream(realtime=False, dt_ms=10):
            events.extend(evts)
            if len(events) > 100:  # 限制事件数量
                break
        
        print(f"生成事件数: {len(events)}")
        
        # 分析事件类型
        event_types = {}
        for event in events:
            typ = event.get("type", "unknown")
            event_types[typ] = event_types.get(typ, 0) + 1
        
        print("事件类型分布:")
        for typ, count in event_types.items():
            print(f"  {typ}: {count}")
        
        # 显示一些事件示例
        print("\n事件示例:")
        for i, event in enumerate(events[:5]):
            print(f"  事件{i+1}: {event}")
        
        print("市场模拟测试完成!")
        return True
        
    except Exception as e:
        print(f"市场模拟测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """测试集成功能"""
    print("\n" + "="*60)
    print("测试V10集成功能")
    print("="*60)
    
    try:
        from ofi_v10_enhanced import V10EnhancedOFI
        from sim import MarketSimulator
        
        # 创建配置
        config = {
            "sim": {
                "seed": 42,
                "seconds": 5,
                "init_mid": 2500.0,
                "tick_size": 0.1,
                "base_spread_ticks": 2,
                "base_depth": 30.0,
                "depth_jitter": 0.6,
                "levels": 5,
                "regimes": [
                    {"name": "mean_rev", "prob": 1.0, "mu": 0.00, "sigma": 1.2, "dur_mean_s": 45}
                ],
                "rates": {
                    "limit_add": 60,
                    "limit_cancel": 40,
                    "market_sweep": 20
                },
                "spoof_ratio": 0.05,
                "sweep_levels_mean": 1.2,
                "minute_drift_bps": 0.0
            }
        }
        
        # 创建模拟器和OFI计算器
        simulator = MarketSimulator(config)
        ofi_calc = V10EnhancedOFI(micro_window_ms=100, z_window_seconds=900, levels=3)
        
        print("集成测试开始...")
        
        # 运行集成测试
        signal_count = 0
        for events in simulator.stream(realtime=False, dt_ms=10):
            for event in events:
                if event["type"] == "best":
                    ofi_calc.on_best(
                        event["t"], event["bid"], event["bid_sz"], 
                        event["ask"], event["ask_sz"]
                    )
                elif event["type"] in ["l2_add", "l2_cancel"]:
                    ofi_calc.on_l2(
                        event["t"], event["type"], event["side"], 
                        event["price"], event["qty"]
                    )
                
                # 检查OFI和信号
                ofi_data = ofi_calc.read()
                if ofi_data:
                    market_data = {
                        "bid": event.get("bid", 0.0),
                        "ask": event.get("ask", 0.0),
                        "bid_sz": event.get("bid_sz", 0.0),
                        "ask_sz": event.get("ask_sz", 0.0),
                        "spread": event.get("ask", 0.0) - event.get("bid", 0.0),
                        "mid_price": (event.get("bid", 0.0) + event.get("ask", 0.0)) / 2
                    }
                    
                    features = ofi_calc.create_features(ofi_data, market_data)
                    signal_result = ofi_calc.predict_signal(features)
                    
                    if signal_result["signal_side"] != 0:
                        signal_count += 1
                        print(f"信号生成: 方向={signal_result['signal_side']}, "
                              f"强度={signal_result['signal_strength']:.3f}, "
                              f"置信度={signal_result['confidence']:.3f}")
        
        print(f"集成测试完成! 生成信号数: {signal_count}")
        
        # 获取最终统计
        stats = ofi_calc.get_statistics()
        print(f"OFI统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("V10.0 增强实时市场模拟器测试")
    print("="*60)
    
    # 测试OFI计算器
    ofi_success = test_v10_enhanced_ofi()
    
    # 测试市场模拟器
    sim_success = test_market_simulation()
    
    # 测试集成功能
    integration_success = test_integration()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"OFI计算器测试: {'通过' if ofi_success else '失败'}")
    print(f"市场模拟器测试: {'通过' if sim_success else '失败'}")
    print(f"集成功能测试: {'通过' if integration_success else '失败'}")
    
    if ofi_success and sim_success and integration_success:
        print("\n所有测试通过! V10.0增强实时市场模拟器准备就绪!")
    else:
        print("\n部分测试失败，请检查错误信息")

if __name__ == "__main__":
    main()
