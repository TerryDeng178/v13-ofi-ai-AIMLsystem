#!/usr/bin/env python3
"""
简单信号生成测试
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime

def test_simple_signal_logic():
    """测试简单信号逻辑"""
    print("=" * 60)
    print("V12 简单信号生成测试")
    print("=" * 60)
    
    # 创建模拟数据
    print("\n1. 创建模拟数据...")
    n_samples = 1000
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'price': 3000 + np.random.randn(n_samples).cumsum() * 0.1,
        'ofi': np.random.randn(n_samples) * 0.5,
        'cvd': np.random.randn(n_samples).cumsum() * 0.1,
        'real_ofi_z': np.random.randn(n_samples),
        'real_cvd_z': np.random.randn(n_samples),
    }
    
    for i in range(1, 6):
        data[f'ofi_{i}'] = np.random.randn(n_samples) * 0.3
    
    df = pd.DataFrame(data)
    print(f"数据形状: {df.shape}")
    
    # 简单信号生成逻辑
    print("\n2. 测试简单信号生成逻辑...")
    
    signals = []
    
    # 设置阈值
    signal_quality_threshold = 0.35  # 降低阈值
    ai_confidence_threshold = 0.55   # 降低阈值
    signal_strength_threshold = 0.15 # 降低阈值
    
    for i in range(len(df)):
        try:
            # 模拟OFI信号质量 (0-1范围)
            ofi_signal_quality = abs(df.iloc[i]['real_ofi_z']) / 3.0  # 标准化到0-1
            ofi_signal_quality = min(ofi_signal_quality, 1.0)
            
            # 模拟AI置信度 (0-1范围)
            ai_confidence = abs(df.iloc[i]['real_cvd_z']) / 3.0  # 标准化到0-1
            ai_confidence = min(ai_confidence, 1.0)
            
            # 计算信号强度
            signal_strength = abs(ofi_signal_quality - 0.5) * 2  # 转换为0-1范围
            
            # 生成信号
            signal = None
            
            # 检查信号质量阈值
            if ofi_signal_quality >= signal_quality_threshold:
                # 检查AI置信度阈值
                if ai_confidence >= ai_confidence_threshold:
                    # 检查信号强度阈值
                    if signal_strength >= signal_strength_threshold:
                        # 确定交易方向
                        if df.iloc[i]['real_ofi_z'] > 0:
                            action = 'buy'
                        else:
                            action = 'sell'
                        
                        signal = {
                            'timestamp': df.iloc[i]['timestamp'],
                            'action': action,
                            'price': df.iloc[i]['price'],
                            'signal_quality': ofi_signal_quality,
                            'ai_confidence': ai_confidence,
                            'signal_strength': signal_strength
                        }
            
            if signal:
                signals.append(signal)
                
        except Exception as e:
            continue
    
    print(f"\n信号生成统计:")
    print(f"总数据点: {len(df)}")
    print(f"生成信号数: {len(signals)}")
    print(f"信号生成率: {len(signals)/len(df)*100:.2f}%")
    
    if signals:
        signals_df = pd.DataFrame(signals)
        buy_signals = len(signals_df[signals_df['action'] == 'buy'])
        sell_signals = len(signals_df[signals_df['action'] == 'sell'])
        
        print(f"买入信号: {buy_signals}")
        print(f"卖出信号: {sell_signals}")
        print(f"平均信号质量: {signals_df['signal_quality'].mean():.4f}")
        print(f"平均AI置信度: {signals_df['ai_confidence'].mean():.4f}")
        print(f"平均信号强度: {signals_df['signal_strength'].mean():.4f}")
        
        # 显示前几个信号
        print(f"\n前5个信号:")
        for i, signal in enumerate(signals[:5]):
            print(f"信号{i+1}: {signal['action']} @ {signal['price']:.2f}, "
                  f"质量: {signal['signal_quality']:.4f}, "
                  f"置信度: {signal['ai_confidence']:.4f}")
    else:
        print("没有生成任何信号，阈值可能设置过高")
    
    print("\n" + "=" * 60)
    print("简单信号生成测试完成")
    print("=" * 60)
    
    return signals

if __name__ == "__main__":
    test_simple_signal_logic()
