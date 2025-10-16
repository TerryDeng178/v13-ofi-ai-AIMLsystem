#!/usr/bin/env python3
"""
测试信号生成
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from src.v12_ofi_expert_model import V12OFIExpertModel
from src.v12_ensemble_ai_model_final import V12EnsembleAIModel
from src.v12_signal_fusion_system import V12SignalFusionSystem

def test_signal_generation():
    """测试信号生成"""
    print("=" * 60)
    print("V12 信号生成测试")
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
    
    # 创建并训练AI模型
    print("\n2. 创建并训练AI模型...")
    
    # OFI专家模型
    ofi_model = V12OFIExpertModel(model_type="ensemble")
    params = {'model_type': 'ensemble'}
    ofi_model.train_model(df, params)
    print(f"OFI模型训练状态: {ofi_model.is_trained}")
    
    # 集成AI模型
    ensemble_config = {
        'device': 'cuda',
        'input_size': 9,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'ensemble_weights': {
            'ofi_expert': 0.4,
            'lstm': 0.2,
            'transformer': 0.2,
            'cnn': 0.2
        }
    }
    
    ensemble_model = V12EnsembleAIModel(ensemble_config)
    feature_columns = ['ofi', 'cvd', 'real_ofi_z', 'real_cvd_z'] + [f'ofi_{i}' for i in range(1, 6)]
    features = df[feature_columns].values
    ensemble_model.train_deep_learning_models(features)
    print(f"集成模型训练状态: {ensemble_model.is_trained}")
    
    # 信号融合系统
    print("\n3. 创建信号融合系统...")
    fusion_config = {
        'signal_quality_threshold': 0.35,  # 降低阈值
        'ai_confidence_threshold': 0.55,   # 降低阈值
        'signal_strength_threshold': 0.15, # 降低阈值
        'max_daily_trades': 50,
        'high_frequency_mode': True
    }
    
    signal_fusion = V12SignalFusionSystem(fusion_config)
    
    # 测试信号生成
    print("\n4. 测试信号生成...")
    
    signals_generated = 0
    signals_passed = 0
    
    for i in range(100):  # 测试前100个数据点
        try:
            # 获取OFI专家模型预测
            ofi_prediction = ofi_model.predict_signal_quality(df.iloc[i:i+1])
            ofi_signal_quality = float(ofi_prediction.iloc[0]) if hasattr(ofi_prediction, 'iloc') else float(ofi_prediction)
            
            # 获取集成AI模型预测
            current_features = features[i:i+1]
            ensemble_prediction = ensemble_model.predict_ensemble(current_features)
            ai_confidence = float(ensemble_prediction)
            
            # 生成融合信号
            signal_data = {
                'ofi_signal_quality': ofi_signal_quality,
                'ai_confidence': ai_confidence,
                'signal_strength': abs(ofi_signal_quality - 0.5) * 2,
                'timestamp': df.iloc[i]['timestamp'],
                'price': df.iloc[i]['price']
            }
            
            fused_signal = signal_fusion.generate_fused_signal(signal_data)
            
            if fused_signal:
                signals_generated += 1
                if fused_signal['action'] != 'hold':
                    signals_passed += 1
                    print(f"信号 {i}: {fused_signal['action']}, 质量: {ofi_signal_quality:.4f}, 置信度: {ai_confidence:.4f}")
            
        except Exception as e:
            print(f"处理第{i}个数据点时出错: {e}")
            continue
    
    print(f"\n信号生成统计:")
    print(f"总信号生成: {signals_generated}")
    print(f"通过信号: {signals_passed}")
    print(f"通过率: {signals_passed/max(signals_generated, 1)*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("信号生成测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_signal_generation()
