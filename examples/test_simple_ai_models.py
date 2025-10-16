#!/usr/bin/env python3
"""
简单测试AI模型
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pandas as pd

def test_simple_ai_models():
    """简单测试AI模型"""
    print("=" * 60)
    print("V12 AI模型简单测试")
    print("=" * 60)
    
    # 创建模拟的OFI数据
    print("\n1. 创建模拟OFI数据...")
    n_samples = 1000
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'price': 3000 + np.random.randn(n_samples).cumsum() * 0.1,
        'ask1': 3000.1 + np.random.randn(n_samples) * 0.01,
        'bid1': 2999.9 + np.random.randn(n_samples) * 0.01,
        'ask_volume1': np.random.randint(1, 100, n_samples),
        'bid_volume1': np.random.randint(1, 100, n_samples),
        'ask2': 3000.2 + np.random.randn(n_samples) * 0.01,
        'bid2': 2999.8 + np.random.randn(n_samples) * 0.01,
        'ask_volume2': np.random.randint(1, 50, n_samples),
        'bid_volume2': np.random.randint(1, 50, n_samples),
        'ask3': 3000.3 + np.random.randn(n_samples) * 0.01,
        'bid3': 2999.7 + np.random.randn(n_samples) * 0.01,
        'ask_volume3': np.random.randint(1, 30, n_samples),
        'bid_volume3': np.random.randint(1, 30, n_samples),
        'ask4': 3000.4 + np.random.randn(n_samples) * 0.01,
        'bid4': 2999.6 + np.random.randn(n_samples) * 0.01,
        'ask_volume4': np.random.randint(1, 20, n_samples),
        'bid_volume4': np.random.randint(1, 20, n_samples),
        'ask5': 3000.5 + np.random.randn(n_samples) * 0.01,
        'bid5': 2999.5 + np.random.randn(n_samples) * 0.01,
        'ask_volume5': np.random.randint(1, 10, n_samples),
        'bid_volume5': np.random.randint(1, 10, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 计算OFI和CVD
    print("2. 计算OFI和CVD...")
    for i in range(1, 6):
        df[f'ofi_{i}'] = (df[f'bid_volume{i}'] - df[f'ask_volume{i}']) / (df[f'bid_volume{i}'] + df[f'ask_volume{i}'])
    
    df['ofi'] = (df['ofi_1'] * 5 + df['ofi_2'] * 4 + df['ofi_3'] * 3 + df['ofi_4'] * 2 + df['ofi_5'] * 1) / 15
    df['cvd'] = df['ofi'].cumsum()
    
    # 计算Z-score
    df['real_ofi_z'] = (df['ofi'] - df['ofi'].rolling(20).mean()) / df['ofi'].rolling(20).std()
    df['real_cvd_z'] = (df['cvd'] - df['cvd'].rolling(20).mean()) / df['cvd'].rolling(20).std()
    
    # 添加目标变量
    df['target'] = np.where(df['real_ofi_z'] > 1, 1, np.where(df['real_ofi_z'] < -1, -1, 0))
    
    print(f"数据形状: {df.shape}")
    print(f"OFI范围: {df['ofi'].min():.4f} - {df['ofi'].max():.4f}")
    print(f"CVD范围: {df['cvd'].min():.4f} - {df['cvd'].max():.4f}")
    
    # 测试OFI专家模型
    print("\n3. 测试OFI专家模型...")
    try:
        from src.v12_ofi_expert_model import V12OFIExpertModel
        
        ofi_model = V12OFIExpertModel(model_type="ensemble")
        
        # 训练模型
        print("训练OFI专家模型...")
        params = {'model_type': 'ensemble'}
        ofi_model.train_model(df, params)
        
        print(f"OFI模型训练状态: {ofi_model.is_trained}")
        
        # 测试预测
        test_df = df.tail(10)
        prediction = ofi_model.predict_signal_quality(test_df)
        print(f"OFI模型预测结果: {prediction}")
        
    except Exception as e:
        print(f"OFI专家模型测试失败: {e}")
    
    # 测试集成AI模型
    print("\n4. 测试集成AI模型...")
    try:
        from src.v12_ensemble_ai_model_final import V12EnsembleAIModel
        
        ensemble_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'input_size': 31,
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
        
        # 准备特征数据
        feature_columns = ['ofi', 'cvd', 'real_ofi_z', 'real_cvd_z'] + [f'ofi_{i}' for i in range(1, 6)]
        features = df[feature_columns].values
        targets = df['target'].values
        
        # 训练模型
        print("训练集成AI模型...")
        ensemble_model.train_deep_learning_models(features)
        
        print(f"集成模型训练状态: {ensemble_model.is_trained}")
        
        # 测试预测
        test_features = features[-10:]
        prediction = ensemble_model.predict_ensemble(test_features)
        print(f"集成模型预测结果: {prediction}")
        
    except Exception as e:
        print(f"集成AI模型测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_simple_ai_models()
