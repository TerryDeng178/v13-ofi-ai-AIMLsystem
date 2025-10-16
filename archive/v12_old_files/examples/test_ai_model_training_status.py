#!/usr/bin/env python3
"""
测试AI模型训练状态
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from src.v12_ofi_expert_model import V12OFIExpertModel
from src.v12_ensemble_ai_model_final import V12EnsembleAIModel

def test_model_training_status():
    """测试模型训练状态"""
    print("=" * 60)
    print("V12 AI模型训练状态测试")
    print("=" * 60)
    
    # 测试OFI专家模型
    print("\n1. 测试OFI专家模型...")
    ofi_model = V12OFIExpertModel(
        model_type="ensemble"
    )
    
    # 生成测试数据
    test_data = np.random.randn(100, 30)
    test_targets = np.random.randint(0, 2, 100)
    
    # 训练模型
    print("训练OFI专家模型...")
    # 创建DataFrame格式的训练数据
    import pandas as pd
    df = pd.DataFrame(test_data)
    df['target'] = test_targets
    params = {'model_type': 'ensemble'}
    ofi_model.train_model(df, params)
    
    # 测试预测
    prediction = ofi_model.predict_signal_quality(test_data[:5])
    print(f"OFI模型预测结果: {prediction}")
    print(f"OFI模型是否训练: {ofi_model.is_trained}")
    
    # 测试集成AI模型
    print("\n2. 测试集成AI模型...")
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
    
    # 生成测试数据
    test_features = np.random.randn(100, 31)
    test_targets = np.random.randint(0, 2, 100)
    
    # 训练模型
    print("训练集成AI模型...")
    ensemble_model.train_models(test_features, test_targets)
    
    # 测试预测
    prediction = ensemble_model.predict_ensemble(test_features[:5])
    print(f"集成模型预测结果: {prediction}")
    print(f"集成模型是否训练: {ensemble_model.is_trained}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_model_training_status()
