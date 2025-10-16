#!/usr/bin/env python3
"""
V10.0 深度学习集成简化测试脚本
测试深度学习模型的基本功能，不依赖外部库
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_simple_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """创建简化的测试数据"""
    np.random.seed(42)
    
    # 生成基础数据
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    volumes = np.random.randint(100, 1000, n_samples)
    
    df = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=n_samples, freq='1s'),
        'price': prices,
        'volume': volumes,
        'bid1': prices - 0.01,
        'ask1': prices + 0.01,
        'bid1_size': np.random.randint(100, 500, n_samples),
        'ask1_size': np.random.randint(100, 500, n_samples),
        'bid2': prices - 0.02,
        'ask2': prices + 0.02,
        'bid2_size': np.random.randint(50, 300, n_samples),
        'ask2_size': np.random.randint(50, 300, n_samples),
        'bid3': prices - 0.03,
        'ask3': prices + 0.03,
        'bid3_size': np.random.randint(20, 200, n_samples),
        'ask3_size': np.random.randint(20, 200, n_samples),
        'high': prices + np.random.uniform(0, 0.05, n_samples),
        'low': prices - np.random.uniform(0, 0.05, n_samples),
        'ofi_z': np.random.randn(n_samples) * 2,
        'cvd_z': np.random.randn(n_samples) * 2,
        'ret_1s': np.random.randn(n_samples) * 0.001,
        'atr': np.random.uniform(0.01, 0.05, n_samples),
        'vwap': prices + np.random.randn(n_samples) * 0.005,
        'signal_quality': np.random.uniform(0, 1, n_samples)
    })
    
    return df

def test_lstm_model():
    """测试LSTM模型"""
    print("=" * 50)
    print("测试LSTM模型...")
    
    try:
        # 直接导入LSTM模块
        sys.path.append('src/models')
        from lstm_predictor import create_lstm_model
        
        # 创建LSTM模型
        model = create_lstm_model(input_dim=50, hidden_dim=128, num_layers=3)
        print(f"LSTM模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        batch_size, seq_len, input_dim = 32, 60, 50
        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)
        print(f"LSTM前向传播测试: 输入形状 {x.shape} -> 输出形状 {output.shape}")
        
        return True
    except Exception as e:
        print(f"LSTM模型测试失败: {e}")
        return False

def test_cnn_model():
    """测试CNN模型"""
    print("=" * 50)
    print("测试CNN模型...")
    
    try:
        # 直接导入CNN模块
        sys.path.append('src/models')
        from cnn_recognizer import create_cnn_model
        
        # 创建CNN模型
        model = create_cnn_model(input_channels=1, num_classes=10, feature_dim=50, sequence_length=60)
        print(f"CNN模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        batch_size, channels, seq_len = 32, 1, 60
        x = torch.randn(batch_size, channels, seq_len)
        output = model(x)
        print(f"CNN前向传播测试: 输入形状 {x.shape} -> 输出形状 {output.shape}")
        
        return True
    except Exception as e:
        print(f"CNN模型测试失败: {e}")
        return False

def test_transformer_model():
    """测试Transformer模型"""
    print("=" * 50)
    print("测试Transformer模型...")
    
    try:
        # 直接导入Transformer模块
        sys.path.append('src/models')
        from transformer_predictor import create_transformer_model
        
        # 创建Transformer模型
        model = create_transformer_model(input_dim=50, d_model=128, nhead=8, num_layers=6)
        print(f"Transformer模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        batch_size, seq_len, input_dim = 32, 60, 50
        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)
        print(f"Transformer前向传播测试: 输入形状 {x.shape} -> 输出形状 {output.shape}")
        
        return True
    except Exception as e:
        print(f"Transformer模型测试失败: {e}")
        return False

def test_ensemble_model():
    """测试集成学习模型"""
    print("=" * 50)
    print("测试集成学习模型...")
    
    try:
        # 直接导入集成学习模块
        sys.path.append('src/models')
        from ensemble_predictor import create_ensemble_model
        
        # 创建集成学习模型
        model = create_ensemble_model(input_dim=50, sequence_length=60, ensemble_method='weighted_average')
        print(f"集成学习模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        batch_size, seq_len, input_dim = 32, 60, 50
        x = torch.randn(batch_size, seq_len, input_dim)
        prediction, uncertainty = model(x)
        print(f"集成学习前向传播测试: 输入形状 {x.shape} -> 预测形状 {prediction.shape}, 不确定性形状 {uncertainty.shape}")
        
        return True
    except Exception as e:
        print(f"集成学习模型测试失败: {e}")
        return False

def test_simple_feature_engineering():
    """测试简单特征工程"""
    print("=" * 50)
    print("测试简单特征工程...")
    
    try:
        # 创建测试数据
        df = create_simple_test_data(1000)
        print(f"原始数据形状: {df.shape}")
        
        # 创建简单特征
        features_df = df.copy()
        
        # 基础特征
        features_df['price_change'] = features_df['price'].pct_change()
        features_df['price_volatility'] = features_df['price_change'].rolling(20).std()
        features_df['volume_ma_ratio'] = features_df['volume'] / features_df['volume'].rolling(20).mean()
        features_df['spread'] = features_df['ask1'] - features_df['bid1']
        features_df['spread_bps'] = features_df['spread'] / features_df['price'] * 10000
        
        # 深度特征
        features_df['bid_depth'] = features_df['bid1_size'] + features_df['bid2_size'] + features_df['bid3_size']
        features_df['ask_depth'] = features_df['ask1_size'] + features_df['ask2_size'] + features_df['ask3_size']
        features_df['depth_imbalance'] = (features_df['bid_depth'] - features_df['ask_depth']) / (features_df['bid_depth'] + features_df['ask_depth'])
        
        # 订单流特征
        features_df['ofi_level1'] = (features_df['bid1_size'] - features_df['ask1_size']) / (features_df['bid1_size'] + features_df['ask1_size'])
        features_df['ofi_level2'] = ((features_df['bid1_size'] + features_df['bid2_size']) - (features_df['ask1_size'] + features_df['ask2_size'])) / ((features_df['bid1_size'] + features_df['bid2_size']) + (features_df['ask1_size'] + features_df['ask2_size']))
        
        # 时间特征
        features_df['hour'] = features_df['ts'].dt.hour
        features_df['minute'] = features_df['ts'].dt.minute
        features_df['day_of_week'] = features_df['ts'].dt.dayofweek
        
        # 填充缺失值
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"特征工程后数据形状: {features_df.shape}")
        print(f"新增特征数量: {len(features_df.columns) - len(df.columns)}")
        
        return features_df
    except Exception as e:
        print(f"简单特征工程测试失败: {e}")
        return None

def test_simple_signals():
    """测试简单信号生成"""
    print("=" * 50)
    print("测试简单信号生成...")
    
    try:
        # 创建测试数据
        df = create_simple_test_data(1000)
        
        # 简单信号生成
        out = df.copy()
        out["sig_type"] = None
        out["sig_side"] = 0
        out["signal_strength"] = 0.0
        out["quality_score"] = 0.0
        
        # 基础信号条件
        ofi_threshold = 1.5
        ofi_signal = abs(out["ofi_z"]) >= ofi_threshold
        price_momentum_long = out["ret_1s"] > 0.00001
        price_momentum_short = out["ret_1s"] < -0.00001
        
        # 方向一致性
        long_mask = ofi_signal & (out["ofi_z"] > 0) & price_momentum_long
        short_mask = ofi_signal & (out["ofi_z"] < 0) & price_momentum_short
        
        # 应用信号
        out.loc[long_mask, "sig_type"] = "simple_long"
        out.loc[long_mask, "sig_side"] = 1
        out.loc[long_mask, "signal_strength"] = abs(out["ofi_z"][long_mask])
        out.loc[long_mask, "quality_score"] = abs(out["ofi_z"][long_mask]) / 3.0
        
        out.loc[short_mask, "sig_type"] = "simple_short"
        out.loc[short_mask, "sig_side"] = -1
        out.loc[short_mask, "signal_strength"] = abs(out["ofi_z"][short_mask])
        out.loc[short_mask, "quality_score"] = abs(out["ofi_z"][short_mask]) / 3.0
        
        # 统计信号
        signal_count = out['sig_side'].abs().sum()
        long_signals = (out['sig_side'] == 1).sum()
        short_signals = (out['sig_side'] == -1).sum()
        
        print(f"简单信号统计:")
        print(f"  总信号数: {signal_count}")
        print(f"  多头信号: {long_signals}")
        print(f"  空头信号: {short_signals}")
        
        if signal_count > 0:
            avg_quality = out[out['sig_side'] != 0]['quality_score'].mean()
            print(f"  平均质量评分: {avg_quality:.4f}")
        
        return out
    except Exception as e:
        print(f"简单信号生成测试失败: {e}")
        return None

def main():
    """主测试函数"""
    print("V10.0 深度学习集成简化测试开始")
    print("=" * 50)
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
    
    # 测试深度学习模型
    model_tests = [
        ("LSTM模型", test_lstm_model),
        ("CNN模型", test_cnn_model),
        ("Transformer模型", test_transformer_model),
        ("集成学习模型", test_ensemble_model)
    ]
    
    passed_tests = 0
    total_tests = len(model_tests)
    
    for test_name, test_func in model_tests:
        try:
            if test_func():
                passed_tests += 1
                print(f"[PASS] {test_name}测试通过")
            else:
                print(f"[FAIL] {test_name}测试失败")
        except Exception as e:
            print(f"[ERROR] {test_name}测试异常: {e}")
    
    # 测试特征工程
    try:
        features_df = test_simple_feature_engineering()
        if features_df is not None:
            print("[PASS] 简单特征工程测试通过")
        else:
            print("[FAIL] 简单特征工程测试失败")
    except Exception as e:
        print(f"[ERROR] 简单特征工程测试异常: {e}")
    
    # 测试信号生成
    try:
        signals_df = test_simple_signals()
        if signals_df is not None:
            print("[PASS] 简单信号生成测试通过")
        else:
            print("[FAIL] 简单信号生成测试失败")
    except Exception as e:
        print(f"[ERROR] 简单信号生成测试异常: {e}")
    
    print("=" * 50)
    print(f"V10.0 深度学习集成简化测试完成")
    print(f"模型测试通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    print("=" * 50)

if __name__ == "__main__":
    main()
