#!/usr/bin/env python3
"""
V10.0 深度学习集成测试脚本
测试深度学习模型、高级特征工程、信号生成等功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yaml
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入V10模块
try:
    from src.models.lstm_predictor import create_lstm_model, train_lstm_model
    from src.models.cnn_recognizer import create_cnn_model, train_cnn_model
    from src.models.transformer_predictor import create_transformer_model, train_transformer_model
    from src.models.ensemble_predictor import create_ensemble_model, train_ensemble_model
    from src.features.advanced_feature_engineering import create_advanced_features
    from src.signals_v10_deep_learning import gen_signals_v10_deep_learning_enhanced, gen_signals_v10_real_time_optimized
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有V10模块都已正确创建")
    # 尝试直接导入
    try:
        import sys
        sys.path.append('src/models')
        sys.path.append('src/features')
        from lstm_predictor import create_lstm_model, train_lstm_model
        from cnn_recognizer import create_cnn_model, train_cnn_model
        from transformer_predictor import create_transformer_model, train_transformer_model
        from ensemble_predictor import create_ensemble_model, train_ensemble_model
        from advanced_feature_engineering import create_advanced_features
        from signals_v10_deep_learning import gen_signals_v10_deep_learning_enhanced, gen_signals_v10_real_time_optimized
        print("使用直接导入方式成功")
    except ImportError as e2:
        print(f"直接导入也失败: {e2}")
        sys.exit(1)

def load_v10_config():
    """加载V10配置"""
    config_path = "config/params_v10_deep_learning.yaml"
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def create_test_data(n_samples: int = 2000) -> pd.DataFrame:
    """创建测试数据"""
    np.random.seed(42)
    
    # 生成价格数据
    price_base = 100
    price_changes = np.random.randn(n_samples) * 0.01
    prices = price_base + np.cumsum(price_changes)
    
    # 生成订单簿数据
    bid_prices = prices - np.random.uniform(0.01, 0.05, n_samples)
    ask_prices = prices + np.random.uniform(0.01, 0.05, n_samples)
    
    # 生成成交量数据
    volumes = np.random.randint(100, 1000, n_samples)
    
    # 生成订单簿深度
    bid_sizes = np.random.randint(100, 500, (n_samples, 3))
    ask_sizes = np.random.randint(100, 500, (n_samples, 3))
    
    # 生成技术指标
    ofi_z = np.random.randn(n_samples) * 2
    cvd_z = np.random.randn(n_samples) * 2
    ret_1s = np.random.randn(n_samples) * 0.001
    atr = np.random.uniform(0.01, 0.05, n_samples)
    vwap = prices + np.random.randn(n_samples) * 0.005
    
    # 生成信号质量标签
    signal_quality = np.random.uniform(0, 1, n_samples)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=n_samples, freq='1s'),
        'price': prices,
        'volume': volumes,
        'bid1': bid_prices,
        'ask1': ask_prices,
        'bid1_size': bid_sizes[:, 0],
        'ask1_size': ask_sizes[:, 0],
        'bid2': bid_prices - 0.01,
        'ask2': ask_prices + 0.01,
        'bid2_size': bid_sizes[:, 1],
        'ask2_size': ask_sizes[:, 1],
        'bid3': bid_prices - 0.02,
        'ask3': ask_prices + 0.02,
        'bid3_size': bid_sizes[:, 2],
        'ask3_size': ask_sizes[:, 2],
        'high': prices + np.random.uniform(0, 0.05, n_samples),
        'low': prices - np.random.uniform(0, 0.05, n_samples),
        'ofi_z': ofi_z,
        'cvd_z': cvd_z,
        'ret_1s': ret_1s,
        'atr': atr,
        'vwap': vwap,
        'signal_quality': signal_quality
    })
    
    return df

def test_advanced_feature_engineering():
    """测试高级特征工程"""
    print("=" * 50)
    print("测试高级特征工程...")
    
    # 创建测试数据
    df = create_test_data(1000)
    print(f"原始数据形状: {df.shape}")
    
    # 创建高级特征
    features_df, selected_features = create_advanced_features(df, feature_dim=50)
    print(f"特征工程后数据形状: {features_df.shape}")
    print(f"选择的特征数量: {len(selected_features)}")
    print(f"前10个选择的特征: {selected_features[:10]}")
    
    return features_df, selected_features

def test_lstm_model():
    """测试LSTM模型"""
    print("=" * 50)
    print("测试LSTM模型...")
    
    # 创建测试数据
    df = create_test_data(1000)
    
    # 创建LSTM模型
    model = create_lstm_model(input_dim=50, hidden_dim=128, num_layers=3)
    print(f"LSTM模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size, seq_len, input_dim = 32, 60, 50
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    print(f"LSTM前向传播测试: 输入形状 {x.shape} -> 输出形状 {output.shape}")
    
    return model

def test_cnn_model():
    """测试CNN模型"""
    print("=" * 50)
    print("测试CNN模型...")
    
    # 创建CNN模型
    model = create_cnn_model(input_channels=1, num_classes=10, feature_dim=50, sequence_length=60)
    print(f"CNN模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size, channels, seq_len = 32, 1, 60
    x = torch.randn(batch_size, channels, seq_len)
    output = model(x)
    print(f"CNN前向传播测试: 输入形状 {x.shape} -> 输出形状 {output.shape}")
    
    return model

def test_transformer_model():
    """测试Transformer模型"""
    print("=" * 50)
    print("测试Transformer模型...")
    
    # 创建Transformer模型
    model = create_transformer_model(input_dim=50, d_model=128, nhead=8, num_layers=6)
    print(f"Transformer模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size, seq_len, input_dim = 32, 60, 50
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    print(f"Transformer前向传播测试: 输入形状 {x.shape} -> 输出形状 {output.shape}")
    
    return model

def test_ensemble_model():
    """测试集成学习模型"""
    print("=" * 50)
    print("测试集成学习模型...")
    
    # 创建集成学习模型
    model = create_ensemble_model(input_dim=50, sequence_length=60, ensemble_method='weighted_average')
    print(f"集成学习模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size, seq_len, input_dim = 32, 60, 50
    x = torch.randn(batch_size, seq_len, input_dim)
    prediction, uncertainty = model(x)
    print(f"集成学习前向传播测试: 输入形状 {x.shape} -> 预测形状 {prediction.shape}, 不确定性形状 {uncertainty.shape}")
    
    return model

def test_deep_learning_signals():
    """测试深度学习信号生成"""
    print("=" * 50)
    print("测试深度学习信号生成...")
    
    # 加载配置
    config = load_v10_config()
    if config is None:
        print("无法加载配置，跳过深度学习信号测试")
        return
    
    # 创建测试数据
    df = create_test_data(1000)
    print(f"测试数据形状: {df.shape}")
    
    # 测试深度学习增强信号
    try:
        signals_df = gen_signals_v10_deep_learning_enhanced(df, config)
        print(f"深度学习增强信号生成完成，数据形状: {signals_df.shape}")
        
        # 统计信号
        signal_count = signals_df['sig_side'].abs().sum()
        long_signals = (signals_df['sig_side'] == 1).sum()
        short_signals = (signals_df['sig_side'] == -1).sum()
        
        print(f"信号统计:")
        print(f"  总信号数: {signal_count}")
        print(f"  多头信号: {long_signals}")
        print(f"  空头信号: {short_signals}")
        
        if signal_count > 0:
            avg_quality = signals_df[signals_df['sig_side'] != 0]['quality_score'].mean()
            avg_ml_pred = signals_df[signals_df['sig_side'] != 0]['ml_prediction'].mean()
            print(f"  平均质量评分: {avg_quality:.4f}")
            print(f"  平均ML预测: {avg_ml_pred:.4f}")
        
    except Exception as e:
        print(f"深度学习信号生成测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试实时优化信号
    try:
        rt_signals_df = gen_signals_v10_real_time_optimized(df, config)
        print(f"实时优化信号生成完成，数据形状: {rt_signals_df.shape}")
        
        # 统计信号
        rt_signal_count = rt_signals_df['sig_side'].abs().sum()
        rt_long_signals = (rt_signals_df['sig_side'] == 1).sum()
        rt_short_signals = (rt_signals_df['sig_side'] == -1).sum()
        
        print(f"实时优化信号统计:")
        print(f"  总信号数: {rt_signal_count}")
        print(f"  多头信号: {rt_long_signals}")
        print(f"  空头信号: {rt_short_signals}")
        
    except Exception as e:
        print(f"实时优化信号生成测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_model_training():
    """测试模型训练"""
    print("=" * 50)
    print("测试模型训练...")
    
    # 创建训练数据
    df = create_test_data(2000)
    
    # 测试LSTM模型训练
    try:
        print("测试LSTM模型训练...")
        lstm_results = train_lstm_model(
            df, 
            target_col='signal_quality',
            input_dim=50,
            hidden_dim=128,
            num_layers=3,
            epochs=10,  # 减少训练轮数用于测试
            batch_size=32,
            learning_rate=0.001
        )
        print(f"LSTM训练完成: 最佳验证损失 = {lstm_results['best_val_loss']:.6f}")
    except Exception as e:
        print(f"LSTM模型训练失败: {e}")
    
    # 测试CNN模型训练
    try:
        print("测试CNN模型训练...")
        cnn_results = train_cnn_model(
            df,
            target_col='signal_quality',
            input_channels=1,
            num_classes=10,
            feature_dim=50,
            sequence_length=60,
            epochs=10,  # 减少训练轮数用于测试
            batch_size=32,
            learning_rate=0.001
        )
        print(f"CNN训练完成: 最佳验证损失 = {cnn_results['best_val_loss']:.6f}, 最佳验证准确率 = {cnn_results['best_val_accuracy']:.2f}%")
    except Exception as e:
        print(f"CNN模型训练失败: {e}")
    
    # 测试Transformer模型训练
    try:
        print("测试Transformer模型训练...")
        transformer_results = train_transformer_model(
            df,
            target_col='signal_quality',
            input_dim=50,
            d_model=128,
            nhead=8,
            num_layers=6,
            epochs=10,  # 减少训练轮数用于测试
            batch_size=32,
            learning_rate=0.001
        )
        print(f"Transformer训练完成: 最佳验证损失 = {transformer_results['best_val_loss']:.6f}")
    except Exception as e:
        print(f"Transformer模型训练失败: {e}")
    
    # 测试集成学习模型训练
    try:
        print("测试集成学习模型训练...")
        ensemble_results = train_ensemble_model(
            df,
            target_col='signal_quality',
            input_dim=50,
            sequence_length=60,
            ensemble_method='weighted_average',
            epochs=10,  # 减少训练轮数用于测试
            batch_size=32,
            learning_rate=0.001
        )
        print(f"集成学习训练完成: 最佳验证损失 = {ensemble_results['best_val_loss']:.6f}")
    except Exception as e:
        print(f"集成学习模型训练失败: {e}")

def main():
    """主测试函数"""
    print("V10.0 深度学习集成测试开始")
    print("=" * 50)
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
    
    # 测试高级特征工程
    try:
        features_df, selected_features = test_advanced_feature_engineering()
    except Exception as e:
        print(f"高级特征工程测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试深度学习模型
    try:
        lstm_model = test_lstm_model()
    except Exception as e:
        print(f"LSTM模型测试失败: {e}")
    
    try:
        cnn_model = test_cnn_model()
    except Exception as e:
        print(f"CNN模型测试失败: {e}")
    
    try:
        transformer_model = test_transformer_model()
    except Exception as e:
        print(f"Transformer模型测试失败: {e}")
    
    try:
        ensemble_model = test_ensemble_model()
    except Exception as e:
        print(f"集成学习模型测试失败: {e}")
    
    # 测试深度学习信号生成
    try:
        test_deep_learning_signals()
    except Exception as e:
        print(f"深度学习信号生成测试失败: {e}")
    
    # 测试模型训练（可选，需要较长时间）
    # try:
    #     test_model_training()
    # except Exception as e:
    #     print(f"模型训练测试失败: {e}")
    
    print("=" * 50)
    print("V10.0 深度学习集成测试完成")
    print("=" * 50)

if __name__ == "__main__":
    main()
