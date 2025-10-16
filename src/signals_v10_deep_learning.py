import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import joblib
import os
from datetime import datetime

# 导入深度学习模型
try:
    from .models.lstm_predictor import LSTMPredictor, LSTMTrainer, LSTMDataProcessor
    from .models.cnn_recognizer import CNNPatternRecognizer, CNNTrainer, CNNDataProcessor
    from .models.transformer_predictor import TransformerPredictor, TransformerTrainer, TransformerDataProcessor
    from .models.ensemble_predictor import EnsemblePredictor, EnsembleTrainer, EnsembleDataProcessor
    from .features.advanced_feature_engineering import AdvancedFeatureEngineering, create_advanced_features
except ImportError:
    from models.lstm_predictor import LSTMPredictor, LSTMTrainer, LSTMDataProcessor
    from models.cnn_recognizer import CNNPatternRecognizer, CNNTrainer, CNNDataProcessor
    from models.transformer_predictor import TransformerPredictor, TransformerTrainer, TransformerDataProcessor
    from models.ensemble_predictor import EnsemblePredictor, EnsembleTrainer, EnsembleDataProcessor
    from features.advanced_feature_engineering import AdvancedFeatureEngineering, create_advanced_features

class DeepLearningSignalGenerator:
    """
    V10 深度学习信号生成器
    集成LSTM、CNN、Transformer和集成学习模型
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.feature_engineer = None
        self.is_trained = False
        self.model_predictions = {}
        self.uncertainty_scores = {}
        
        # 初始化特征工程器
        self.feature_engineer = AdvancedFeatureEngineering(
            feature_dim=config.get('features', {}).get('feature_dim', 50)
        )
        
        # 初始化模型
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化深度学习模型"""
        dl_config = self.config.get('deep_learning', {})
        
        # LSTM模型
        if dl_config.get('models', {}).get('lstm', {}).get('enabled', True):
            lstm_config = dl_config['models']['lstm']
            self.models['lstm'] = LSTMPredictor(
                input_dim=lstm_config.get('input_dim', 50),
                hidden_dim=lstm_config.get('hidden_dim', 128),
                num_layers=lstm_config.get('num_layers', 3),
                dropout=lstm_config.get('dropout', 0.2)
            )
        
        # CNN模型
        if dl_config.get('models', {}).get('cnn', {}).get('enabled', True):
            cnn_config = dl_config['models']['cnn']
            self.models['cnn'] = CNNPatternRecognizer(
                input_channels=cnn_config.get('input_channels', 1),
                num_classes=cnn_config.get('num_classes', 10),
                feature_dim=cnn_config.get('feature_dim', 50),
                sequence_length=cnn_config.get('sequence_length', 60)
            )
        
        # Transformer模型
        if dl_config.get('models', {}).get('transformer', {}).get('enabled', True):
            transformer_config = dl_config['models']['transformer']
            self.models['transformer'] = TransformerPredictor(
                input_dim=transformer_config.get('input_dim', 50),
                d_model=transformer_config.get('d_model', 128),
                nhead=transformer_config.get('nhead', 8),
                num_layers=transformer_config.get('num_layers', 6),
                dim_feedforward=transformer_config.get('dim_feedforward', 512),
                dropout=transformer_config.get('dropout', 0.1)
            )
        
        # 集成学习模型
        if dl_config.get('models', {}).get('ensemble', {}).get('enabled', True):
            ensemble_config = dl_config['models']['ensemble']
            self.models['ensemble'] = EnsemblePredictor(
                input_dim=ensemble_config.get('input_dim', 50),
                sequence_length=ensemble_config.get('sequence_length', 60),
                ensemble_method=ensemble_config.get('ensemble_method', 'weighted_average')
            )
        
        print(f"深度学习模型初始化完成: {list(self.models.keys())}")
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'signal_quality') -> pd.DataFrame:
        """
        准备训练数据
        Args:
            df: 输入数据
            target_col: 目标列名
        Returns:
            包含高级特征的DataFrame
        """
        # 创建高级特征
        features_df = self.feature_engineer.create_all_features(df)
        
        # 如果目标列不存在，创建默认值
        if target_col not in features_df.columns:
            features_df[target_col] = 0.5  # 默认中性质量
        
        return features_df
    
    def train_models(self, df: pd.DataFrame, target_col: str = 'signal_quality'):
        """
        训练所有深度学习模型
        Args:
            df: 训练数据
            target_col: 目标列名
        """
        print("开始训练深度学习模型...")
        
        # 准备训练数据
        training_df = self.prepare_training_data(df, target_col)
        
        # 选择特征
        feature_cols = [col for col in training_df.columns if col != target_col]
        selected_features = self.feature_engineer.select_features(
            training_df[feature_cols], 
            training_df[target_col], 
            k=self.config.get('features', {}).get('feature_dim', 50)
        )
        
        # 缩放特征
        training_df[selected_features] = self.feature_engineer.scale_features(
            training_df[selected_features], 
            method=self.config.get('features', {}).get('feature_scaling', 'standard')
        )
        
        # 训练各个模型
        for model_name, model in self.models.items():
            print(f"训练{model_name}模型...")
            
            try:
                if model_name == 'lstm':
                    self._train_lstm_model(training_df, selected_features, target_col)
                elif model_name == 'cnn':
                    self._train_cnn_model(training_df, selected_features, target_col)
                elif model_name == 'transformer':
                    self._train_transformer_model(training_df, selected_features, target_col)
                elif model_name == 'ensemble':
                    self._train_ensemble_model(training_df, selected_features, target_col)
                
                print(f"{model_name}模型训练完成")
                
            except Exception as e:
                print(f"{model_name}模型训练失败: {e}")
                continue
        
        self.is_trained = True
        print("所有深度学习模型训练完成")
    
    def _train_lstm_model(self, df: pd.DataFrame, feature_cols: List[str], target_col: str):
        """训练LSTM模型"""
        lstm_config = self.config['deep_learning']['models']['lstm']
        
        # 准备序列数据
        processor = LSTMDataProcessor(
            sequence_length=self.config.get('features', {}).get('sequence_length', 60),
            feature_dim=len(feature_cols)
        )
        X, y = processor.prepare_sequences(df, target_col)
        train_loader, val_loader = processor.create_data_loaders(X, y, lstm_config.get('batch_size', 32))
        
        # 训练模型
        trainer = LSTMTrainer(self.models['lstm'], learning_rate=lstm_config.get('learning_rate', 0.001))
        results = trainer.train(train_loader, val_loader, epochs=lstm_config.get('epochs', 100))
        
        # 保存模型
        model_path = f"models/lstm_model_v10.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model_path)
        
        print(f"LSTM模型训练结果: 最佳验证损失 = {results['best_val_loss']:.6f}")
    
    def _train_cnn_model(self, df: pd.DataFrame, feature_cols: List[str], target_col: str):
        """训练CNN模型"""
        cnn_config = self.config['deep_learning']['models']['cnn']
        
        # 准备序列数据
        processor = CNNDataProcessor(
            sequence_length=self.config.get('features', {}).get('sequence_length', 60),
            feature_dim=len(feature_cols),
            num_classes=cnn_config.get('num_classes', 10)
        )
        X, y = processor.prepare_sequences(df, target_col)
        train_loader, val_loader = processor.create_data_loaders(X, y, cnn_config.get('batch_size', 32))
        
        # 训练模型
        trainer = CNNTrainer(self.models['cnn'], learning_rate=cnn_config.get('learning_rate', 0.001))
        results = trainer.train(train_loader, val_loader, epochs=cnn_config.get('epochs', 100))
        
        # 保存模型
        model_path = f"models/cnn_model_v10.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model_path)
        
        print(f"CNN模型训练结果: 最佳验证损失 = {results['best_val_loss']:.6f}, 最佳验证准确率 = {results['best_val_accuracy']:.2f}%")
    
    def _train_transformer_model(self, df: pd.DataFrame, feature_cols: List[str], target_col: str):
        """训练Transformer模型"""
        transformer_config = self.config['deep_learning']['models']['transformer']
        
        # 准备序列数据
        processor = TransformerDataProcessor(
            sequence_length=self.config.get('features', {}).get('sequence_length', 60),
            feature_dim=len(feature_cols)
        )
        X, y = processor.prepare_sequences(df, target_col)
        train_loader, val_loader = processor.create_data_loaders(X, y, transformer_config.get('batch_size', 32))
        
        # 训练模型
        trainer = TransformerTrainer(self.models['transformer'], learning_rate=transformer_config.get('learning_rate', 0.001))
        results = trainer.train(train_loader, val_loader, epochs=transformer_config.get('epochs', 100))
        
        # 保存模型
        model_path = f"models/transformer_model_v10.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model_path)
        
        print(f"Transformer模型训练结果: 最佳验证损失 = {results['best_val_loss']:.6f}")
    
    def _train_ensemble_model(self, df: pd.DataFrame, feature_cols: List[str], target_col: str):
        """训练集成学习模型"""
        ensemble_config = self.config['deep_learning']['models']['ensemble']
        
        # 准备序列数据
        processor = EnsembleDataProcessor(
            sequence_length=self.config.get('features', {}).get('sequence_length', 60),
            feature_dim=len(feature_cols)
        )
        X, y = processor.prepare_sequences(df, target_col)
        train_loader, val_loader = processor.create_data_loaders(X, y, ensemble_config.get('batch_size', 32))
        
        # 训练模型
        trainer = EnsembleTrainer(self.models['ensemble'], learning_rate=ensemble_config.get('learning_rate', 0.001))
        results = trainer.train(train_loader, val_loader, epochs=ensemble_config.get('epochs', 100))
        
        # 保存模型
        model_path = f"models/ensemble_model_v10.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model_path)
        
        print(f"集成学习模型训练结果: 最佳验证损失 = {results['best_val_loss']:.6f}")
    
    def predict_signal_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预测信号质量
        Args:
            df: 输入数据
        Returns:
            包含预测结果的DataFrame
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train_models方法")
        
        # 准备特征
        features_df = self.feature_engineer.create_all_features(df)
        selected_features = self.feature_engineer.get_selected_features()
        
        # 缩放特征
        features_df[selected_features] = self.feature_engineer.scale_features(
            features_df[selected_features], 
            method=self.config.get('features', {}).get('feature_scaling', 'standard')
        )
        
        # 准备序列数据
        sequence_length = self.config.get('features', {}).get('sequence_length', 60)
        X_sequences, _ = self.feature_engineer.create_sequence_data(
            features_df, sequence_length, 'signal_quality'
        )
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X_sequences)
        
        # 各模型预测
        predictions = {}
        uncertainties = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'ensemble':
                    pred, uncertainty = model.predict_signal_quality(X_tensor)
                    predictions[model_name] = pred.numpy()
                    uncertainties[model_name] = uncertainty.numpy()
                else:
                    pred = model.predict_signal_quality(X_tensor)
                    predictions[model_name] = pred.numpy()
                    uncertainties[model_name] = np.ones_like(pred.numpy()) * 0.5  # 默认不确定性
                
            except Exception as e:
                print(f"{model_name}模型预测失败: {e}")
                continue
        
        # 保存预测结果
        self.model_predictions = predictions
        self.uncertainty_scores = uncertainties
        
        # 创建结果DataFrame
        result_df = df.copy()
        
        # 添加预测结果
        for model_name, pred in predictions.items():
            result_df[f'{model_name}_prediction'] = np.nan
            result_df.iloc[sequence_length:, result_df.columns.get_loc(f'{model_name}_prediction')] = pred.flatten()
        
        # 添加不确定性
        for model_name, uncertainty in uncertainties.items():
            result_df[f'{model_name}_uncertainty'] = np.nan
            result_df.iloc[sequence_length:, result_df.columns.get_loc(f'{model_name}_uncertainty')] = uncertainty.flatten()
        
        return result_df

def gen_signals_v10_deep_learning_enhanced(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    V10 深度学习增强信号生成
    Args:
        df: 输入数据
        params: 参数配置
    Returns:
        包含信号的DataFrame
    """
    p = params["signals"]["deep_learning_enhanced"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    out["quality_score"] = 0.0
    out["ml_prediction"] = 0.0
    out["ml_uncertainty"] = 0.0
    out["deep_learning_score"] = 0.0
    
    # 初始化深度学习信号生成器
    if not hasattr(gen_signals_v10_deep_learning_enhanced, 'dl_generator'):
        gen_signals_v10_deep_learning_enhanced.dl_generator = DeepLearningSignalGenerator(params)
        
        # 训练模型
        gen_signals_v10_deep_learning_enhanced.dl_generator.train_models(df)
    
    dl_generator = gen_signals_v10_deep_learning_enhanced.dl_generator
    
    # 获取深度学习预测
    dl_predictions = dl_generator.predict_signal_quality(df)
    
    # 基础信号生成
    ofi_threshold = p.get("ofi_z_min", 1.2)
    ofi_signal = abs(out["ofi_z"]) >= ofi_threshold
    
    # 深度学习增强筛选
    ml_threshold = p.get("min_ml_prediction", 0.8)
    uncertainty_min = p.get("min_uncertainty", 0.1)
    uncertainty_max = p.get("max_uncertainty", 0.9)
    
    # 使用集成学习模型的预测
    if 'ensemble_prediction' in dl_predictions.columns:
        ml_enhanced = (dl_predictions['ensemble_prediction'] >= ml_threshold) & \
                      (dl_predictions['ensemble_uncertainty'] >= uncertainty_min) & \
                      (dl_predictions['ensemble_uncertainty'] <= uncertainty_max)
        out["ml_prediction"] = dl_predictions['ensemble_prediction']
        out["ml_uncertainty"] = dl_predictions['ensemble_uncertainty']
    else:
        # 如果没有集成学习模型，使用LSTM模型
        if 'lstm_prediction' in dl_predictions.columns:
            ml_enhanced = dl_predictions['lstm_prediction'] >= ml_threshold
            out["ml_prediction"] = dl_predictions['lstm_prediction']
            out["ml_uncertainty"] = dl_predictions.get('lstm_uncertainty', 0.5)
        else:
            ml_enhanced = pd.Series(False, index=out.index)
            out["ml_prediction"] = 0.5
            out["ml_uncertainty"] = 0.5
    
    # 信号强度筛选
    signal_strength = abs(out["ofi_z"])
    min_signal_strength = p.get("min_signal_strength", 1.6)
    strong_signal = signal_strength >= min_signal_strength
    
    # 价格动量确认
    price_momentum_threshold = p.get("price_momentum_threshold", 0.00001)
    price_momentum_long = out["ret_1s"] > price_momentum_threshold
    price_momentum_short = out["ret_1s"] < -price_momentum_threshold
    
    # 方向一致性检查
    direction_consistent_long = (out["ofi_z"] > 0) & price_momentum_long
    direction_consistent_short = (out["ofi_z"] < 0) & price_momentum_short
    
    # 组合深度学习增强信号
    long_mask = ofi_signal & strong_signal & ml_enhanced & direction_consistent_long
    short_mask = ofi_signal & strong_signal & ml_enhanced & direction_consistent_short
    
    # 应用信号
    out.loc[long_mask, "sig_type"] = "deep_learning_enhanced"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[long_mask, "signal_strength"] = signal_strength[long_mask]
    out.loc[long_mask, "quality_score"] = out["ml_prediction"][long_mask]
    out.loc[long_mask, "deep_learning_score"] = out["ml_prediction"][long_mask]
    
    out.loc[short_mask, "sig_type"] = "deep_learning_enhanced"
    out.loc[short_mask, "sig_side"] = -1
    out.loc[short_mask, "signal_strength"] = signal_strength[short_mask]
    out.loc[short_mask, "quality_score"] = out["ml_prediction"][short_mask]
    out.loc[short_mask, "deep_learning_score"] = out["ml_prediction"][short_mask]
    
    return out

def gen_signals_v10_real_time_optimized(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    V10 实时优化信号生成
    Args:
        df: 输入数据
        params: 参数配置
    Returns:
        包含信号的DataFrame
    """
    p = params["signals"]["deep_learning_enhanced"]
    rt_params = params["signals"]["real_time_optimization"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    out["quality_score"] = 0.0
    out["real_time_score"] = 0.0
    
    # 实时优化参数
    if not hasattr(gen_signals_v10_real_time_optimized, 'optimization_state'):
        gen_signals_v10_real_time_optimized.optimization_state = {
            'performance_history': [],
            'current_threshold': p.get("ofi_z_min", 1.2),
            'adaptation_rate': rt_params.get("adaptation_rate", 0.15),
            'update_counter': 0
        }
    
    state = gen_signals_v10_real_time_optimized.optimization_state
    
    # 基础信号生成
    ofi_threshold = state['current_threshold']
    ofi_signal = abs(out["ofi_z"]) >= ofi_threshold
    
    # 信号强度筛选
    signal_strength = abs(out["ofi_z"])
    min_signal_strength = p.get("min_signal_strength", 1.6)
    strong_signal = signal_strength >= min_signal_strength
    
    # 价格动量确认
    price_momentum_threshold = p.get("price_momentum_threshold", 0.00001)
    price_momentum_long = out["ret_1s"] > price_momentum_threshold
    price_momentum_short = out["ret_1s"] < -price_momentum_threshold
    
    # 方向一致性检查
    direction_consistent_long = (out["ofi_z"] > 0) & price_momentum_long
    direction_consistent_short = (out["ofi_z"] < 0) & price_momentum_short
    
    # 组合实时优化信号
    long_mask = ofi_signal & strong_signal & direction_consistent_long
    short_mask = ofi_signal & strong_signal & direction_consistent_short
    
    # 应用信号
    out.loc[long_mask, "sig_type"] = "real_time_optimized"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[long_mask, "signal_strength"] = signal_strength[long_mask]
    out.loc[long_mask, "quality_score"] = signal_strength[long_mask] / 3.0
    out.loc[long_mask, "real_time_score"] = signal_strength[long_mask] / 3.0
    
    out.loc[short_mask, "sig_type"] = "real_time_optimized"
    out.loc[short_mask, "sig_side"] = -1
    out.loc[short_mask, "signal_strength"] = signal_strength[short_mask]
    out.loc[short_mask, "quality_score"] = signal_strength[short_mask] / 3.0
    out.loc[short_mask, "real_time_score"] = signal_strength[short_mask] / 3.0
    
    # 实时优化更新
    state['update_counter'] += 1
    if state['update_counter'] >= rt_params.get("update_frequency", 5):
        # 更新优化状态
        state['update_counter'] = 0
        # 这里可以添加实时性能评估和阈值调整逻辑
    
    return out

if __name__ == "__main__":
    # 测试深度学习信号生成
    print("测试深度学习信号生成...")
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    
    test_data = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=n_samples, freq='1s'),
        'price': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'volume': np.random.randint(100, 1000, n_samples),
        'bid1': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - 0.01,
        'ask1': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + 0.01,
        'bid1_size': np.random.randint(100, 500, n_samples),
        'ask1_size': np.random.randint(100, 500, n_samples),
        'bid2': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - 0.02,
        'ask2': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + 0.02,
        'bid2_size': np.random.randint(50, 300, n_samples),
        'ask2_size': np.random.randint(50, 300, n_samples),
        'bid3': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - 0.03,
        'ask3': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + 0.03,
        'bid3_size': np.random.randint(20, 200, n_samples),
        'ask3_size': np.random.randint(20, 200, n_samples),
        'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + 0.05,
        'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - 0.05,
        'ofi_z': np.random.randn(n_samples) * 2,
        'cvd_z': np.random.randn(n_samples) * 2,
        'ret_1s': np.random.randn(n_samples) * 0.001,
        'atr': np.random.uniform(0.01, 0.05, n_samples),
        'vwap': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'signal_quality': np.random.uniform(0, 1, n_samples)
    })
    
    # 测试深度学习信号生成
    print("深度学习信号生成测试完成!")
