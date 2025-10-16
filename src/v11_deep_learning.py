#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11深度学习模块
实现LSTM、Transformer、CNN等深度学习模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11LSTMModel(nn.Module):
    """V11 LSTM模型 - 时间序列预测"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, 
                 output_size: int = 1, dropout: float = 0.2):
        super(V11LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后一个时间步的输出
        output = self.fc(lstm_out[:, -1, :])
        
        return output

class V11TransformerModel(nn.Module):
    """V11 Transformer模型 - 注意力机制"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, output_size: int = 1, dropout: float = 0.1):
        super(V11TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 取最后一个时间步的输出
        output = self.output_layer(x[:, -1, :])
        
        return output

class V11CNNModel(nn.Module):
    """V11 CNN模型 - 卷积神经网络"""
    
    def __init__(self, input_size: int, output_size: int = 1, dropout: float = 0.2):
        super(V11CNNModel, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool1d(2)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        # 转换维度 (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        
        # 卷积层
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        
        # 全局平均池化
        x = torch.mean(x, dim=2)
        
        # 全连接层
        output = self.fc(x)
        
        return output

class V11EnsembleModel(nn.Module):
    """V11集成模型 - 多模型融合"""
    
    def __init__(self, lstm_model, transformer_model, cnn_model, 
                 ensemble_size: int = 3, output_size: int = 1):
        super(V11EnsembleModel, self).__init__()
        
        self.lstm_model = lstm_model
        self.transformer_model = transformer_model
        self.cnn_model = cnn_model
        
        # 集成权重
        self.ensemble_weights = nn.Parameter(torch.ones(ensemble_size) / ensemble_size)
        
        # 元学习器
        self.meta_learner = nn.Sequential(
            nn.Linear(ensemble_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )
        
    def forward(self, x):
        # 获取各模型的预测
        lstm_pred = self.lstm_model(x)
        transformer_pred = self.transformer_model(x)
        cnn_pred = self.cnn_model(x)
        
        # 确保所有预测具有相同的形状
        if lstm_pred.dim() > 1:
            lstm_pred = lstm_pred.squeeze()
        if transformer_pred.dim() > 1:
            transformer_pred = transformer_pred.squeeze()
        if cnn_pred.dim() > 1:
            cnn_pred = cnn_pred.squeeze()
        
        # 加权集成
        ensemble_input = torch.stack([lstm_pred, transformer_pred, cnn_pred], dim=1)
        ensemble_input = ensemble_input * self.ensemble_weights
        
        # 元学习器
        output = self.meta_learner(ensemble_input)
        
        return output

class V11DeepLearning:
    """V11深度学习主类"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.scalers = {}
        self.training_history = {}
        
        logger.info(f"V11深度学习模块初始化完成，使用设备: {self.device}")
    
    def create_lstm_model(self, input_size: int, **kwargs) -> V11LSTMModel:
        """创建LSTM模型"""
        model = V11LSTMModel(input_size=input_size, **kwargs)
        model.to(self.device)
        self.models['lstm'] = model
        logger.info(f"LSTM模型创建完成，输入维度: {input_size}")
        return model
    
    def create_transformer_model(self, input_size: int, **kwargs) -> V11TransformerModel:
        """创建Transformer模型"""
        model = V11TransformerModel(input_size=input_size, **kwargs)
        model.to(self.device)
        self.models['transformer'] = model
        logger.info(f"Transformer模型创建完成，输入维度: {input_size}")
        return model
    
    def create_cnn_model(self, input_size: int, **kwargs) -> V11CNNModel:
        """创建CNN模型"""
        model = V11CNNModel(input_size=input_size, **kwargs)
        model.to(self.device)
        self.models['cnn'] = model
        logger.info(f"CNN模型创建完成，输入维度: {input_size}")
        return model
    
    def create_ensemble_model(self, input_size: int, **kwargs) -> V11EnsembleModel:
        """创建集成模型"""
        # 先创建基础模型
        lstm_model = self.create_lstm_model(input_size)
        transformer_model = self.create_transformer_model(input_size)
        cnn_model = self.create_cnn_model(input_size)
        
        # 创建集成模型
        ensemble_model = V11EnsembleModel(lstm_model, transformer_model, cnn_model)
        ensemble_model.to(self.device)
        self.models['ensemble'] = ensemble_model
        logger.info("集成模型创建完成")
        return ensemble_model
    
    def prepare_data(self, features: np.ndarray, targets: np.ndarray, 
                     sequence_length: int = 60, test_size: float = 0.2) -> Tuple:
        """准备训练数据"""
        logger.info("开始准备训练数据...")
        
        # 数据标准化
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        features_scaled = feature_scaler.fit_transform(features)
        targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # 创建序列数据
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(targets_scaled[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # 分割训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        # 保存标准化器
        self.scalers['features'] = feature_scaler
        self.scalers['targets'] = target_scaler
        
        logger.info(f"数据准备完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_name: str, X_train: torch.Tensor, y_train: torch.Tensor,
                   X_test: torch.Tensor, y_test: torch.Tensor, 
                   epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001) -> Dict:
        """训练模型"""
        logger.info(f"开始训练 {model_name} 模型...")
        
        model = self.models[model_name]
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 训练历史
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 测试阶段
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs.squeeze(), y_test).item()
            
            train_losses.append(train_loss / len(train_loader))
            test_losses.append(test_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.6f}, Test Loss: {test_losses[-1]:.6f}")
        
        # 保存训练历史
        self.training_history[model_name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1]
        }
        
        logger.info(f"{model_name} 模型训练完成")
        return self.training_history[model_name]
    
    def predict(self, model_name: str, X: torch.Tensor) -> np.ndarray:
        """模型预测"""
        model = self.models[model_name]
        model.eval()
        
        with torch.no_grad():
            predictions = model(X)
            predictions = predictions.cpu().numpy()
        
        # 反标准化
        if 'targets' in self.scalers:
            predictions = self.scalers['targets'].inverse_transform(predictions)
        
        return predictions
    
    def evaluate_model(self, model_name: str, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict:
        """评估模型性能"""
        model = self.models[model_name]
        model.eval()
        
        with torch.no_grad():
            predictions = model(X_test)
            predictions = predictions.cpu().numpy()
            y_test_np = y_test.cpu().numpy()
        
        # 反标准化
        if 'targets' in self.scalers:
            predictions = self.scalers['targets'].inverse_transform(predictions)
            y_test_np = self.scalers['targets'].inverse_transform(y_test_np.reshape(-1, 1)).flatten()
        
        # 计算评估指标
        mse = np.mean((predictions.flatten() - y_test_np) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions.flatten() - y_test_np))
        mape = np.mean(np.abs((y_test_np - predictions.flatten()) / y_test_np)) * 100
        
        # 计算方向准确率
        direction_accuracy = np.mean(
            np.sign(predictions.flatten()[1:] - predictions.flatten()[:-1]) == 
            np.sign(y_test_np[1:] - y_test_np[:-1])
        ) * 100
        
        evaluation = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }
        
        logger.info(f"{model_name} 模型评估结果:")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  方向准确率: {direction_accuracy:.2f}%")
        
        return evaluation
    
    def save_models(self, save_path: str):
        """保存模型"""
        torch.save({
            'models': {name: model.state_dict() for name, model in self.models.items()},
            'scalers': self.scalers,
            'training_history': self.training_history
        }, save_path)
        logger.info(f"模型已保存到: {save_path}")
    
    def load_models(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        for name, state_dict in checkpoint['models'].items():
            if name in self.models:
                self.models[name].load_state_dict(state_dict)
        
        self.scalers = checkpoint['scalers']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"模型已从 {load_path} 加载")

def main():
    """主函数 - 演示V11深度学习模块"""
    logger.info("=" * 60)
    logger.info("V11深度学习模块演示")
    logger.info("=" * 60)
    
    # 创建深度学习实例
    dl = V11DeepLearning()
    
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    sequence_length = 60
    
    # 生成特征数据
    features = np.random.randn(n_samples, n_features)
    
    # 生成目标数据（价格变化）
    targets = np.random.randn(n_samples) * 0.01
    
    logger.info(f"生成示例数据: {n_samples} 样本, {n_features} 特征")
    
    # 准备数据
    X_train, X_test, y_train, y_test = dl.prepare_data(
        features, targets, sequence_length=sequence_length
    )
    
    # 创建和训练模型
    models_to_train = ['lstm', 'transformer', 'cnn']
    
    for model_name in models_to_train:
        logger.info(f"\n训练 {model_name} 模型...")
        
        if model_name == 'lstm':
            model = dl.create_lstm_model(input_size=n_features)
        elif model_name == 'transformer':
            model = dl.create_transformer_model(input_size=n_features)
        elif model_name == 'cnn':
            model = dl.create_cnn_model(input_size=n_features)
        
        # 训练模型
        training_history = dl.train_model(
            model_name, X_train, y_train, X_test, y_test,
            epochs=50, batch_size=32, learning_rate=0.001
        )
        
        # 评估模型
        evaluation = dl.evaluate_model(model_name, X_test, y_test)
    
    # 创建集成模型
    logger.info("\n创建集成模型...")
    ensemble_model = dl.create_ensemble_model(input_size=n_features)
    
    # 训练集成模型
    ensemble_history = dl.train_model(
        'ensemble', X_train, y_train, X_test, y_test,
        epochs=50, batch_size=32, learning_rate=0.001
    )
    
    # 评估集成模型
    ensemble_evaluation = dl.evaluate_model('ensemble', X_test, y_test)
    
    logger.info("\n" + "=" * 60)
    logger.info("V11深度学习模块演示完成")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
