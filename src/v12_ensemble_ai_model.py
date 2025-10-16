"""
V12 集成AI模型
融合OFI专家模型 + 深度学习模型 (LSTM/Transformer/CNN)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 导入V12组件
try:
    from .v12_ofi_expert_model import V12OFIExpertModel
except ImportError:
    from v12_ofi_expert_model import V12OFIExpertModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V12LSTMModel(nn.Module):
    """V12 LSTM模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, 
                 output_size: int = 1, dropout: float = 0.2):
        super(V12LSTMModel, self).__init__()
        
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
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        output = self.fc(lstm_out[:, -1, :])
        
        return output

class V12TransformerModel(nn.Module):
    """V12 Transformer模型"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, output_size: int = 1, dropout: float = 0.1):
        super(V12TransformerModel, self).__init__()
        
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
            nn.Linear(d_model // 2, output_size),
            nn.Sigmoid()
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

class V12CNNModel(nn.Module):
    """V12 CNN模型"""
    
    def __init__(self, input_size: int, output_size: int = 1, dropout: float = 0.2):
        super(V12CNNModel, self).__init__()
        
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
            nn.Linear(64, output_size),
            nn.Sigmoid()
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

class V12EnsembleAIModel:
    """
    V12 集成AI模型
    融合OFI专家模型 + 深度学习模型
    """
    
    def __init__(self, config: Dict):
        """
        初始化集成AI模型
        
        Args:
            config: 配置参数
        """
        self.config = config
        
        # 模型组件
        self.ofi_expert = V12OFIExpertModel(
            model_type=config.get('ofi_ai_fusion', {}).get('ai_models', {}).get('v9_ml_weight', 0.5)
        )
        
        # 深度学习模型
        self.lstm_model = None
        self.transformer_model = None
        self.cnn_model = None
        
        # 融合权重
        self.fusion_weights = {
            'ofi_expert': config.get('ofi_ai_fusion', {}).get('ai_models', {}).get('v9_ml_weight', 0.5),
            'lstm': config.get('ofi_ai_fusion', {}).get('ai_models', {}).get('lstm_weight', 0.2),
            'transformer': config.get('ofi_ai_fusion', {}).get('ai_models', {}).get('transformer_weight', 0.2),
            'cnn': config.get('ofi_ai_fusion', {}).get('ai_models', {}).get('cnn_weight', 0.1)
        }
        
        # 训练状态
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 统计信息
        self.stats = {
            'training_samples': 0,
            'validation_samples': 0,
            'ofi_expert_accuracy': 0.0,
            'lstm_accuracy': 0.0,
            'transformer_accuracy': 0.0,
            'cnn_accuracy': 0.0,
            'ensemble_accuracy': 0.0,
            'last_training': None
        }
        
        logger.info(f"V12集成AI模型初始化完成 - 设备: {self.device}")
        logger.info(f"融合权重: {self.fusion_weights}")
    
    def prepare_deep_learning_data(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备深度学习数据
        
        Args:
            df: 输入数据框
            sequence_length: 序列长度
            
        Returns:
            特征序列和目标值
        """
        try:
            # 选择数值特征
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in ['signal_quality', 'timestamp']]
            
            # 提取特征和目标
            features = df[feature_cols].values
            targets = df.get('signal_quality', pd.Series(0.5, index=df.index)).values
            
            # 创建序列数据
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, len(features)):
                X_sequences.append(features[i-sequence_length:i])
                y_sequences.append(targets[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            logger.info(f"深度学习数据准备完成 - 序列数: {len(X_sequences)}, 特征维度: {X_sequences.shape[2]}")
            
            return X_sequences, y_sequences
            
        except Exception as e:
            logger.error(f"准备深度学习数据失败: {e}")
            return np.array([]), np.array([])
    
    def train_deep_learning_models(self, df: pd.DataFrame):
        """
        训练深度学习模型
        
        Args:
            df: 训练数据
        """
        try:
            logger.info("开始训练V12深度学习模型...")
            
            # 准备数据
            X_sequences, y_sequences = self.prepare_deep_learning_data(df)
            
            if len(X_sequences) == 0:
                logger.warning("没有足够的序列数据，跳过深度学习模型训练")
                return
            
            # 分割训练和验证集
            split_idx = int(len(X_sequences) * 0.8)
            X_train, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
            y_train, y_val = y_sequences[:split_idx], y_sequences[split_idx:]
            
            # 转换为张量
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            
            input_size = X_sequences.shape[2]
            
            # 训练LSTM模型
            self.lstm_model = V12LSTMModel(input_size=input_size).to(self.device)
            lstm_optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
            lstm_criterion = nn.MSELoss()
            
            self._train_pytorch_model(
                self.lstm_model, lstm_optimizer, lstm_criterion,
                train_loader, X_val_tensor, y_val_tensor, "LSTM"
            )
            
            # 训练Transformer模型
            self.transformer_model = V12TransformerModel(input_size=input_size).to(self.device)
            transformer_optimizer = optim.Adam(self.transformer_model.parameters(), lr=0.001)
            transformer_criterion = nn.MSELoss()
            
            self._train_pytorch_model(
                self.transformer_model, transformer_optimizer, transformer_criterion,
                train_loader, X_val_tensor, y_val_tensor, "Transformer"
            )
            
            # 训练CNN模型
            self.cnn_model = V12CNNModel(input_size=input_size).to(self.device)
            cnn_optimizer = optim.Adam(self.cnn_model.parameters(), lr=0.001)
            cnn_criterion = nn.MSELoss()
            
            self._train_pytorch_model(
                self.cnn_model, cnn_optimizer, cnn_criterion,
                train_loader, X_val_tensor, y_val_tensor, "CNN"
            )
            
            logger.info("V12深度学习模型训练完成")
            
        except Exception as e:
            logger.error(f"训练深度学习模型失败: {e}")
    
    def _train_pytorch_model(self, model, optimizer, criterion, train_loader, X_val, y_val, model_name):
        """训练PyTorch模型"""
        try:
            model.train()
            
            for epoch in range(20):  # 训练20个epoch
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                # 验证
                if epoch % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val)
                        val_loss = criterion(val_outputs.squeeze(), y_val)
                        logger.info(f"{model_name} Epoch {epoch}: Train Loss = {total_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}")
                    model.train()
            
            # 最终评估
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_predictions = val_outputs.squeeze().cpu().numpy()
                val_targets = y_val.cpu().numpy()
                
                # 计算准确率 (使用0.5作为阈值)
                val_binary_pred = (val_predictions > 0.5).astype(int)
                val_binary_target = (val_targets > 0.5).astype(int)
                accuracy = (val_binary_pred == val_binary_target).mean()
                
                logger.info(f"{model_name} 最终验证准确率: {accuracy:.4f}")
                
                # 更新统计信息
                if model_name == "LSTM":
                    self.stats['lstm_accuracy'] = accuracy
                elif model_name == "Transformer":
                    self.stats['transformer_accuracy'] = accuracy
                elif model_name == "CNN":
                    self.stats['cnn_accuracy'] = accuracy
            
        except Exception as e:
            logger.error(f"训练{model_name}模型失败: {e}")
    
    def train_ensemble_model(self, df: pd.DataFrame, params: dict):
        """
        训练集成模型
        
        Args:
            df: 训练数据
            params: 策略参数
        """
        try:
            logger.info("开始训练V12集成AI模型...")
            
            # 训练OFI专家模型
            self.ofi_expert.train_model(df, params)
            
            # 训练深度学习模型
            self.train_deep_learning_models(df)
            
            # 更新统计信息
            self.stats['training_samples'] = len(df)
            self.stats['ofi_expert_accuracy'] = self.ofi_expert.stats['model_accuracy']
            self.stats['last_training'] = datetime.now()
            
            # 计算集成准确率
            self.stats['ensemble_accuracy'] = (
                self.fusion_weights['ofi_expert'] * self.stats['ofi_expert_accuracy'] +
                self.fusion_weights['lstm'] * self.stats['lstm_accuracy'] +
                self.fusion_weights['transformer'] * self.stats['transformer_accuracy'] +
                self.fusion_weights['cnn'] * self.stats['cnn_accuracy']
            )
            
            self.is_trained = True
            
            logger.info(f"V12集成AI模型训练完成 - 集成准确率: {self.stats['ensemble_accuracy']:.4f}")
            logger.info(f"各模型准确率:")
            logger.info(f"  OFI专家: {self.stats['ofi_expert_accuracy']:.4f}")
            logger.info(f"  LSTM: {self.stats['lstm_accuracy']:.4f}")
            logger.info(f"  Transformer: {self.stats['transformer_accuracy']:.4f}")
            logger.info(f"  CNN: {self.stats['cnn_accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"训练集成模型失败: {e}")
    
    def predict_ensemble(self, df: pd.DataFrame) -> pd.Series:
        """
        集成预测
        
        Args:
            df: 输入数据
            
        Returns:
            集成预测结果
        """
        try:
            if not self.is_trained:
                logger.warning("模型未训练，返回默认预测")
                return pd.Series(0.5, index=df.index)
            
            # OFI专家模型预测
            ofi_prediction = self.ofi_expert.predict_signal_quality(df)
            
            # 深度学习模型预测
            if self.lstm_model is not None:
                lstm_prediction = self._predict_deep_learning(df, self.lstm_model)
            else:
                lstm_prediction = pd.Series(0.5, index=df.index)
            
            if self.transformer_model is not None:
                transformer_prediction = self._predict_deep_learning(df, self.transformer_model)
            else:
                transformer_prediction = pd.Series(0.5, index=df.index)
            
            if self.cnn_model is not None:
                cnn_prediction = self._predict_deep_learning(df, self.cnn_model)
            else:
                cnn_prediction = pd.Series(0.5, index=df.index)
            
            # 融合预测
            ensemble_prediction = (
                self.fusion_weights['ofi_expert'] * ofi_prediction +
                self.fusion_weights['lstm'] * lstm_prediction +
                self.fusion_weights['transformer'] * transformer_prediction +
                self.fusion_weights['cnn'] * cnn_prediction
            )
            
            return ensemble_prediction
            
        except Exception as e:
            logger.error(f"集成预测失败: {e}")
            return pd.Series(0.5, index=df.index)
    
    def _predict_deep_learning(self, df: pd.DataFrame, model) -> pd.Series:
        """深度学习模型预测"""
        try:
            # 准备序列数据
            X_sequences, _ = self.prepare_deep_learning_data(df)
            
            if len(X_sequences) == 0:
                return pd.Series(0.5, index=df.index)
            
            # 转换为张量
            X_tensor = torch.FloatTensor(X_sequences).to(self.device)
            
            # 预测
            model.eval()
            with torch.no_grad():
                predictions = model(X_tensor)
                predictions = predictions.squeeze().cpu().numpy()
            
            # 扩展到原始数据长度
            full_predictions = np.full(len(df), 0.5)
            if len(predictions) > 0:
                full_predictions[-len(predictions):] = predictions
            
            return pd.Series(full_predictions, index=df.index)
            
        except Exception as e:
            logger.error(f"深度学习模型预测失败: {e}")
            return pd.Series(0.5, index=df.index)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'is_trained': self.is_trained,
            'device': str(self.device),
            'fusion_weights': self.fusion_weights,
            **self.stats
        }


def test_v12_ensemble_ai_model():
    """测试V12集成AI模型"""
    logger.info("开始测试V12集成AI模型...")
    
    # 配置参数
    config = {
        'ofi_ai_fusion': {
            'ai_models': {
                'v9_ml_weight': 0.5,
                'lstm_weight': 0.2,
                'transformer_weight': 0.2,
                'cnn_weight': 0.1
            }
        }
    }
    
    # 创建集成模型
    ensemble_model = V12EnsembleAIModel(config)
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    mock_data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1s'),
        'price': 3000 + np.random.randn(n_samples).cumsum() * 0.1,
        'bid1': 3000 + np.random.randn(n_samples).cumsum() * 0.1 - 0.5,
        'ask1': 3000 + np.random.randn(n_samples).cumsum() * 0.1 + 0.5,
        'bid1_size': 100 + np.random.randn(n_samples) * 10,
        'ask1_size': 100 + np.random.randn(n_samples) * 10,
        'size': 50 + np.random.randn(n_samples) * 5,
        'ofi_z': np.random.randn(n_samples) * 2,
        'cvd_z': np.random.randn(n_samples) * 2,
        'ret_1s': np.random.randn(n_samples) * 0.001,
        'atr': 1.0 + np.random.randn(n_samples) * 0.1,
        'vwap': 3000 + np.random.randn(n_samples) * 0.1,
        'signal_quality': np.random.rand(n_samples)
    }
    
    df = pd.DataFrame(mock_data)
    
    # V12参数配置
    v12_params = {
        'signals': {
            'momentum': {
                'ofi_z_min': 1.4,
                'cvd_z_min': 0.6,
                'min_signal_strength': 1.8
            }
        }
    }
    
    # 训练集成模型
    ensemble_model.train_ensemble_model(df, v12_params)
    
    # 集成预测
    predictions = ensemble_model.predict_ensemble(df)
    logger.info(f"集成预测结果统计: 均值={predictions.mean():.4f}, 标准差={predictions.std():.4f}")
    
    # 获取统计信息
    stats = ensemble_model.get_statistics()
    logger.info(f"集成模型统计信息: {stats}")
    
    logger.info("V12集成AI模型测试完成")


if __name__ == "__main__":
    test_v12_ensemble_ai_model()
