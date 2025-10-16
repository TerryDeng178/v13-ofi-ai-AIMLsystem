"""
V12 集成AI模型 - 最终修复版本
完全解决Tensor兼容性和维度不匹配问题
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
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
    """V12 LSTM模型 - 最终修复版本"""
    
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
        # 确保输入是3D张量 (batch_size, sequence_length, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加序列维度
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        output = self.fc(lstm_out[:, -1, :])
        
        return output

class V12TransformerModel(nn.Module):
    """V12 Transformer模型 - 最终修复版本"""
    
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
        # 确保输入是3D张量
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加序列维度
        
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)
        
        # 位置编码
        if seq_len <= self.pos_encoding.size(0):
            x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 取最后一个时间步的输出
        output = self.output_layer(x[:, -1, :])
        
        return output

class V12CNNModel(nn.Module):
    """V12 CNN模型 - 最终修复版本"""
    
    def __init__(self, input_size: int, sequence_length: int = 60, output_size: int = 1):
        super(V12CNNModel, self).__init__()
        
        self.sequence_length = sequence_length
        
        # 卷积层 - 使用正确的输入通道数
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 确保输入是3D张量 (batch_size, sequence_length, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加序列维度
        
        # 转换为卷积格式 (batch_size, features, sequence_length)
        x = x.transpose(1, 2)
        
        # 卷积层
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # 全局平均池化
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        output = self.fc(x)
        
        return output

class V12EnsembleAIModel:
    """V12集成AI模型 - 最终修复版本"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化集成AI模型
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型组件
        self.ofi_expert = V12OFIExpertModel()
        self.lstm_model = None
        self.transformer_model = None
        self.cnn_model = None
        
        # 融合权重
        self.fusion_weights = self.config.get('fusion_weights', {
            'ofi_expert': 0.5,
            'lstm': 0.2,
            'transformer': 0.2,
            'cnn': 0.1
        })
        
        # 训练状态
        self.is_trained = False
        self.stats = {}
        
        # 动态输入尺寸
        self.dynamic_input_size = None
        
        logger.info(f"V12集成AI模型初始化完成 - 设备: {self.device}")
        logger.info(f"融合权重: {self.fusion_weights}")
    
    def _ensure_numpy_array(self, data: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]) -> np.ndarray:
        """确保数据是numpy数组格式"""
        try:
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            elif isinstance(data, pd.DataFrame):
                return data.select_dtypes(include=[np.number]).values
            elif isinstance(data, pd.Series):
                return data.values
            elif isinstance(data, np.ndarray):
                return data
            else:
                # 尝试转换为numpy数组
                return np.array(data)
        except Exception as e:
            logger.error(f"数据格式转换失败: {e}")
            return np.array([])
    
    def _prepare_features(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """准备特征数据，统一格式"""
        try:
            # 转换为numpy数组
            features = self._ensure_numpy_array(data)
            
            if features.size == 0:
                logger.warning("特征数据为空")
                return np.array([])
            
            # 确保是2D数组
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # 处理NaN值
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return features
            
        except Exception as e:
            logger.error(f"特征准备失败: {e}")
            return np.array([])
    
    def prepare_deep_learning_data(self, data: Union[np.ndarray, pd.DataFrame], 
                                 sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备深度学习数据 - 最终修复版本
        
        Args:
            data: 输入数据
            sequence_length: 序列长度
            
        Returns:
            特征序列和目标值
        """
        try:
            # 准备特征数据
            features = self._prepare_features(data)
            
            if features.size == 0:
                logger.warning("没有有效的特征数据")
                return np.array([]), np.array([])
            
            # 如果数据量不足，返回空数组
            if len(features) < sequence_length:
                logger.warning(f"数据量不足: {len(features)} < {sequence_length}")
                return np.array([]), np.array([])
            
            # 创建序列数据
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, len(features)):
                X_sequences.append(features[i-sequence_length:i])
                # 使用简单的目标值（基于价格变化或其他指标）
                if features.shape[1] > 0:  # 确保有特征列
                    # 使用第一个特征（通常是价格）的变化作为目标
                    price_change = (features[i, 0] - features[i-1, 0]) / (features[i-1, 0] + 1e-8)
                    y_sequences.append(max(0, min(1, 0.5 + price_change * 0.1)))  # 归一化到[0,1]
                else:
                    y_sequences.append(0.5)
            
            if len(X_sequences) == 0:
                logger.warning("没有生成有效的序列数据")
                return np.array([]), np.array([])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            # 记录输入尺寸
            self.dynamic_input_size = X_sequences.shape[2]
            
            logger.info(f"深度学习数据准备完成 - 序列数: {len(X_sequences)}, 特征维度: {X_sequences.shape[2]}")
            
            return X_sequences, y_sequences
            
        except Exception as e:
            logger.error(f"准备深度学习数据失败: {e}")
            return np.array([]), np.array([])
    
    def train_deep_learning_models(self, data: Union[np.ndarray, pd.DataFrame]):
        """
        训练深度学习模型 - 最终修复版本
        
        Args:
            data: 训练数据
        """
        try:
            logger.info("开始训练深度学习模型...")
            
            # 准备数据
            X_sequences, y_sequences = self.prepare_deep_learning_data(data)
            
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
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            input_size = X_sequences.shape[2]  # 使用实际的输入尺寸
            sequence_length = X_sequences.shape[1]
            
            # 训练LSTM模型
            logger.info(f"训练LSTM模型 - 输入尺寸: {input_size}")
            self.lstm_model = V12LSTMModel(input_size=input_size).to(self.device)
            lstm_optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
            lstm_criterion = nn.MSELoss()
            
            self._train_pytorch_model(
                self.lstm_model, lstm_optimizer, lstm_criterion,
                train_loader, X_val_tensor, y_val_tensor, "LSTM"
            )
            
            # 训练Transformer模型
            logger.info(f"训练Transformer模型 - 输入尺寸: {input_size}")
            self.transformer_model = V12TransformerModel(input_size=input_size).to(self.device)
            transformer_optimizer = optim.Adam(self.transformer_model.parameters(), lr=0.001)
            transformer_criterion = nn.MSELoss()
            
            self._train_pytorch_model(
                self.transformer_model, transformer_optimizer, transformer_criterion,
                train_loader, X_val_tensor, y_val_tensor, "Transformer"
            )
            
            # 训练CNN模型
            logger.info(f"训练CNN模型 - 输入尺寸: {input_size}, 序列长度: {sequence_length}")
            self.cnn_model = V12CNNModel(input_size=input_size, sequence_length=sequence_length).to(self.device)
            cnn_optimizer = optim.Adam(self.cnn_model.parameters(), lr=0.001)
            cnn_criterion = nn.MSELoss()
            
            self._train_pytorch_model(
                self.cnn_model, cnn_optimizer, cnn_criterion,
                train_loader, X_val_tensor, y_val_tensor, "CNN"
            )
            
            self.is_trained = True
            logger.info("深度学习模型训练完成")
            
        except Exception as e:
            logger.error(f"训练深度学习模型失败: {e}")
    
    def _train_pytorch_model(self, model, optimizer, criterion, train_loader, 
                           X_val, y_val, model_name: str, epochs: int = 10):
        """训练PyTorch模型"""
        try:
            model.train()
            
            for epoch in range(epochs):
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
                        logger.info(f"{model_name} Epoch {epoch}: Train Loss={total_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}")
                    model.train()
            
            logger.info(f"{model_name}模型训练完成")
            
        except Exception as e:
            logger.error(f"{model_name}模型训练失败: {e}")
    
    def predict_ensemble(self, data: Union[np.ndarray, pd.DataFrame, List[float]]) -> Union[float, np.ndarray]:
        """
        集成预测 - 最终修复版本
        
        Args:
            data: 输入数据
            
        Returns:
            集成预测结果
        """
        try:
            if not self.is_trained:
                logger.warning("模型未训练，使用默认预测")
                return 0.5
            
            # 准备输入数据
            if isinstance(data, list):
                data = np.array(data)
            
            # 确保数据是DataFrame格式用于OFI专家模型
            if isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                # 创建DataFrame
                feature_names = [f'feature_{i}' for i in range(data.shape[1])]
                df = pd.DataFrame(data, columns=feature_names)
            else:
                df = data
            
            # OFI专家模型预测
            try:
                ofi_prediction = self.ofi_expert.predict_signal_quality(df)
                if isinstance(ofi_prediction, pd.Series):
                    ofi_prediction = ofi_prediction.iloc[0] if len(ofi_prediction) > 0 else 0.5
                elif isinstance(ofi_prediction, (list, np.ndarray)):
                    ofi_prediction = ofi_prediction[0] if len(ofi_prediction) > 0 else 0.5
            except Exception as e:
                logger.warning(f"OFI专家模型预测失败: {e}")
                ofi_prediction = 0.5
            
            # 深度学习模型预测
            lstm_prediction = self._predict_deep_learning(data, self.lstm_model)
            transformer_prediction = self._predict_deep_learning(data, self.transformer_model)
            cnn_prediction = self._predict_deep_learning(data, self.cnn_model)
            
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
            return 0.5
    
    def _predict_deep_learning(self, data: Union[np.ndarray, pd.DataFrame, List[float]], model) -> float:
        """深度学习模型预测 - 最终修复版本"""
        try:
            if model is None:
                return 0.5
            
            # 准备输入数据
            features = self._prepare_features(data)
            
            if features.size == 0:
                return 0.5
            
            # 确保有足够的序列长度
            sequence_length = 60
            if len(features) < sequence_length:
                # 如果数据不足，重复最后一个值
                padding_needed = sequence_length - len(features)
                padding = np.tile(features[-1:], (padding_needed, 1))
                features = np.vstack([features, padding])
            
            # 取最后的序列
            X_sequence = features[-sequence_length:].reshape(1, sequence_length, -1)
            
            # 转换为张量
            X_tensor = torch.FloatTensor(X_sequence).to(self.device)
            
            # 预测
            model.eval()
            with torch.no_grad():
                prediction = model(X_tensor)
                prediction = prediction.squeeze().cpu().item()
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"深度学习模型预测失败: {e}")
            return 0.5
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'is_trained': self.is_trained,
            'device': str(self.device),
            'fusion_weights': self.fusion_weights,
            'dynamic_input_size': self.dynamic_input_size,
            'models_status': {
                'lstm': self.lstm_model is not None,
                'transformer': self.transformer_model is not None,
                'cnn': self.cnn_model is not None
            },
            **self.stats
        }


def test_v12_ensemble_ai_model_final():
    """测试V12集成AI模型 - 最终修复版本"""
    logger.info("开始测试V12集成AI模型最终修复版本...")
    
    # 配置参数
    config = {
        'lstm_sequence_length': 60,
        'transformer_sequence_length': 60,
        'cnn_lookback': 20,
        'ensemble_weights': [0.3, 0.3, 0.4],
        'fusion_weights': {
            'ofi_expert': 0.4,
            'lstm': 0.25,
            'transformer': 0.25,
            'cnn': 0.1
        }
    }
    
    # 创建集成模型
    ensemble_model = V12EnsembleAIModel(config)
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 31  # 使用31个特征
    
    # 创建模拟数据
    data = np.random.randn(n_samples, n_features)
    
    # 训练模型
    ensemble_model.train_deep_learning_models(data)
    
    # 测试预测
    test_data = np.random.randn(1, n_features)
    prediction = ensemble_model.predict_ensemble(test_data)
    
    logger.info(f"集成预测结果: {prediction}")
    
    # 获取统计信息
    stats = ensemble_model.get_statistics()
    logger.info(f"集成模型统计信息: {stats}")
    
    logger.info("V12集成AI模型最终修复版本测试完成")


if __name__ == "__main__":
    test_v12_ensemble_ai_model_final()
