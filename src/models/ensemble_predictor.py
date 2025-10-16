import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List, Union
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 导入深度学习模型
try:
    from .lstm_predictor import LSTMPredictor, LSTMTrainer, LSTMDataProcessor
    from .cnn_recognizer import CNNPatternRecognizer, CNNTrainer, CNNDataProcessor
    from .transformer_predictor import TransformerPredictor, TransformerTrainer, TransformerDataProcessor
except ImportError:
    from lstm_predictor import LSTMPredictor, LSTMTrainer, LSTMDataProcessor
    from cnn_recognizer import CNNPatternRecognizer, CNNTrainer, CNNDataProcessor
    from transformer_predictor import TransformerPredictor, TransformerTrainer, TransformerDataProcessor

class EnsemblePredictor(nn.Module):
    """
    V10 集成学习预测器 - 融合多个深度学习模型
    架构: LSTM + CNN + Transformer + 传统ML模型
    """
    
    def __init__(self, input_dim: int = 50, sequence_length: int = 60,
                 lstm_hidden_dim: int = 128, lstm_layers: int = 3,
                 cnn_num_classes: int = 10, transformer_d_model: int = 128,
                 transformer_nhead: int = 8, transformer_layers: int = 6,
                 ensemble_method: str = 'weighted_average', dropout: float = 0.2):
        super(EnsemblePredictor, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.ensemble_method = ensemble_method
        self.dropout = dropout
        
        # 深度学习模型
        self.lstm_model = LSTMPredictor(
            input_dim=input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout
        )
        
        self.cnn_model = CNNPatternRecognizer(
            input_channels=1,
            num_classes=cnn_num_classes,
            feature_dim=input_dim,
            sequence_length=sequence_length
        )
        
        self.transformer_model = TransformerPredictor(
            input_dim=input_dim,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # 集成层
        if ensemble_method == 'weighted_average':
            # 加权平均
            self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
            self.ensemble_layer = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
        elif ensemble_method == 'attention':
            # 注意力机制
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=3,
                num_heads=1,
                dropout=dropout,
                batch_first=True
            )
            self.ensemble_layer = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        elif ensemble_method == 'stacking':
            # 堆叠集成
            self.ensemble_layer = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
        
        # 不确定性量化
        self.uncertainty_layer = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Softplus()  # 确保输出为正
        )
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
        Returns:
            prediction: 预测结果 (batch_size, 1)
            uncertainty: 不确定性 (batch_size, 1)
        """
        # 准备CNN输入 (需要调整维度)
        x_cnn = x.permute(0, 2, 1).unsqueeze(1)  # (batch_size, 1, input_dim, seq_len)
        
        # 各模型预测
        lstm_pred = self.lstm_model(x)
        cnn_pred = self.cnn_model(x_cnn)
        transformer_pred = self.transformer_model(x)
        
        # 调整CNN输出维度
        cnn_pred = cnn_pred.mean(dim=1, keepdim=True)  # 取平均作为回归输出
        
        # 组合预测
        predictions = torch.cat([lstm_pred, cnn_pred, transformer_pred], dim=1)
        # predictions: (batch_size, 3)
        
        # 集成预测
        if self.ensemble_method == 'weighted_average':
            # 加权平均
            weights = F.softmax(self.ensemble_weights, dim=0)
            weighted_pred = torch.sum(predictions * weights, dim=1, keepdim=True)
            ensemble_pred = self.ensemble_layer(predictions)
        elif self.ensemble_method == 'attention':
            # 注意力机制
            attn_out, _ = self.attention_layer(predictions.unsqueeze(1), 
                                             predictions.unsqueeze(1), 
                                             predictions.unsqueeze(1))
            ensemble_pred = self.ensemble_layer(attn_out.squeeze(1))
        elif self.ensemble_method == 'stacking':
            # 堆叠集成
            ensemble_pred = self.ensemble_layer(predictions)
        
        # 不确定性量化
        uncertainty = self.uncertainty_layer(predictions)
        
        return ensemble_pred, uncertainty
    
    def predict_signal_quality(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测信号质量
        Args:
            x: 输入特征 (batch_size, seq_len, input_dim)
        Returns:
            prediction: 信号质量预测 (batch_size, 1)
            uncertainty: 预测不确定性 (batch_size, 1)
        """
        self.eval()
        with torch.no_grad():
            prediction, uncertainty = self.forward(x)
        return prediction, uncertainty
    
    def get_model_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取各模型的预测结果
        Args:
            x: 输入特征 (batch_size, seq_len, input_dim)
        Returns:
            各模型预测结果字典
        """
        self.eval()
        with torch.no_grad():
            # 准备CNN输入
            x_cnn = x.permute(0, 2, 1).unsqueeze(1)
            
            # 各模型预测
            lstm_pred = self.lstm_model(x)
            cnn_pred = self.cnn_model(x_cnn)
            transformer_pred = self.transformer_model(x)
            
            # 调整CNN输出
            cnn_pred = cnn_pred.mean(dim=1, keepdim=True)
            
            return {
                'lstm': lstm_pred,
                'cnn': cnn_pred,
                'transformer': transformer_pred
            }

class EnsembleTrainer:
    """
    集成学习模型训练器
    """
    
    def __init__(self, model: EnsemblePredictor, learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.uncertainty_losses = []
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_uncertainty_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            
            # 前向传播
            predictions, uncertainty = self.model(batch_x)
            
            # 预测损失
            pred_loss = self.criterion(predictions, batch_y)
            
            # 不确定性损失 (鼓励模型在不确定时输出高不确定性)
            uncertainty_loss = torch.mean(uncertainty)
            
            # 总损失
            total_loss_batch = pred_loss + 0.1 * uncertainty_loss
            
            # 反向传播
            total_loss_batch.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += pred_loss.item()
            total_uncertainty_loss += uncertainty_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_uncertainty_loss = total_uncertainty_loss / num_batches
        
        self.train_losses.append(avg_loss)
        self.uncertainty_losses.append(avg_uncertainty_loss)
        
        return avg_loss, avg_uncertainty_loss
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        total_uncertainty_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions, uncertainty = self.model(batch_x)
                
                pred_loss = self.criterion(predictions, batch_y)
                uncertainty_loss = torch.mean(uncertainty)
                
                total_loss += pred_loss.item()
                total_uncertainty_loss += uncertainty_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_uncertainty_loss = total_uncertainty_loss / num_batches
        
        self.val_losses.append(avg_loss)
        
        return avg_loss, avg_uncertainty_loss
    
    def train(self, train_loader, val_loader, epochs: int = 100,
              early_stopping_patience: int = 20) -> Dict[str, Any]:
        """
        训练模型
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值
        Returns:
            训练结果字典
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"开始训练集成学习模型，共{epochs}个epoch...")
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_uncertainty_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_uncertainty_loss = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 打印进度
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, "
                      f"Train Uncertainty = {train_uncertainty_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f}, "
                      f"Val Uncertainty = {val_uncertainty_loss:.6f}, "
                      f"LR = {current_lr:.6f}")
            
            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"早停触发，在第{epoch}个epoch停止训练")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 返回训练结果
        return {
            'best_val_loss': best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'uncertainty_losses': self.uncertainty_losses,
            'learning_rates': self.learning_rates,
            'epochs_trained': epoch + 1
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'uncertainty_losses': self.uncertainty_losses,
            'learning_rates': self.learning_rates
        }, filepath)
        print(f"集成学习模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.uncertainty_losses = checkpoint['uncertainty_losses']
        self.learning_rates = checkpoint['learning_rates']
        print(f"集成学习模型已从{filepath}加载")

class EnsembleDataProcessor:
    """
    集成学习数据处理器
    """
    
    def __init__(self, sequence_length: int = 60, feature_dim: int = 50):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.scaler = None
    
    def prepare_sequences(self, data: pd.DataFrame, target_col: str = 'signal_quality') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备序列数据
        Args:
            data: 输入数据
            target_col: 目标列名
        Returns:
            X: 输入序列 (samples, seq_len, features)
            y: 目标值 (samples, 1)
        """
        # 选择特征列
        feature_cols = [col for col in data.columns if col != target_col]
        X_data = data[feature_cols].values
        y_data = data[target_col].values
        
        # 标准化
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_data)
        
        # 创建序列
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
            y_sequences.append(y_data[i])
        
        # 转换为张量
        X_tensor = torch.FloatTensor(np.array(X_sequences))
        y_tensor = torch.FloatTensor(np.array(y_sequences)).unsqueeze(1)
        
        return X_tensor, y_tensor
    
    def create_data_loaders(self, X: torch.Tensor, y: torch.Tensor,
                          batch_size: int = 32, train_ratio: float = 0.8) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        创建数据加载器
        Args:
            X: 输入数据
            y: 目标数据
            batch_size: 批次大小
            train_ratio: 训练集比例
        Returns:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        # 分割数据
        train_size = int(len(X) * train_ratio)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # 创建数据集
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader

def create_ensemble_model(input_dim: int = 50, sequence_length: int = 60,
                         ensemble_method: str = 'weighted_average') -> EnsemblePredictor:
    """
    创建集成学习模型
    Args:
        input_dim: 输入特征维度
        sequence_length: 序列长度
        ensemble_method: 集成方法 ('weighted_average', 'attention', 'stacking')
    Returns:
        集成学习模型实例
    """
    model = EnsemblePredictor(
        input_dim=input_dim,
        sequence_length=sequence_length,
        ensemble_method=ensemble_method
    )
    
    print(f"集成学习模型创建完成:")
    print(f"  输入维度: {input_dim}")
    print(f"  序列长度: {sequence_length}")
    print(f"  集成方法: {ensemble_method}")
    print(f"  总参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def train_ensemble_model(data: pd.DataFrame, target_col: str = 'signal_quality',
                        input_dim: int = 50, sequence_length: int = 60,
                        ensemble_method: str = 'weighted_average',
                        epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                        model_save_path: str = "models/ensemble_model.pth") -> Dict[str, Any]:
    """
    训练集成学习模型
    Args:
        data: 训练数据
        target_col: 目标列名
        input_dim: 输入特征维度
        sequence_length: 序列长度
        ensemble_method: 集成方法
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        model_save_path: 模型保存路径
    Returns:
        训练结果
    """
    # 创建模型
    model = create_ensemble_model(input_dim, sequence_length, ensemble_method)
    
    # 创建训练器
    trainer = EnsembleTrainer(model, learning_rate=learning_rate)
    
    # 准备数据
    processor = EnsembleDataProcessor(sequence_length, input_dim)
    X, y = processor.prepare_sequences(data, target_col)
    train_loader, val_loader = processor.create_data_loaders(X, y, batch_size)
    
    print(f"数据准备完成:")
    print(f"  总样本数: {len(X)}")
    print(f"  训练样本数: {len(train_loader.dataset)}")
    print(f"  验证样本数: {len(val_loader.dataset)}")
    
    # 训练模型
    results = trainer.train(train_loader, val_loader, epochs=epochs)
    
    # 保存模型
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    trainer.save_model(model_save_path)
    
    # 保存数据处理器
    processor_save_path = model_save_path.replace('.pth', '_processor.joblib')
    joblib.dump(processor, processor_save_path)
    
    return results

if __name__ == "__main__":
    # 测试集成学习模型
    print("测试集成学习模型...")
    
    # 创建测试数据
    batch_size, seq_len, input_dim = 32, 60, 50
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 创建模型
    model = create_ensemble_model(input_dim=input_dim, ensemble_method='weighted_average')
    
    # 前向传播测试
    prediction, uncertainty = model(x)
    print(f"输入形状: {x.shape}")
    print(f"预测形状: {prediction.shape}")
    print(f"不确定性形状: {uncertainty.shape}")
    print(f"预测范围: [{prediction.min().item():.4f}, {prediction.max().item():.4f}]")
    print(f"不确定性范围: [{uncertainty.min().item():.4f}, {uncertainty.max().item():.4f}]")
    
    # 各模型预测测试
    model_predictions = model.get_model_predictions(x)
    print(f"LSTM预测形状: {model_predictions['lstm'].shape}")
    print(f"CNN预测形状: {model_predictions['cnn'].shape}")
    print(f"Transformer预测形状: {model_predictions['transformer'].shape}")
    
    # 信号质量预测测试
    quality_pred, quality_uncertainty = model.predict_signal_quality(x)
    print(f"信号质量预测形状: {quality_pred.shape}")
    print(f"信号质量不确定性形状: {quality_uncertainty.shape}")
    
    print("集成学习模型测试完成!")
