import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import math
import joblib
import os
from datetime import datetime

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        Args:
            x: 输入张量 (seq_len, batch_size, d_model)
        Returns:
            带位置编码的张量
        """
        return x + self.pe[:x.size(0), :]

class TransformerPredictor(nn.Module):
    """
    V10 Transformer注意力机制模型 - 用于序列建模和信号预测
    架构: 多头注意力，位置编码，前馈网络，层归一化
    """
    
    def __init__(self, input_dim: int = 50, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 512, dropout: float = 0.1,
                 max_len: int = 5000):
        super(TransformerPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 注意力池化
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            src_mask: 源掩码 (可选)
        Returns:
            预测结果 (batch_size, 1)
        """
        batch_size, seq_len, _ = x.size()
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer编码
        transformer_out = self.transformer(x, src_mask=src_mask)
        # transformer_out: (batch_size, seq_len, d_model)
        
        # 层归一化
        transformer_out = self.layer_norm(transformer_out)
        
        # 注意力池化
        # 使用查询向量进行注意力池化
        query = torch.mean(transformer_out, dim=1, keepdim=True)  # (batch_size, 1, d_model)
        attn_out, attn_weights = self.attention_pooling(query, transformer_out, transformer_out)
        # attn_out: (batch_size, 1, d_model)
        
        # 输出层
        output = self.output_layer(attn_out.squeeze(1))  # (batch_size, 1)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
        Returns:
            注意力权重 (batch_size, seq_len, seq_len)
        """
        self.eval()
        with torch.no_grad():
            batch_size, seq_len, _ = x.size()
            
            # 输入投影
            x = self.input_projection(x)
            
            # 位置编码
            x = x.transpose(0, 1)
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)
            
            # Transformer编码
            transformer_out = self.transformer(x)
            transformer_out = self.layer_norm(transformer_out)
            
            # 注意力池化
            query = torch.mean(transformer_out, dim=1, keepdim=True)
            _, attn_weights = self.attention_pooling(query, transformer_out, transformer_out)
            
        return attn_weights.squeeze(1)  # (batch_size, seq_len)
    
    def predict_signal_quality(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测信号质量
        Args:
            x: 输入特征 (batch_size, seq_len, input_dim)
        Returns:
            信号质量预测 (batch_size, 1)
        """
        self.eval()
        with torch.no_grad():
            prediction = self.forward(x)
        return prediction

class TransformerTrainer:
    """
    Transformer模型训练器
    """
    
    def __init__(self, model: TransformerPredictor, learning_rate: float = 0.001,
                 weight_decay: float = 1e-5, warmup_steps: int = 1000):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # 学习率调度器 (带预热)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: self.get_lr_lambda(step, warmup_steps)
        )
        
        self.criterion = nn.MSELoss()
        self.warmup_steps = warmup_steps
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def get_lr_lambda(self, step: int, warmup_steps: int) -> float:
        """学习率调度函数"""
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (10000 - warmup_steps)))
    
    def train_epoch(self, train_loader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            
            # 前向传播
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
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
        
        print(f"开始训练Transformer模型，共{epochs}个epoch...")
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # 记录学习率
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
                      f"Val Loss = {val_loss:.6f}, LR = {current_lr:.6f}")
            
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
            'learning_rates': self.learning_rates,
            'epochs_trained': epoch + 1
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }, filepath)
        print(f"Transformer模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        print(f"Transformer模型已从{filepath}加载")

class TransformerDataProcessor:
    """
    Transformer数据处理器
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

def create_transformer_model(input_dim: int = 50, d_model: int = 128, nhead: int = 8,
                           num_layers: int = 6, dim_feedforward: int = 512) -> TransformerPredictor:
    """
    创建Transformer模型
    Args:
        input_dim: 输入特征维度
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: 编码器层数
        dim_feedforward: 前馈网络维度
    Returns:
        Transformer模型实例
    """
    model = TransformerPredictor(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.1
    )
    
    print(f"Transformer模型创建完成:")
    print(f"  输入维度: {input_dim}")
    print(f"  模型维度: {d_model}")
    print(f"  注意力头数: {nhead}")
    print(f"  编码器层数: {num_layers}")
    print(f"  前馈网络维度: {dim_feedforward}")
    print(f"  总参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def train_transformer_model(data: pd.DataFrame, target_col: str = 'signal_quality',
                          input_dim: int = 50, d_model: int = 128, nhead: int = 8,
                          num_layers: int = 6, dim_feedforward: int = 512,
                          epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                          model_save_path: str = "models/transformer_model.pth") -> Dict[str, Any]:
    """
    训练Transformer模型
    Args:
        data: 训练数据
        target_col: 目标列名
        input_dim: 输入特征维度
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: 编码器层数
        dim_feedforward: 前馈网络维度
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        model_save_path: 模型保存路径
    Returns:
        训练结果
    """
    # 创建模型
    model = create_transformer_model(input_dim, d_model, nhead, num_layers, dim_feedforward)
    
    # 创建训练器
    trainer = TransformerTrainer(model, learning_rate=learning_rate)
    
    # 准备数据
    processor = TransformerDataProcessor(sequence_length=60, feature_dim=input_dim)
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
    # 测试Transformer模型
    print("测试Transformer模型...")
    
    # 创建测试数据
    batch_size, seq_len, input_dim = 32, 60, 50
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 创建模型
    model = create_transformer_model(input_dim=input_dim)
    
    # 前向传播测试
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 注意力权重测试
    attn_weights = model.get_attention_weights(x)
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"注意力权重和: {attn_weights.sum(dim=1)[:5]}")
    
    # 信号质量预测测试
    predictions = model.predict_signal_quality(x)
    print(f"预测形状: {predictions.shape}")
    print(f"预测范围: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    
    print("Transformer模型测试完成!")
