import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import joblib
import os
from datetime import datetime

class CNNPatternRecognizer(nn.Module):
    """
    V10 CNN模式识别模型 - 用于识别市场模式
    架构: 3层卷积层，批归一化，池化层，全连接层
    """
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10, 
                 feature_dim: int = 50, sequence_length: int = 60):
        super(CNNPatternRecognizer, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            # 第二层卷积
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            # 第三层卷积
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            # 第四层卷积
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.2)
        )
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量 (batch_size, channels, seq_len)
        Returns:
            分类结果 (batch_size, num_classes)
        """
        # 卷积特征提取
        features = self.conv_layers(x)
        
        # 展平
        features = features.view(features.size(0), -1)
        
        # 分类
        output = self.classifier(features)
        
        return output
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征
        Args:
            x: 输入张量 (batch_size, channels, seq_len)
        Returns:
            特征向量 (batch_size, 256)
        """
        self.eval()
        with torch.no_grad():
            features = self.conv_layers(x)
            features = features.view(features.size(0), -1)
        return features
    
    def predict_pattern(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测市场模式
        Args:
            x: 输入张量 (batch_size, channels, seq_len)
        Returns:
            模式预测 (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probabilities = F.softmax(output, dim=1)
        return probabilities

class CNNTrainer:
    """
    CNN模型训练器
    """
    
    def __init__(self, model: CNNPatternRecognizer, learning_rate: float = 0.001,
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
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        accuracy = 100 * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        accuracy = 100 * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
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
        best_val_accuracy = 0.0
        patience_counter = 0
        best_model_state = None
        
        print(f"开始训练CNN模型，共{epochs}个epoch...")
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # 早停检查
            if val_loss < best_val_loss or val_acc > best_val_accuracy:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 打印进度
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Train Acc = {train_acc:.2f}%, "
                      f"Val Loss = {val_loss:.6f}, Val Acc = {val_acc:.2f}%, LR = {current_lr:.6f}")
            
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
            'best_val_accuracy': best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
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
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates
        }, filepath)
        print(f"CNN模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        self.learning_rates = checkpoint['learning_rates']
        print(f"CNN模型已从{filepath}加载")

class CNNDataProcessor:
    """
    CNN数据处理器
    """
    
    def __init__(self, sequence_length: int = 60, feature_dim: int = 50, num_classes: int = 10):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.scaler = None
        self.label_encoder = None
    
    def prepare_sequences(self, data: pd.DataFrame, target_col: str = 'pattern_type') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备序列数据
        Args:
            data: 输入数据
            target_col: 目标列名
        Returns:
            X: 输入序列 (samples, channels, seq_len)
            y: 目标值 (samples,)
        """
        # 选择特征列
        feature_cols = [col for col in data.columns if col != target_col]
        X_data = data[feature_cols].values
        y_data = data[target_col].values
        
        # 标准化
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_data)
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_data)
        
        # 创建序列
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
            y_sequences.append(y_encoded[i])
        
        # 转换为张量
        X_tensor = torch.FloatTensor(np.array(X_sequences))
        y_tensor = torch.LongTensor(np.array(y_sequences))
        
        # 调整维度 (samples, channels, seq_len)
        X_tensor = X_tensor.permute(0, 2, 1)  # (samples, features, seq_len)
        X_tensor = X_tensor.unsqueeze(1)  # (samples, 1, features, seq_len)
        
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

def create_cnn_model(input_channels: int = 1, num_classes: int = 10,
                    feature_dim: int = 50, sequence_length: int = 60) -> CNNPatternRecognizer:
    """
    创建CNN模型
    Args:
        input_channels: 输入通道数
        num_classes: 类别数
        feature_dim: 特征维度
        sequence_length: 序列长度
    Returns:
        CNN模型实例
    """
    model = CNNPatternRecognizer(
        input_channels=input_channels,
        num_classes=num_classes,
        feature_dim=feature_dim,
        sequence_length=sequence_length
    )
    
    print(f"CNN模型创建完成:")
    print(f"  输入通道数: {input_channels}")
    print(f"  类别数: {num_classes}")
    print(f"  特征维度: {feature_dim}")
    print(f"  序列长度: {sequence_length}")
    print(f"  总参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def train_cnn_model(data: pd.DataFrame, target_col: str = 'pattern_type',
                   input_channels: int = 1, num_classes: int = 10,
                   feature_dim: int = 50, sequence_length: int = 60,
                   epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                   model_save_path: str = "models/cnn_model.pth") -> Dict[str, Any]:
    """
    训练CNN模型
    Args:
        data: 训练数据
        target_col: 目标列名
        input_channels: 输入通道数
        num_classes: 类别数
        feature_dim: 特征维度
        sequence_length: 序列长度
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        model_save_path: 模型保存路径
    Returns:
        训练结果
    """
    # 创建模型
    model = create_cnn_model(input_channels, num_classes, feature_dim, sequence_length)
    
    # 创建训练器
    trainer = CNNTrainer(model, learning_rate=learning_rate)
    
    # 准备数据
    processor = CNNDataProcessor(sequence_length, feature_dim, num_classes)
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
    # 测试CNN模型
    print("测试CNN模型...")
    
    # 创建测试数据
    batch_size, channels, seq_len = 32, 1, 60
    x = torch.randn(batch_size, channels, seq_len)
    
    # 创建模型
    model = create_cnn_model(input_channels=channels, num_classes=10)
    
    # 前向传播测试
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 特征提取测试
    features = model.extract_features(x)
    print(f"特征形状: {features.shape}")
    
    # 模式预测测试
    probabilities = model.predict_pattern(x)
    print(f"概率形状: {probabilities.shape}")
    print(f"概率和: {probabilities.sum(dim=1)[:5]}")
    
    print("CNN模型测试完成!")
