"""
V11 Phase 3: 实时学习系统
实现在线学习、增量学习、模型监控和自适应调整
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import time
import json
import os
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OnlineLearningSystem:
    """
    V11实时学习系统
    支持在线学习、增量学习、模型监控和自适应调整
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 学习系统组件
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.metrics_history = deque(maxlen=1000)
        self.performance_monitor = PerformanceMonitor()
        self.adaptive_controller = AdaptiveController(config)
        
        # 在线学习参数
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.update_frequency = config.get('update_frequency', 100)  # 每100个样本更新一次
        self.performance_threshold = config.get('performance_threshold', 0.6)
        
        # 增量学习参数
        self.memory_size = config.get('memory_size', 1000)
        self.experience_replay = ExperienceReplay(self.memory_size)
        self.learning_scheduler = LearningScheduler(config)
        
        logger.info(f"V11实时学习系统初始化完成，设备: {self.device}")
    
    def initialize_models(self, feature_dim: int, sequence_length: int):
        """初始化深度学习模型"""
        logger.info("初始化深度学习模型...")
        
        # LSTM模型
        self.models['lstm'] = LSTMModel(
            input_dim=feature_dim,
            hidden_dim=64,
            output_dim=1,
            num_layers=2
        ).to(self.device)
        
        # Transformer模型
        self.models['transformer'] = TransformerModel(
            input_dim=feature_dim,
            d_model=64,
            nhead=8,
            num_layers=2,
            output_dim=1
        ).to(self.device)
        
        # CNN模型
        self.models['cnn'] = CNNModel(
            input_dim=feature_dim,
            sequence_length=sequence_length,
            output_dim=1
        ).to(self.device)
        
        # Ensemble模型
        self.models['ensemble'] = EnsembleModel(
            models=[self.models['lstm'], self.models['transformer'], self.models['cnn']],
            weights=[0.33, 0.33, 0.34]
        ).to(self.device)
        
        # 初始化优化器
        for name, model in self.models.items():
            self.optimizers[name] = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            self.schedulers[name] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers[name], mode='min', patience=10, factor=0.5
            )
        
        logger.info("深度学习模型初始化完成")
    
    def online_update(self, new_data: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """在线学习更新"""
        logger.info("执行在线学习更新...")
        
        # 数据预处理
        features = self._preprocess_features(new_data)
        
        # 添加到经验回放
        self.experience_replay.add(features, labels)
        
        # 检查是否需要更新
        if len(self.experience_replay) >= self.update_frequency:
            return self._perform_batch_update()
        
        return {"status": "waiting", "samples": len(self.experience_replay)}
    
    def _perform_batch_update(self) -> Dict[str, Any]:
        """执行批量更新"""
        logger.info("执行批量模型更新...")
        
        # 获取训练数据
        batch_data, batch_labels = self.experience_replay.sample(self.batch_size)
        
        # 转换为张量
        X = torch.FloatTensor(batch_data).to(self.device)
        y = torch.FloatTensor(batch_labels).to(self.device)
        
        update_results = {}
        
        # 更新每个模型
        for name, model in self.models.items():
            model.train()
            optimizer = self.optimizers[name]
            scheduler = self.schedulers[name]
            
            # 前向传播
            predictions = model(X)
            loss = nn.MSELoss()(predictions.squeeze(), y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            # 记录结果
            update_results[name] = {
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            }
        
        # 清空经验回放
        self.experience_replay.clear()
        
        logger.info("批量更新完成")
        return update_results
    
    def incremental_learning(self, new_data: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """增量学习"""
        logger.info("执行增量学习...")
        
        # 数据预处理
        features = self._preprocess_features(new_data)
        
        # 增量更新每个模型
        results = {}
        for name, model in self.models.items():
            result = self._incremental_update_model(model, features, labels, name)
            results[name] = result
        
        # 更新性能监控
        self.performance_monitor.update(results)
        
        return results
    
    def _incremental_update_model(self, model: nn.Module, features: np.ndarray, 
                                labels: np.ndarray, model_name: str) -> Dict[str, Any]:
        """增量更新单个模型"""
        model.train()
        optimizer = self.optimizers[model_name]
        
        # 转换为张量
        X = torch.FloatTensor(features).to(self.device)
        y = torch.FloatTensor(labels).to(self.device)
        
        # 前向传播
        predictions = model(X)
        
        # 确保预测和标签维度匹配
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        if y.dim() > 1:
            y = y.squeeze()
        
        # 确保形状匹配
        if predictions.shape != y.shape:
            min_len = min(len(predictions), len(y))
            predictions = predictions[:min_len]
            y = y[:min_len]
        
        loss = nn.MSELoss()(predictions, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'accuracy': self._calculate_accuracy(predictions, y),
            'model_name': model_name
        }
    
    def monitor_performance(self) -> Dict[str, Any]:
        """监控模型性能"""
        logger.info("监控模型性能...")
        
        performance_report = {}
        
        # 获取性能指标
        for name, model in self.models.items():
            model.eval()
            
            # 计算性能指标
            metrics = self.performance_monitor.get_model_metrics(name)
            performance_report[name] = metrics
        
        # 检查是否需要自适应调整
        if self.adaptive_controller.should_adjust(performance_report):
            adjustments = self.adaptive_controller.get_adjustments(performance_report)
            self._apply_adjustments(adjustments)
            performance_report['adjustments'] = adjustments
        
        return performance_report
    
    def _apply_adjustments(self, adjustments: Dict[str, Any]):
        """应用自适应调整"""
        logger.info("应用自适应调整...")
        
        for model_name, adjustment in adjustments.items():
            if model_name in self.models:
                # 调整学习率
                if 'learning_rate' in adjustment:
                    for param_group in self.optimizers[model_name].param_groups:
                        param_group['lr'] = adjustment['learning_rate']
                
                # 调整模型参数
                if 'model_params' in adjustment:
                    self._adjust_model_parameters(model_name, adjustment['model_params'])
    
    def _adjust_model_parameters(self, model_name: str, params: Dict[str, Any]):
        """调整模型参数"""
        model = self.models[model_name]
        
        # 这里可以实现更复杂的参数调整逻辑
        # 例如：调整隐藏层大小、dropout率等
        pass
    
    def _preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        """预处理特征数据"""
        # 选择数值特征
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        features = data[numeric_cols].values
        
        # 处理缺失值
        features = np.nan_to_num(features, nan=0.0)
        
        # 确保数据是3D张量 (batch_size, sequence_length, features)
        if len(features.shape) == 2:
            # 如果输入是2D，需要添加序列维度
            sequence_length = self.config.get('sequence_length', 60)
            feature_dim = self.config.get('feature_dim', 128)
            
            # 如果特征数不匹配，进行填充或截断
            if features.shape[1] != feature_dim:
                if features.shape[1] > feature_dim:
                    features = features[:, :feature_dim]
                else:
                    # 填充到指定维度
                    padding = np.zeros((features.shape[0], feature_dim - features.shape[1]))
                    features = np.concatenate([features, padding], axis=1)
            
            # 添加序列维度
            features = features.reshape(-1, 1, features.shape[1])
        
        return features
    
    def _calculate_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """计算准确率"""
        with torch.no_grad():
            pred_labels = (predictions > 0.5).float()
            correct = (pred_labels == targets).float().sum()
            accuracy = correct / len(targets)
            return accuracy.item()
    
    def save_models(self, filepath: str):
        """保存模型"""
        logger.info(f"保存模型到: {filepath}")
        
        model_state = {}
        for name, model in self.models.items():
            model_state[name] = {
                'state_dict': model.state_dict(),
                'optimizer': self.optimizers[name].state_dict(),
                'scheduler': self.schedulers[name].state_dict()
            }
        
        torch.save(model_state, filepath)
        logger.info("模型保存完成")
    
    def load_models(self, filepath: str):
        """加载模型"""
        logger.info(f"从 {filepath} 加载模型")
        
        if os.path.exists(filepath):
            model_state = torch.load(filepath, map_location=self.device)
            
            for name, state in model_state.items():
                if name in self.models:
                    self.models[name].load_state_dict(state['state_dict'])
                    self.optimizers[name].load_state_dict(state['optimizer'])
                    self.schedulers[name].load_state_dict(state['scheduler'])
            
            logger.info("模型加载完成")
        else:
            logger.warning(f"模型文件不存在: {filepath}")


class ExperienceReplay:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, features: np.ndarray, labels: np.ndarray):
        """添加经验"""
        self.buffer.append((features, labels))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """采样经验"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        features = np.array([item[0] for item in batch])
        labels = np.array([item[1] for item in batch])
        
        return features, labels
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        self.history = deque(maxlen=1000)
    
    def update(self, results: Dict[str, Any]):
        """更新性能指标"""
        timestamp = time.time()
        
        for model_name, result in results.items():
            if model_name not in self.metrics:
                self.metrics[model_name] = {
                    'loss_history': deque(maxlen=100),
                    'accuracy_history': deque(maxlen=100),
                    'last_update': timestamp
                }
            
            self.metrics[model_name]['loss_history'].append(result.get('loss', 0))
            self.metrics[model_name]['accuracy_history'].append(result.get('accuracy', 0))
            self.metrics[model_name]['last_update'] = timestamp
        
        self.history.append({
            'timestamp': timestamp,
            'results': results
        })
    
    def get_model_metrics(self, model_name: str) -> Dict[str, Any]:
        """获取模型性能指标"""
        if model_name not in self.metrics:
            return {}
        
        metrics = self.metrics[model_name]
        
        return {
            'current_loss': metrics['loss_history'][-1] if metrics['loss_history'] else 0,
            'avg_loss': np.mean(metrics['loss_history']) if metrics['loss_history'] else 0,
            'current_accuracy': metrics['accuracy_history'][-1] if metrics['accuracy_history'] else 0,
            'avg_accuracy': np.mean(metrics['accuracy_history']) if metrics['accuracy_history'] else 0,
            'last_update': metrics['last_update']
        }


class AdaptiveController:
    """自适应控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_threshold = config.get('performance_threshold', 0.6)
        self.adjustment_history = deque(maxlen=100)
    
    def should_adjust(self, performance_report: Dict[str, Any]) -> bool:
        """判断是否需要调整"""
        for model_name, metrics in performance_report.items():
            if 'avg_accuracy' in metrics:
                if metrics['avg_accuracy'] < self.performance_threshold:
                    return True
        
        return False
    
    def get_adjustments(self, performance_report: Dict[str, Any]) -> Dict[str, Any]:
        """获取调整建议"""
        adjustments = {}
        
        for model_name, metrics in performance_report.items():
            if 'avg_accuracy' in metrics and metrics['avg_accuracy'] < self.performance_threshold:
                adjustments[model_name] = {
                    'learning_rate': self._calculate_new_lr(metrics),
                    'model_params': self._calculate_model_adjustments(metrics)
                }
        
        return adjustments
    
    def _calculate_new_lr(self, metrics: Dict[str, Any]) -> float:
        """计算新的学习率"""
        current_accuracy = metrics.get('avg_accuracy', 0.5)
        
        # 如果准确率低于阈值，增加学习率
        if current_accuracy < self.performance_threshold:
            return 0.002  # 增加学习率
        else:
            return 0.001  # 保持当前学习率
    
    def _calculate_model_adjustments(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """计算模型调整参数"""
        return {
            'dropout_rate': 0.1,
            'hidden_size': 64
        }


class LearningScheduler:
    """学习调度器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.schedule_type = config.get('schedule_type', 'adaptive')
        self.learning_phases = config.get('learning_phases', [])
    
    def get_current_phase(self, iteration: int) -> Dict[str, Any]:
        """获取当前学习阶段"""
        for phase in self.learning_phases:
            if phase['start'] <= iteration <= phase['end']:
                return phase
        
        return self.learning_phases[-1] if self.learning_phases else {}


# 深度学习模型定义
class LSTMModel(nn.Module):
    """LSTM模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(self.dropout(lstm_out[:, -1, :]))
        return torch.sigmoid(output)


class TransformerModel(nn.Module):
    """Transformer模型"""
    
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, output_dim: int):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.input_projection(x)
        transformer_out = self.transformer(x)
        output = self.fc(self.dropout(transformer_out[:, -1, :]))
        return torch.sigmoid(output)


class CNNModel(nn.Module):
    """CNN模型"""
    
    def __init__(self, input_dim: int, sequence_length: int, output_dim: int):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, sequence)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        output = self.fc(self.dropout(x))
        return torch.sigmoid(output)


class EnsembleModel(nn.Module):
    """集成模型"""
    
    def __init__(self, models: List[nn.Module], weights: List[float]):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = torch.FloatTensor(weights)
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # 加权平均
        ensemble_pred = torch.stack(predictions, dim=1)
        weights = self.weights.to(x.device)
        output = torch.sum(ensemble_pred * weights, dim=1, keepdim=True)
        
        return output


if __name__ == "__main__":
    # 测试在线学习系统
    config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'update_frequency': 100,
        'performance_threshold': 0.6,
        'memory_size': 1000
    }
    
    # 创建在线学习系统
    online_system = OnlineLearningSystem(config)
    
    # 初始化模型
    online_system.initialize_models(feature_dim=128, sequence_length=60)
    
    # 模拟数据
    sample_data = pd.DataFrame(np.random.randn(100, 128))
    sample_labels = np.random.randint(0, 2, 100)
    
    # 测试在线学习
    result = online_system.online_update(sample_data, sample_labels)
    print(f"在线学习结果: {result}")
    
    # 测试性能监控
    performance = online_system.monitor_performance()
    print(f"性能监控结果: {performance}")
    
    print("V11实时学习系统测试完成！")
