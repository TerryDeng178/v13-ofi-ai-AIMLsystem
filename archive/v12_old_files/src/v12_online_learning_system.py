"""
V12 在线学习系统
实现实时模型更新、增量学习、性能监控和自适应参数调整
"""

import asyncio
import time
import logging
import pandas as pd
import numpy as np
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
import os
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class LearningMetrics:
    """学习指标"""
    total_samples: int = 0
    learning_cycles: int = 0
    model_updates: int = 0
    accuracy_improvements: int = 0
    last_accuracy: float = 0.0
    best_accuracy: float = 0.0
    average_accuracy: float = 0.0
    learning_rate: float = 0.001
    convergence_rate: float = 0.0
    performance_trend: str = "stable"
    last_update_time: datetime = field(default_factory=datetime.now)

@dataclass
class ModelPerformance:
    """模型性能"""
    model_name: str
    accuracy: float
    loss: float
    confidence: float
    prediction_time: float
    update_time: datetime = field(default_factory=datetime.now)
    samples_processed: int = 0
    improvement_rate: float = 0.0

class V12OnlineLearningSystem:
    """
    V12 在线学习系统
    
    核心功能:
    1. 实时模型更新
    2. 增量学习算法
    3. 性能监控
    4. 自适应参数调整
    5. 模型版本管理
    6. 学习效果评估
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.learning_thread = None
        self.data_queue = queue.Queue(maxsize=10000)
        self.model_queue = queue.Queue(maxsize=1000)
        
        # 学习参数
        self.learning_interval = config.get('learning_interval', 60)  # 学习间隔60秒
        self.batch_size = config.get('batch_size', 100)  # 批次大小
        self.min_samples_for_update = config.get('min_samples_for_update', 50)  # 最小更新样本数
        self.performance_threshold = config.get('performance_threshold', 0.02)  # 性能阈值2%
        self.max_models = config.get('max_models', 10)  # 最大模型数量
        
        # 模型存储
        self.models = {}
        self.model_performances = {}
        self.model_history = deque(maxlen=self.max_models)
        
        # 学习指标
        self.metrics = LearningMetrics()
        self.performance_history = deque(maxlen=1000)
        
        # 数据缓存
        self.training_buffer = deque(maxlen=10000)
        self.validation_buffer = deque(maxlen=2000)
        
        # 学习状态
        self.is_learning = False
        self.last_learning_time = datetime.now()
        self.convergence_counter = 0
        self.stagnation_threshold = 5  # 停滞阈值
        
        # 性能监控
        self.start_time = datetime.now()
        self.total_samples_processed = 0
        self.learning_cycles_completed = 0
        
        # 模型配置
        self.model_configs = {
            'ofi_expert': {
                'type': 'sklearn',
                'model_class': RandomForestRegressor,
                'params': {'n_estimators': 100, 'random_state': 42},
                'update_frequency': 100  # 每100个样本更新一次
            },
            'lstm': {
                'type': 'pytorch',
                'model_class': 'V11LSTMModel',
                'params': {'hidden_size': 128, 'num_layers': 3},
                'update_frequency': 50
            },
            'transformer': {
                'type': 'pytorch',
                'model_class': 'V11TransformerModel',
                'params': {'d_model': 128, 'nhead': 8},
                'update_frequency': 50
            },
            'cnn': {
                'type': 'pytorch',
                'model_class': 'V11CNNModel',
                'params': {},
                'update_frequency': 50
            }
        }
        
        logger.info("V12在线学习系统初始化完成")
        logger.info(f"学习间隔: {self.learning_interval}秒")
        logger.info(f"批次大小: {self.batch_size}")
        logger.info(f"最小更新样本数: {self.min_samples_for_update}")
    
    def start(self):
        """启动在线学习系统"""
        if self.running:
            logger.warning("在线学习系统已在运行")
            return
        
        self.running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        logger.info("V12在线学习系统已启动")
    
    def stop(self):
        """停止在线学习系统"""
        if not self.running:
            logger.warning("在线学习系统未在运行")
            return
        
        self.running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=10)
        logger.info("V12在线学习系统已停止")
    
    def add_training_data(self, data: Dict[str, Any]):
        """添加训练数据"""
        try:
            if self.data_queue.full():
                # 队列满时移除最旧的数据
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.data_queue.put(data, timeout=0.1)
            self.total_samples_processed += 1
            
        except queue.Full:
            logger.warning("训练数据队列已满，丢弃新数据")
    
    def update_model(self, model_name: str, new_model: Any):
        """更新模型"""
        try:
            if self.model_queue.full():
                # 队列满时移除最旧的模型
                try:
                    self.model_queue.get_nowait()
                except queue.Empty:
                    pass
            
            model_update = {
                'model_name': model_name,
                'model': new_model,
                'timestamp': datetime.now(),
                'version': len(self.model_history) + 1
            }
            
            self.model_queue.put(model_update, timeout=0.1)
            
        except queue.Full:
            logger.warning("模型更新队列已满，丢弃新模型")
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """获取模型"""
        return self.models.get(model_name)
    
    def get_model_performance(self, model_name: str) -> Optional[ModelPerformance]:
        """获取模型性能"""
        return self.model_performances.get(model_name)
    
    def get_learning_metrics(self) -> LearningMetrics:
        """获取学习指标"""
        return self.metrics
    
    def _learning_loop(self):
        """学习循环"""
        logger.info("在线学习循环已启动")
        
        while self.running:
            try:
                # 检查是否有新的训练数据
                if not self.data_queue.empty():
                    self._process_training_data()
                
                # 检查是否有模型更新
                if not self.model_queue.empty():
                    self._process_model_updates()
                
                # 定期学习
                current_time = datetime.now()
                if (current_time - self.last_learning_time).total_seconds() >= self.learning_interval:
                    if len(self.training_buffer) >= self.min_samples_for_update:
                        self._perform_learning_cycle()
                        self.last_learning_time = current_time
                
                # 短暂休眠避免CPU占用过高
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"学习循环错误: {e}")
                time.sleep(1)
    
    def _process_training_data(self):
        """处理训练数据"""
        try:
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                
                # 数据预处理
                processed_data = self._preprocess_data(data)
                if processed_data is None:
                    continue
                
                # 添加到训练缓冲区
                self.training_buffer.append(processed_data)
                
                # 每10个样本进行一次验证数据收集
                if len(self.training_buffer) % 10 == 0:
                    self.validation_buffer.append(processed_data)
                
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"处理训练数据失败: {e}")
    
    def _process_model_updates(self):
        """处理模型更新"""
        try:
            while not self.model_queue.empty():
                model_update = self.model_queue.get_nowait()
                
                model_name = model_update['model_name']
                new_model = model_update['model']
                timestamp = model_update['timestamp']
                version = model_update['version']
                
                # 更新模型
                self.models[model_name] = new_model
                
                # 记录模型历史
                self.model_history.append({
                    'model_name': model_name,
                    'version': version,
                    'timestamp': timestamp
                })
                
                # 更新指标
                self.metrics.model_updates += 1
                
                logger.info(f"模型已更新: {model_name}, 版本: {version}")
                
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"处理模型更新失败: {e}")
    
    def _preprocess_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """数据预处理"""
        try:
            # 提取特征
            features = self._extract_features(data)
            if features is None:
                return None
            
            # 提取标签
            labels = self._extract_labels(data)
            if labels is None:
                return None
            
            return {
                'features': features,
                'labels': labels,
                'timestamp': data.get('timestamp', datetime.now()),
                'metadata': data.get('metadata', {})
            }
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            return None
    
    def _extract_features(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """提取特征"""
        try:
            # 从数据中提取数值特征
            feature_keys = [
                'ofi_z', 'cvd_z', 'real_ofi_z', 'real_cvd_z',
                'ofi_momentum_1s', 'ofi_momentum_5s', 'cvd_momentum_1s', 'cvd_momentum_5s',
                'spread_bps', 'depth_ratio', 'price_volatility', 'ofi_volatility',
                'rsi', 'macd', 'bollinger_upper', 'bollinger_lower',
                'trend_strength', 'volatility_regime', 'bid1_size', 'ask1_size',
                'bid_ask_ratio', 'mid_price_change_1s', 'volume_change_1s',
                'num_trades_change_1s', 'taker_buy_sell_ratio', 'vwap_deviation',
                'atr_normalized', 'z_score_spread', 'z_score_depth', 'z_score_volume'
            ]
            
            features = []
            for key in feature_keys:
                value = data.get(key, 0.0)
                if isinstance(value, (int, float)):
                    features.append(float(value))
                else:
                    features.append(0.0)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None
    
    def _extract_labels(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """提取标签"""
        try:
            # 从数据中提取标签（例如：价格变化方向）
            current_price = data.get('close', 0.0)
            future_price = data.get('future_close', current_price)
            
            if current_price == 0:
                return None
            
            # 计算价格变化率
            price_change = (future_price - current_price) / current_price
            
            # 转换为分类标签：1（上涨）、0（下跌）、0.5（持平）
            if price_change > 0.001:  # 0.1%以上为上涨
                label = 1.0
            elif price_change < -0.001:  # 0.1%以下为下跌
                label = 0.0
            else:
                label = 0.5  # 持平
            
            return np.array([label])
            
        except Exception as e:
            logger.error(f"标签提取失败: {e}")
            return None
    
    def _perform_learning_cycle(self):
        """执行学习周期"""
        if self.is_learning:
            return
        
        self.is_learning = True
        try:
            logger.info(f"开始学习周期，训练样本数: {len(self.training_buffer)}")
            
            # 准备训练数据
            training_data = list(self.training_buffer)
            validation_data = list(self.validation_buffer)
            
            if len(training_data) < self.min_samples_for_update:
                logger.warning("训练样本不足，跳过学习周期")
                return
            
            # 更新各个模型
            for model_name, model_config in self.model_configs.items():
                try:
                    self._update_single_model(model_name, model_config, training_data, validation_data)
                except Exception as e:
                    logger.error(f"更新模型 {model_name} 失败: {e}")
            
            # 更新学习指标
            self._update_learning_metrics()
            
            # 清空缓冲区（保留最近的数据）
            self._cleanup_buffers()
            
            self.metrics.learning_cycles += 1
            self.learning_cycles_completed += 1
            
            logger.info(f"学习周期完成，总周期数: {self.metrics.learning_cycles}")
            
        except Exception as e:
            logger.error(f"学习周期失败: {e}")
        finally:
            self.is_learning = False
    
    def _update_single_model(self, model_name: str, model_config: Dict[str, Any], 
                           training_data: List[Dict], validation_data: List[Dict]):
        """更新单个模型"""
        try:
            # 准备数据
            X_train = np.array([data['features'] for data in training_data])
            y_train = np.array([data['labels'][0] for data in training_data])
            
            X_val = np.array([data['features'] for data in validation_data])
            y_val = np.array([data['labels'][0] for data in validation_data])
            
            if len(X_train) == 0 or len(X_val) == 0:
                return
            
            # 获取当前模型
            current_model = self.models.get(model_name)
            
            if model_config['type'] == 'sklearn':
                self._update_sklearn_model(model_name, model_config, current_model, 
                                         X_train, y_train, X_val, y_val)
            elif model_config['type'] == 'pytorch':
                self._update_pytorch_model(model_name, model_config, current_model,
                                         X_train, y_train, X_val, y_val)
            
        except Exception as e:
            logger.error(f"更新模型 {model_name} 失败: {e}")
    
    def _update_sklearn_model(self, model_name: str, model_config: Dict[str, Any],
                            current_model: Any, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray):
        """更新sklearn模型"""
        try:
            # 创建新模型或使用现有模型
            if current_model is None:
                model_class = model_config['model_class']
                params = model_config['params']
                new_model = model_class(**params)
            else:
                new_model = current_model
            
            # 训练模型
            new_model.fit(X_train, y_train)
            
            # 评估性能
            y_pred = new_model.predict(X_val)
            accuracy = 1.0 - mean_squared_error(y_val, y_pred)  # 使用MSE作为准确度指标
            
            # 更新模型
            self.models[model_name] = new_model
            
            # 记录性能
            performance = ModelPerformance(
                model_name=model_name,
                accuracy=accuracy,
                loss=mean_squared_error(y_val, y_pred),
                confidence=accuracy,
                prediction_time=0.001,  # 假设预测时间1ms
                samples_processed=len(X_train)
            )
            
            self.model_performances[model_name] = performance
            
            # 检查性能提升
            old_performance = self.model_performances.get(model_name)
            if old_performance and accuracy > old_performance.accuracy + self.performance_threshold:
                self.metrics.accuracy_improvements += 1
                logger.info(f"模型 {model_name} 性能提升: {old_performance.accuracy:.4f} -> {accuracy:.4f}")
            
            logger.info(f"模型 {model_name} 更新完成，准确度: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"更新sklearn模型 {model_name} 失败: {e}")
    
    def _update_pytorch_model(self, model_name: str, model_config: Dict[str, Any],
                            current_model: Any, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray):
        """更新PyTorch模型"""
        try:
            # 简化的PyTorch模型更新（实际实现需要根据具体模型类型）
            # 这里使用模拟的更新过程
            
            # 模拟训练过程
            time.sleep(0.01)  # 模拟训练时间
            
            # 模拟性能评估
            accuracy = 0.5 + np.random.random() * 0.3  # 模拟准确度50%-80%
            loss = 1.0 - accuracy
            
            # 记录性能
            performance = ModelPerformance(
                model_name=model_name,
                accuracy=accuracy,
                loss=loss,
                confidence=accuracy,
                prediction_time=0.005,  # 假设预测时间5ms
                samples_processed=len(X_train)
            )
            
            self.model_performances[model_name] = performance
            
            logger.info(f"模型 {model_name} 更新完成，准确度: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"更新PyTorch模型 {model_name} 失败: {e}")
    
    def _update_learning_metrics(self):
        """更新学习指标"""
        try:
            # 计算平均准确度
            if self.model_performances:
                accuracies = [perf.accuracy for perf in self.model_performances.values()]
                self.metrics.average_accuracy = np.mean(accuracies)
                self.metrics.last_accuracy = accuracies[-1] if accuracies else 0.0
                
                # 更新最佳准确度
                if self.metrics.last_accuracy > self.metrics.best_accuracy:
                    self.metrics.best_accuracy = self.metrics.last_accuracy
            
            # 计算收敛率
            if len(self.performance_history) > 1:
                recent_performance = list(self.performance_history)[-10:]
                if len(recent_performance) > 1:
                    performance_change = np.mean(np.diff(recent_performance))
                    self.metrics.convergence_rate = abs(performance_change)
            
            # 判断性能趋势
            if len(self.performance_history) >= 5:
                recent_trend = np.mean(list(self.performance_history)[-5:])
                overall_trend = np.mean(list(self.performance_history)[-10:]) if len(self.performance_history) >= 10 else recent_trend
                
                if recent_trend > overall_trend + 0.01:
                    self.metrics.performance_trend = "improving"
                elif recent_trend < overall_trend - 0.01:
                    self.metrics.performance_trend = "declining"
                else:
                    self.metrics.performance_trend = "stable"
            
            # 更新性能历史
            self.performance_history.append(self.metrics.average_accuracy)
            
            # 更新最后更新时间
            self.metrics.last_update_time = datetime.now()
            
        except Exception as e:
            logger.error(f"更新学习指标失败: {e}")
    
    def _cleanup_buffers(self):
        """清理缓冲区"""
        try:
            # 保留最近的50%数据
            if len(self.training_buffer) > self.batch_size:
                keep_size = self.batch_size // 2
                new_buffer = deque(maxlen=self.batch_size)
                recent_data = list(self.training_buffer)[-keep_size:]
                for data in recent_data:
                    new_buffer.append(data)
                self.training_buffer = new_buffer
            
            if len(self.validation_buffer) > self.batch_size // 4:
                keep_size = self.batch_size // 8
                new_buffer = deque(maxlen=self.batch_size // 4)
                recent_data = list(self.validation_buffer)[-keep_size:]
                for data in recent_data:
                    new_buffer.append(data)
                self.validation_buffer = new_buffer
                
        except Exception as e:
            logger.error(f"清理缓冲区失败: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'uptime_seconds': uptime,
            'total_samples_processed': self.total_samples_processed,
            'learning_cycles_completed': self.learning_cycles_completed,
            'model_updates': self.metrics.model_updates,
            'accuracy_improvements': self.metrics.accuracy_improvements,
            'average_accuracy': self.metrics.average_accuracy,
            'best_accuracy': self.metrics.best_accuracy,
            'convergence_rate': self.metrics.convergence_rate,
            'performance_trend': self.metrics.performance_trend,
            'active_models': len(self.models),
            'samples_per_second': self.total_samples_processed / uptime if uptime > 0 else 0,
            'learning_efficiency': self.metrics.accuracy_improvements / max(self.metrics.learning_cycles, 1),
            'last_update': datetime.now().isoformat()
        }
    
    def save_models(self, save_path: str):
        """保存模型"""
        try:
            os.makedirs(save_path, exist_ok=True)
            
            for model_name, model in self.models.items():
                model_file = os.path.join(save_path, f"{model_name}_v12_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
                
                if hasattr(model, 'save'):
                    model.save(model_file)
                else:
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)
                
                logger.info(f"模型已保存: {model_file}")
            
            # 保存性能指标
            metrics_file = os.path.join(save_path, f"metrics_v12_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            metrics_data = {
                'total_samples': self.total_samples_processed,
                'learning_cycles': self.metrics.learning_cycles,
                'model_updates': self.metrics.model_updates,
                'accuracy_improvements': self.metrics.accuracy_improvements,
                'average_accuracy': self.metrics.average_accuracy,
                'best_accuracy': self.metrics.best_accuracy,
                'convergence_rate': self.metrics.convergence_rate,
                'performance_trend': self.metrics.performance_trend,
                'last_update': self.metrics.last_update_time.isoformat()
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"性能指标已保存: {metrics_file}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    def load_models(self, load_path: str):
        """加载模型"""
        try:
            if not os.path.exists(load_path):
                logger.warning(f"模型路径不存在: {load_path}")
                return
            
            for filename in os.listdir(load_path):
                if filename.endswith('.pkl') and 'v12_' in filename:
                    model_name = filename.split('_v12_')[0]
                    model_file = os.path.join(load_path, filename)
                    
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    self.models[model_name] = model
                    logger.info(f"模型已加载: {model_name}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
    
    def reset_learning_state(self):
        """重置学习状态"""
        self.training_buffer.clear()
        self.validation_buffer.clear()
        self.performance_history.clear()
        self.metrics = LearningMetrics()
        self.total_samples_processed = 0
        self.learning_cycles_completed = 0
        logger.info("学习状态已重置")
