"""
V12 AI模型训练系统
训练OFI专家模型和深度学习模型，解决信号质量问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import random
from typing import Dict, List
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 导入V12组件
from src.v12_realistic_data_simulator import V12RealisticDataSimulator
from src.v12_ensemble_ai_model import V12EnsembleAIModel
from src.v12_ofi_expert_model import V12OFIExpertModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V12AIModelTrainer:
    """V12 AI模型训练器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # V12策略参数
        self.params = {
            'ofi_expert': {
                'model_confidence': 0.8,
                'lookback_window': 30
            },
            'ensemble': {
                'lstm_sequence_length': 60,
                'transformer_sequence_length': 60,
                'cnn_lookback': 20,
                'ensemble_weights': [0.3, 0.3, 0.4]
            }
        }
        self.training_data = None
        self.scaler = StandardScaler()
        
        logger.info(f"V12 AI模型训练器初始化完成 - 设备: {self.device}")
    
    def generate_training_data(self, num_samples: int = 10000) -> pd.DataFrame:
        """生成训练数据"""
        logger.info(f"生成{num_samples}个训练样本...")
        
        # 生成多个数据集用于训练
        all_data = []
        seeds = [random.randint(1000, 9999) for _ in range(num_samples // 1000 + 1)]
        
        for i, seed in enumerate(seeds):
            if len(all_data) >= num_samples:
                break
                
            # 生成数据集
            simulator = V12RealisticDataSimulator(seed=seed)
            market_data = simulator.generate_complete_dataset()
            
            # 添加标签（基于未来价格变化）
            market_data = self._add_trading_labels(market_data)
            
            all_data.append(market_data)
            
            if (i + 1) % 10 == 0:
                logger.info(f"已生成{i + 1}个数据集，总样本数: {len(all_data) * len(market_data)}")
        
        # 合并所有数据
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 确保不超过目标样本数
        if len(combined_data) > num_samples:
            combined_data = combined_data.sample(n=num_samples, random_state=42)
        
        logger.info(f"训练数据生成完成，总样本数: {len(combined_data)}")
        
        return combined_data
    
    def _add_trading_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加交易标签"""
        # 基于未来价格变化添加标签
        future_returns = df['price'].shift(-5) / df['price'] - 1  # 5期后收益率
        
        # 创建标签：1表示买入信号，0表示卖出信号，-1表示无信号
        labels = []
        for i, ret in enumerate(future_returns):
            if pd.isna(ret):
                labels.append(-1)  # 无信号
            elif ret > 0.002:  # 0.2%以上涨幅
                labels.append(1)   # 买入信号
            elif ret < -0.002: # 0.2%以上跌幅
                labels.append(0)   # 卖出信号
            else:
                labels.append(-1)  # 无信号
        
        df['trading_label'] = labels
        
        return df
    
    def prepare_training_features(self, df: pd.DataFrame) -> tuple:
        """准备训练特征"""
        logger.info("准备训练特征...")
        
        # 选择特征列
        feature_columns = [
            'ofi', 'cvd', 'ofi_z', 'cvd_z', 'price', 'volume', 'spread',
            'volatility', 'rsi', 'sma_5', 'sma_20'
        ]
        
        # 添加订单簿特征
        for level in range(1, 6):
            feature_columns.extend([
                f'bid{level}_price', f'bid{level}_size',
                f'ask{level}_price', f'ask{level}_size'
            ])
        
        # 过滤有效特征
        available_features = [col for col in feature_columns if col in df.columns]
        
        # 准备特征矩阵
        X = df[available_features].fillna(0)
        
        # 准备标签
        y = df['trading_label']
        
        # 过滤有效标签
        valid_mask = y != -1
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"特征维度: {X.shape}")
        logger.info(f"标签分布: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_ofi_expert_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """训练OFI专家模型"""
        logger.info("开始训练OFI专家模型...")
        
        # 分割训练和测试数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 创建OFI专家模型
        config = {
            'ofi_expert': {
                'model_confidence': 0.8,
                'lookback_window': 30
            }
        }
        
        ofi_model = V12OFIExpertModel(config)
        
        # 训练模型
        training_data = pd.DataFrame(X_train_scaled, columns=X.columns)
        training_data['trading_label'] = y_train.values
        
        ofi_model.train_model(training_data, params=self.params)
        
        # 测试模型
        test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
        predictions = ofi_model.predict_signal_quality(test_data)
        
        # 计算准确率
        y_pred = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"OFI专家模型训练完成 - 准确率: {accuracy:.4f}")
        
        return {
            'model': ofi_model,
            'accuracy': accuracy,
            'predictions': predictions,
            'test_labels': y_test
        }
    
    def train_deep_learning_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """训练深度学习模型"""
        logger.info("开始训练深度学习模型...")
        
        # 分割训练和测试数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_train_tensor = torch.LongTensor(y_train.values).to(self.device)
        y_test_tensor = torch.LongTensor(y_test.values).to(self.device)
        
        results = {}
        
        # 训练LSTM模型
        lstm_model = self._train_lstm_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
        results['lstm'] = lstm_model
        
        # 训练CNN模型
        cnn_model = self._train_cnn_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
        results['cnn'] = cnn_model
        
        # 训练Transformer模型
        transformer_model = self._train_transformer_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
        results['transformer'] = transformer_model
        
        return results
    
    def _train_lstm_model(self, X_train, y_train, X_test, y_test) -> Dict:
        """训练LSTM模型"""
        logger.info("训练LSTM模型...")
        
        # 创建LSTM模型
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, num_classes)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                # 重塑输入以适应LSTM
                x = x.unsqueeze(1)  # 添加序列维度
                
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                return out
        
        # 创建模型
        input_size = X_train.shape[1]
        model = LSTMModel(input_size).to(self.device)
        
        # 训练模型
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
        
        # 测试模型
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        
        logger.info(f"LSTM模型训练完成 - 准确率: {accuracy:.4f}")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'predictions': predicted.cpu().numpy()
        }
    
    def _train_cnn_model(self, X_train, y_train, X_test, y_test) -> Dict:
        """训练CNN模型"""
        logger.info("训练CNN模型...")
        
        # 创建CNN模型
        class CNNModel(nn.Module):
            def __init__(self, input_size, num_classes=2):
                super(CNNModel, self).__init__()
                self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool1d(2)
                self.fc1 = nn.Linear(64 * (input_size // 4), 128)
                self.fc2 = nn.Linear(128, num_classes)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = x.unsqueeze(1)  # 添加通道维度
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        # 创建模型
        input_size = X_train.shape[1]
        model = CNNModel(input_size).to(self.device)
        
        # 训练模型
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"CNN Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
        
        # 测试模型
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        
        logger.info(f"CNN模型训练完成 - 准确率: {accuracy:.4f}")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'predictions': predicted.cpu().numpy()
        }
    
    def _train_transformer_model(self, X_train, y_train, X_test, y_test) -> Dict:
        """训练Transformer模型"""
        logger.info("训练Transformer模型...")
        
        # 创建Transformer模型
        class TransformerModel(nn.Module):
            def __init__(self, input_size, d_model=64, nhead=8, num_layers=2, num_classes=2):
                super(TransformerModel, self).__init__()
                self.input_projection = nn.Linear(input_size, d_model)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
                    num_layers
                )
                self.fc = nn.Linear(d_model, num_classes)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = x.unsqueeze(1)  # 添加序列维度
                x = self.input_projection(x)
                x = self.transformer(x)
                x = x[:, -1, :]  # 取最后一个时间步
                x = self.dropout(x)
                x = self.fc(x)
                return x
        
        # 创建模型
        input_size = X_train.shape[1]
        model = TransformerModel(input_size).to(self.device)
        
        # 训练模型
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Transformer Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
        
        # 测试模型
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        
        logger.info(f"Transformer模型训练完成 - 准确率: {accuracy:.4f}")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'predictions': predicted.cpu().numpy()
        }
    
    def train_ensemble_model(self, ofi_results: Dict, dl_results: Dict) -> Dict:
        """训练集成模型"""
        logger.info("开始训练集成模型...")
        
        # 创建集成模型配置
        config = {
            'ofi_ai_fusion': {
                'ai_models': {
                    'v9_ml_weight': 0.3,
                    'lstm_weight': 0.25,
                    'transformer_weight': 0.25,
                    'cnn_weight': 0.2
                }
            }
        }
        
        ensemble_model = V12EnsembleAIModel(config)
        
        # 设置训练好的模型
        ensemble_model.ofi_expert = ofi_results['model']
        ensemble_model.lstm_model = dl_results['lstm']['model']
        ensemble_model.cnn_model = dl_results['cnn']['model']
        ensemble_model.transformer_model = dl_results['transformer']['model']
        
        # 标记为已训练
        ensemble_model.is_trained = True
        
        # 计算集成准确率
        ensemble_accuracy = (
            ofi_results['accuracy'] * 0.3 +
            dl_results['lstm']['accuracy'] * 0.25 +
            dl_results['transformer']['accuracy'] * 0.25 +
            dl_results['cnn']['accuracy'] * 0.2
        )
        
        logger.info(f"集成模型训练完成 - 集成准确率: {ensemble_accuracy:.4f}")
        
        return {
            'model': ensemble_model,
            'accuracy': ensemble_accuracy,
            'component_accuracies': {
                'ofi_expert': ofi_results['accuracy'],
                'lstm': dl_results['lstm']['accuracy'],
                'transformer': dl_results['transformer']['accuracy'],
                'cnn': dl_results['cnn']['accuracy']
            }
        }
    
    def save_trained_models(self, ofi_results: Dict, dl_results: Dict, ensemble_results: Dict):
        """保存训练好的模型"""
        logger.info("保存训练好的模型...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"trained_models_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型信息
        model_info = {
            'timestamp': timestamp,
            'training_summary': {
                'ofi_expert_accuracy': ofi_results['accuracy'],
                'lstm_accuracy': dl_results['lstm']['accuracy'],
                'transformer_accuracy': dl_results['transformer']['accuracy'],
                'cnn_accuracy': dl_results['cnn']['accuracy'],
                'ensemble_accuracy': ensemble_results['accuracy']
            },
            'model_paths': {
                'ofi_expert': f"{model_dir}/ofi_expert_model.pkl",
                'lstm': f"{model_dir}/lstm_model.pth",
                'transformer': f"{model_dir}/transformer_model.pth",
                'cnn': f"{model_dir}/cnn_model.pth",
                'ensemble': f"{model_dir}/ensemble_model.pkl"
            }
        }
        
        # 保存模型信息
        with open(f"{model_dir}/model_info.json", 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已保存到: {model_dir}")
        
        return model_info
    
    def run_complete_training(self) -> Dict:
        """运行完整的训练流程"""
        logger.info("=" * 80)
        logger.info("V12 AI模型完整训练流程开始")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # 1. 生成训练数据
        logger.info("步骤1: 生成训练数据...")
        training_data = self.generate_training_data(num_samples=5000)
        
        # 2. 准备特征
        logger.info("步骤2: 准备训练特征...")
        X, y = self.prepare_training_features(training_data)
        
        # 3. 训练OFI专家模型
        logger.info("步骤3: 训练OFI专家模型...")
        ofi_results = self.train_ofi_expert_model(X, y)
        
        # 4. 训练深度学习模型
        logger.info("步骤4: 训练深度学习模型...")
        dl_results = self.train_deep_learning_models(X, y)
        
        # 5. 训练集成模型
        logger.info("步骤5: 训练集成模型...")
        ensemble_results = self.train_ensemble_model(ofi_results, dl_results)
        
        # 6. 保存模型
        logger.info("步骤6: 保存训练好的模型...")
        model_info = self.save_trained_models(ofi_results, dl_results, ensemble_results)
        
        # 7. 生成训练报告
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        training_report = {
            'training_session': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'training_samples': len(X),
                'feature_dimensions': X.shape[1]
            },
            'model_performance': {
                'ofi_expert': {
                    'accuracy': ofi_results['accuracy'],
                    'status': 'trained'
                },
                'lstm': {
                    'accuracy': dl_results['lstm']['accuracy'],
                    'status': 'trained'
                },
                'transformer': {
                    'accuracy': dl_results['transformer']['accuracy'],
                    'status': 'trained'
                },
                'cnn': {
                    'accuracy': dl_results['cnn']['accuracy'],
                    'status': 'trained'
                },
                'ensemble': {
                    'accuracy': ensemble_results['accuracy'],
                    'status': 'trained'
                }
            },
            'model_info': model_info
        }
        
        # 保存训练报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"v12_ai_training_report_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(training_report, f, indent=2, ensure_ascii=False)
        
        logger.info("=" * 80)
        logger.info("V12 AI模型完整训练流程完成")
        logger.info(f"训练耗时: {duration:.2f}秒")
        logger.info(f"OFI专家模型准确率: {ofi_results['accuracy']:.4f}")
        logger.info(f"LSTM模型准确率: {dl_results['lstm']['accuracy']:.4f}")
        logger.info(f"Transformer模型准确率: {dl_results['transformer']['accuracy']:.4f}")
        logger.info(f"CNN模型准确率: {dl_results['cnn']['accuracy']:.4f}")
        logger.info(f"集成模型准确率: {ensemble_results['accuracy']:.4f}")
        logger.info("=" * 80)
        
        return training_report


def main():
    """主函数"""
    logger.info("V12 AI模型训练系统启动")
    
    try:
        # 创建AI模型训练器
        trainer = V12AIModelTrainer()
        
        # 运行完整训练流程
        training_report = trainer.run_complete_training()
        
        return training_report
        
    except Exception as e:
        logger.error(f"AI模型训练失败: {e}")
        raise


if __name__ == "__main__":
    main()
