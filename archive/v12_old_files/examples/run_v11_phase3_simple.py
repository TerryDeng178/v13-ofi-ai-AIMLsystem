"""
V11 Phase 3: 简化版实时学习系统测试
测试在线学习、增量学习、模型监控和自适应调整
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime
import time
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleOnlineLearningSystem:
    """简化版在线学习系统"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.optimizers = {}
        self.performance_history = []
        self.learning_results = []
        
        logger.info(f"简化版在线学习系统初始化完成，设备: {self.device}")
    
    def initialize_models(self, feature_dim: int):
        """初始化简化模型"""
        logger.info("初始化简化模型...")
        
        # 简单的线性模型
        self.models['linear'] = torch.nn.Linear(feature_dim, 1).to(self.device)
        self.optimizers['linear'] = torch.optim.Adam(self.models['linear'].parameters(), lr=0.001)
        
        logger.info("简化模型初始化完成")
    
    def online_update(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """在线学习更新"""
        logger.info("执行在线学习更新...")
        
        # 转换为张量
        X = torch.FloatTensor(features).to(self.device)
        y = torch.FloatTensor(labels).to(self.device)
        
        # 确保维度匹配
        if X.shape[0] != y.shape[0]:
            min_len = min(X.shape[0], y.shape[0])
            X = X[:min_len]
            y = y[:min_len]
        
        # 训练模型
        self.models['linear'].train()
        self.optimizers['linear'].zero_grad()
        
        # 前向传播
        predictions = self.models['linear'](X)
        loss = torch.nn.MSELoss()(predictions.squeeze(), y)
        
        # 反向传播
        loss.backward()
        self.optimizers['linear'].step()
        
        # 计算准确率
        with torch.no_grad():
            pred_labels = (predictions > 0.5).float()
            correct = (pred_labels == y).float().sum()
            accuracy = correct / len(y)
        
        result = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'samples': len(features)
        }
        
        self.learning_results.append(result)
        logger.info(f"在线学习完成: 损失={loss.item():.4f}, 准确率={accuracy.item():.4f}")
        
        return result
    
    def incremental_learning(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """增量学习"""
        logger.info("执行增量学习...")
        
        # 转换为张量
        X = torch.FloatTensor(features).to(self.device)
        y = torch.FloatTensor(labels).to(self.device)
        
        # 确保维度匹配
        if X.shape[0] != y.shape[0]:
            min_len = min(X.shape[0], y.shape[0])
            X = X[:min_len]
            y = y[:min_len]
        
        # 增量更新
        self.models['linear'].train()
        
        # 多次小批量更新
        batch_size = min(32, len(X))
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            self.optimizers['linear'].zero_grad()
            predictions = self.models['linear'](batch_X)
            loss = torch.nn.MSELoss()(predictions.squeeze(), batch_y)
            loss.backward()
            self.optimizers['linear'].step()
        
        # 计算最终性能
        with torch.no_grad():
            predictions = self.models['linear'](X)
            pred_labels = (predictions > 0.5).float()
            correct = (pred_labels == y).float().sum()
            accuracy = correct / len(y)
        
        result = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'samples': len(features)
        }
        
        self.learning_results.append(result)
        logger.info(f"增量学习完成: 损失={loss.item():.4f}, 准确率={accuracy.item():.4f}")
        
        return result
    
    def monitor_performance(self) -> dict:
        """监控性能"""
        logger.info("监控模型性能...")
        
        if not self.learning_results:
            return {'status': 'no_data'}
        
        # 计算性能统计
        recent_results = self.learning_results[-10:]  # 最近10次结果
        
        avg_loss = np.mean([r['loss'] for r in recent_results])
        avg_accuracy = np.mean([r['accuracy'] for r in recent_results])
        total_samples = sum([r['samples'] for r in recent_results])
        
        performance = {
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'total_samples': total_samples,
            'recent_updates': len(recent_results)
        }
        
        self.performance_history.append({
            'timestamp': time.time(),
            'performance': performance
        })
        
        logger.info(f"性能监控: 平均损失={avg_loss:.4f}, 平均准确率={avg_accuracy:.4f}")
        
        return performance
    
    def adaptive_adjustment(self) -> dict:
        """自适应调整"""
        logger.info("执行自适应调整...")
        
        if not self.performance_history:
            return {'status': 'no_data'}
        
        latest_performance = self.performance_history[-1]['performance']
        
        # 简单的自适应逻辑
        adjustments = {}
        
        if latest_performance['avg_accuracy'] < 0.6:
            # 如果准确率低，增加学习率
            for param_group in self.optimizers['linear'].param_groups:
                param_group['lr'] *= 1.1
            adjustments['learning_rate'] = 'increased'
            logger.info("学习率已增加")
        
        if latest_performance['avg_loss'] > 1.0:
            # 如果损失高，减少学习率
            for param_group in self.optimizers['linear'].param_groups:
                param_group['lr'] *= 0.9
            adjustments['learning_rate'] = 'decreased'
            logger.info("学习率已减少")
        
        if not adjustments:
            adjustments['status'] = 'no_adjustment_needed'
            logger.info("无需调整")
        
        return adjustments


class V11Phase3SimpleTester:
    """V11 Phase 3 简化版测试器"""
    
    def __init__(self):
        self.config = {
            'learning_rate': 0.001,
            'feature_dim': 128
        }
        
        self.online_system = SimpleOnlineLearningSystem(self.config)
        self.test_data = None
        
        logger.info("V11 Phase 3 简化版测试器初始化完成")
    
    def generate_test_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """生成测试数据"""
        logger.info(f"生成 {num_samples} 条测试数据...")
        
        np.random.seed(42)
        
        # 生成特征数据
        features = np.random.randn(num_samples, self.config['feature_dim'])
        
        # 生成标签（基于特征的线性组合）
        weights = np.random.randn(self.config['feature_dim'])
        linear_combination = np.dot(features, weights)
        labels = (linear_combination > 0).astype(int)
        
        # 添加一些噪声
        noise = np.random.normal(0, 0.1, num_samples)
        labels = (linear_combination + noise > 0).astype(int)
        
        df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(self.config['feature_dim'])])
        df['label'] = labels
        
        self.test_data = df
        logger.info(f"测试数据生成完成: {len(df)} 条记录")
        return df
    
    def test_online_learning_system(self):
        """测试在线学习系统"""
        logger.info("=" * 60)
        logger.info("V11 Phase 3 简化版实时学习系统测试")
        logger.info("=" * 60)
        
        # 生成测试数据
        df = self.generate_test_data(1000)
        
        # 分离特征和标签
        features = df.drop('label', axis=1).values
        labels = df['label'].values
        
        # 初始化在线学习系统
        logger.info("步骤1: 初始化在线学习系统...")
        self.online_system.initialize_models(self.config['feature_dim'])
        logger.info("在线学习系统初始化完成")
        
        # 测试在线学习
        logger.info("步骤2: 测试在线学习...")
        self._test_online_learning(features, labels)
        
        # 测试增量学习
        logger.info("步骤3: 测试增量学习...")
        self._test_incremental_learning(features, labels)
        
        # 测试性能监控
        logger.info("步骤4: 测试性能监控...")
        self._test_performance_monitoring()
        
        # 测试自适应调整
        logger.info("步骤5: 测试自适应调整...")
        self._test_adaptive_adjustment()
        
        # 生成测试报告
        self._generate_test_report()
    
    def _test_online_learning(self, features: np.ndarray, labels: np.ndarray):
        """测试在线学习"""
        logger.info("开始测试在线学习...")
        
        # 分批处理数据
        batch_size = 100
        num_batches = len(features) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(features))
            
            batch_features = features[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            
            # 在线学习更新
            result = self.online_system.online_update(batch_features, batch_labels)
            
            if i % 5 == 0:
                logger.info(f"处理批次 {i}/{num_batches}, 样本数: {len(batch_features)}")
        
        logger.info(f"在线学习完成: {num_batches} 个批次")
    
    def _test_incremental_learning(self, features: np.ndarray, labels: np.ndarray):
        """测试增量学习"""
        logger.info("开始测试增量学习...")
        
        # 选择部分数据进行增量学习
        sample_indices = np.random.choice(len(features), 200, replace=False)
        sample_features = features[sample_indices]
        sample_labels = labels[sample_indices]
        
        # 执行增量学习
        result = self.online_system.incremental_learning(sample_features, sample_labels)
        
        logger.info(f"增量学习结果: {result}")
    
    def _test_performance_monitoring(self):
        """测试性能监控"""
        logger.info("开始测试性能监控...")
        
        # 获取性能报告
        performance_report = self.online_system.monitor_performance()
        
        logger.info("性能监控结果:")
        for key, value in performance_report.items():
            logger.info(f"  {key}: {value}")
    
    def _test_adaptive_adjustment(self):
        """测试自适应调整"""
        logger.info("开始测试自适应调整...")
        
        # 执行自适应调整
        adjustments = self.online_system.adaptive_adjustment()
        
        logger.info(f"自适应调整结果: {adjustments}")
    
    def _generate_test_report(self):
        """生成测试报告"""
        logger.info("=" * 60)
        logger.info("V11 Phase 3 简化版测试报告")
        logger.info("=" * 60)
        
        # 学习结果统计
        total_updates = len(self.online_system.learning_results)
        total_samples = sum([r['samples'] for r in self.online_system.learning_results])
        
        logger.info(f"学习统计:")
        logger.info(f"  总更新次数: {total_updates}")
        logger.info(f"  总样本数: {total_samples}")
        logger.info(f"  性能监控次数: {len(self.online_system.performance_history)}")
        
        # 最新性能
        if self.online_system.learning_results:
            latest_result = self.online_system.learning_results[-1]
            logger.info(f"最新学习结果:")
            logger.info(f"  损失: {latest_result['loss']:.4f}")
            logger.info(f"  准确率: {latest_result['accuracy']:.4f}")
            logger.info(f"  样本数: {latest_result['samples']}")
        
        # 保存测试结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"v11_phase3_simple_results_{timestamp}.json"
        
        test_results = {
            'timestamp': timestamp,
            'config': self.config,
            'learning_results': self.online_system.learning_results,
            'performance_history': self.online_system.performance_history,
            'summary': {
                'total_updates': total_updates,
                'total_samples': total_samples,
                'performance_monitoring_count': len(self.online_system.performance_history)
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"测试结果已保存到: {results_file}")
        logger.info("V11 Phase 3 简化版测试完成！")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("V11 Phase 3 简化版实时学习系统")
    logger.info("=" * 60)
    
    # 创建测试器
    tester = V11Phase3SimpleTester()
    
    # 运行测试
    tester.test_online_learning_system()
    
    logger.info("V11 Phase 3 简化版测试完成！")


if __name__ == "__main__":
    main()
