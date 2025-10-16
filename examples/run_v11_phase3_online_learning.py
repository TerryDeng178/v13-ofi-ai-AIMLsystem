"""
V11 Phase 3: 实时学习系统测试
测试在线学习、增量学习、模型监控和自适应调整
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime, timedelta
import time
import json

# 导入V11模块
from src.v11_online_learning import OnlineLearningSystem
from src.v11_advanced_features import V11AdvancedFeatureEngine
from src.v11_deep_learning import V11DeepLearning
from src.v11_signal_optimizer import V11SignalOptimizer
from src.v11_risk_manager import V11RiskManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11Phase3OnlineLearningTester:
    """V11 Phase 3 实时学习系统测试器"""
    
    def __init__(self):
        self.config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'update_frequency': 50,  # 每50个样本更新一次
            'performance_threshold': 0.6,
            'memory_size': 1000,
            'sequence_length': 60,
            'feature_dim': 128
        }
        
        # 初始化组件
        self.feature_engineer = V11AdvancedFeatureEngine()
        self.ml_models = V11DeepLearning(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.signal_optimizer = V11SignalOptimizer()
        self.risk_manager = V11RiskManager()
        self.online_system = OnlineLearningSystem(self.config)
        
        # 测试数据
        self.test_data = None
        self.learning_results = []
        self.performance_history = []
        
        logger.info("V11 Phase 3 实时学习系统测试器初始化完成")
    
    def generate_test_data(self, num_samples: int = 2000) -> pd.DataFrame:
        """生成测试数据"""
        logger.info(f"生成 {num_samples} 条测试数据...")
        
        # 生成模拟市场数据
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=num_samples, freq='1min')
        
        # 价格数据
        price_base = 100.0
        price_changes = np.random.normal(0, 0.01, num_samples)
        prices = [price_base]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        # 创建DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, num_samples)
        })
        
        # 确保high >= low
        df['high'] = np.maximum(df['high'], df['low'])
        
        self.test_data = df
        logger.info(f"测试数据生成完成: {len(df)} 条记录")
        return df
    
    def test_online_learning_system(self):
        """测试在线学习系统"""
        logger.info("=" * 60)
        logger.info("V11 Phase 3 实时学习系统测试")
        logger.info("=" * 60)
        
        # 生成测试数据
        df = self.generate_test_data(2000)
        
        # 特征工程
        logger.info("步骤1: 特征工程...")
        df_features = self.feature_engineer.create_all_features(df)
        logger.info(f"特征工程完成: {df_features.shape[1]} 个特征")
        
        # 初始化在线学习系统
        logger.info("步骤2: 初始化在线学习系统...")
        self.online_system.initialize_models(
            feature_dim=self.config['feature_dim'],
            sequence_length=self.config['sequence_length']
        )
        logger.info("在线学习系统初始化完成")
        
        # 模拟实时数据流
        logger.info("步骤3: 模拟实时数据流...")
        self._simulate_real_time_learning(df_features)
        
        # 测试增量学习
        logger.info("步骤4: 测试增量学习...")
        self._test_incremental_learning(df_features)
        
        # 测试性能监控
        logger.info("步骤5: 测试性能监控...")
        self._test_performance_monitoring()
        
        # 测试自适应调整
        logger.info("步骤6: 测试自适应调整...")
        self._test_adaptive_adjustment()
        
        # 生成测试报告
        self._generate_test_report()
    
    def _simulate_real_time_learning(self, df_features: pd.DataFrame):
        """模拟实时学习"""
        logger.info("开始模拟实时学习...")
        
        # 分批处理数据
        batch_size = 100
        num_batches = len(df_features) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df_features))
            
            # 获取当前批次数据
            batch_data = df_features.iloc[start_idx:end_idx]
            
            # 生成标签（简化：基于价格变化）
            if 'future_return_1' in batch_data.columns:
                labels = (batch_data['future_return_1'] > 0).astype(int).values
            else:
                labels = np.random.randint(0, 2, len(batch_data))
            
            # 在线学习更新
            result = self.online_system.online_update(batch_data, labels)
            
            # 记录结果
            self.learning_results.append({
                'batch': i,
                'samples': len(batch_data),
                'result': result
            })
            
            if i % 10 == 0:
                logger.info(f"处理批次 {i}/{num_batches}, 样本数: {len(batch_data)}")
        
        logger.info(f"实时学习完成: {num_batches} 个批次")
    
    def _test_incremental_learning(self, df_features: pd.DataFrame):
        """测试增量学习"""
        logger.info("开始测试增量学习...")
        
        # 选择部分数据进行增量学习
        sample_data = df_features.sample(n=200, random_state=42)
        
        # 生成标签
        if 'future_return_1' in sample_data.columns:
            labels = (sample_data['future_return_1'] > 0).astype(int).values
        else:
            labels = np.random.randint(0, 2, len(sample_data))
        
        # 执行增量学习
        result = self.online_system.incremental_learning(sample_data, labels)
        
        logger.info(f"增量学习结果: {result}")
        
        # 记录结果
        self.learning_results.append({
            'type': 'incremental',
            'samples': len(sample_data),
            'result': result
        })
    
    def _test_performance_monitoring(self):
        """测试性能监控"""
        logger.info("开始测试性能监控...")
        
        # 获取性能报告
        performance_report = self.online_system.monitor_performance()
        
        logger.info("性能监控结果:")
        for model_name, metrics in performance_report.items():
            logger.info(f"  {model_name}:")
            for metric_name, value in metrics.items():
                logger.info(f"    {metric_name}: {value}")
        
        # 记录性能历史
        self.performance_history.append({
            'timestamp': time.time(),
            'performance': performance_report
        })
    
    def _test_adaptive_adjustment(self):
        """测试自适应调整"""
        logger.info("开始测试自适应调整...")
        
        # 模拟性能下降
        logger.info("模拟性能下降场景...")
        
        # 获取当前性能
        performance_report = self.online_system.monitor_performance()
        
        # 检查是否需要调整
        if self.online_system.adaptive_controller.should_adjust(performance_report):
            adjustments = self.online_system.adaptive_controller.get_adjustments(performance_report)
            logger.info(f"自适应调整建议: {adjustments}")
            
            # 应用调整
            self.online_system._apply_adjustments(adjustments)
            logger.info("自适应调整已应用")
        else:
            logger.info("当前性能良好，无需调整")
    
    def _generate_test_report(self):
        """生成测试报告"""
        logger.info("=" * 60)
        logger.info("V11 Phase 3 测试报告")
        logger.info("=" * 60)
        
        # 学习结果统计
        total_batches = len([r for r in self.learning_results if 'batch' in r])
        total_samples = sum([r['samples'] for r in self.learning_results if 'samples' in r])
        
        logger.info(f"学习统计:")
        logger.info(f"  总批次数: {total_batches}")
        logger.info(f"  总样本数: {total_samples}")
        logger.info(f"  学习结果数: {len(self.learning_results)}")
        logger.info(f"  性能监控次数: {len(self.performance_history)}")
        
        # 模型性能分析
        if self.performance_history:
            latest_performance = self.performance_history[-1]['performance']
            logger.info("最新模型性能:")
            for model_name, metrics in latest_performance.items():
                logger.info(f"  {model_name}:")
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"    {metric_name}: {value:.4f}")
                    else:
                        logger.info(f"    {metric_name}: {value}")
        
        # 保存测试结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"v11_phase3_online_learning_results_{timestamp}.json"
        
        test_results = {
            'timestamp': timestamp,
            'config': self.config,
            'learning_results': self.learning_results,
            'performance_history': self.performance_history,
            'summary': {
                'total_batches': total_batches,
                'total_samples': total_samples,
                'learning_results_count': len(self.learning_results),
                'performance_monitoring_count': len(self.performance_history)
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"测试结果已保存到: {results_file}")
        logger.info("V11 Phase 3 实时学习系统测试完成！")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("V11 Phase 3 实时学习系统")
    logger.info("=" * 60)
    
    # 创建测试器
    tester = V11Phase3OnlineLearningTester()
    
    # 运行测试
    tester.test_online_learning_system()
    
    logger.info("V11 Phase 3 测试完成！")


if __name__ == "__main__":
    main()
