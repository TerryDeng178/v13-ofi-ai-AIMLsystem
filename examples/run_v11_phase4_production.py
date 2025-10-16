"""
V11 Phase 4: 生产部署系统测试
测试系统部署、实盘测试、性能优化和监控告警
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
import threading

# 导入V11模块
from src.v11_production_deployment import ProductionDeploymentSystem
from src.v11_advanced_features import V11AdvancedFeatureEngine
from src.v11_deep_learning import V11DeepLearning
from src.v11_signal_optimizer import V11SignalOptimizer
from src.v11_risk_manager import V11RiskManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11Phase4ProductionTester:
    """V11 Phase 4 生产部署系统测试器"""
    
    def __init__(self):
        self.config = {
            'max_memory_usage': 0.8,
            'max_gpu_usage': 0.8,
            'performance_threshold': 0.6,
            'alert_threshold': 0.5,
            'feature_dim': 128,
            'sequence_length': 60
        }
        
        # 初始化组件
        self.feature_engineer = V11AdvancedFeatureEngine()
        self.ml_models = V11DeepLearning(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.signal_optimizer = V11SignalOptimizer()
        self.risk_manager = V11RiskManager()
        self.deployment_system = ProductionDeploymentSystem(self.config)
        
        # 测试数据
        self.test_data = None
        self.production_results = []
        
        logger.info("V11 Phase 4 生产部署系统测试器初始化完成")
    
    def generate_production_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """生成生产测试数据"""
        logger.info(f"生成 {num_samples} 条生产测试数据...")
        
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=num_samples, freq='1min')
        
        # 生成模拟市场数据
        price_base = 100.0
        price_changes = np.random.normal(0, 0.01, num_samples)
        prices = [price_base]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
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
        logger.info(f"生产测试数据生成完成: {len(df)} 条记录")
        return df
    
    def test_production_deployment(self):
        """测试生产部署系统"""
        logger.info("=" * 60)
        logger.info("V11 Phase 4 生产部署系统测试")
        logger.info("=" * 60)
        
        # 生成测试数据
        df = self.generate_production_data(1000)
        
        # 特征工程
        logger.info("步骤1: 特征工程...")
        df_features = self.feature_engineer.create_all_features(df)
        logger.info(f"特征工程完成: {df_features.shape[1]} 个特征")
        
        # 部署生产系统
        logger.info("步骤2: 部署生产系统...")
        deployment_result = self.deployment_system.deploy_system()
        logger.info(f"部署结果: {deployment_result}")
        
        if deployment_result['status'] != 'success':
            logger.error("生产系统部署失败")
            return
        
        # 测试数据处理
        logger.info("步骤3: 测试数据处理...")
        self._test_data_processing(df_features)
        
        # 测试性能监控
        logger.info("步骤4: 测试性能监控...")
        self._test_performance_monitoring()
        
        # 测试告警系统
        logger.info("步骤5: 测试告警系统...")
        self._test_alert_system()
        
        # 测试系统稳定性
        logger.info("步骤6: 测试系统稳定性...")
        self._test_system_stability()
        
        # 生成生产报告
        self._generate_production_report()
        
        # 停止生产系统
        logger.info("步骤7: 停止生产系统...")
        self.deployment_system.stop_system()
        logger.info("生产系统已停止")
    
    def _test_data_processing(self, df_features: pd.DataFrame):
        """测试数据处理"""
        logger.info("开始测试数据处理...")
        
        # 分批发送数据
        batch_size = 100
        num_batches = len(df_features) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df_features))
            
            batch_data = df_features.iloc[start_idx:end_idx]
            
            # 准备数据
            features = batch_data.select_dtypes(include=[np.number]).values
            
            # 发送数据到生产系统
            for j, feature_row in enumerate(features):
                data = {
                    'features': feature_row,
                    'timestamp': time.time(),
                    'batch_id': i,
                    'sample_id': j
                }
                
                success = self.deployment_system.add_data(data)
                if not success:
                    logger.warning(f"数据发送失败: batch {i}, sample {j}")
            
            if i % 5 == 0:
                logger.info(f"处理批次 {i}/{num_batches}, 样本数: {len(features)}")
        
        logger.info(f"数据处理完成: {num_batches} 个批次")
    
    def _test_performance_monitoring(self):
        """测试性能监控"""
        logger.info("开始测试性能监控...")
        
        # 等待一段时间让系统运行
        time.sleep(5)
        
        # 获取系统状态
        status = self.deployment_system.get_system_status()
        
        logger.info("系统状态:")
        logger.info(f"  运行状态: {status['is_running']}")
        logger.info(f"  线程数: {status['threads_count']}")
        logger.info(f"  数据队列大小: {status['queue_sizes']['data_queue']}")
        logger.info(f"  结果队列大小: {status['queue_sizes']['result_queue']}")
        
        # 获取性能指标
        performance = status.get('performance', {})
        if performance:
            logger.info("性能指标:")
            for key, value in performance.items():
                logger.info(f"  {key}: {value}")
        
        # 记录结果
        self.production_results.append({
            'timestamp': time.time(),
            'type': 'performance_monitoring',
            'status': status
        })
    
    def _test_alert_system(self):
        """测试告警系统"""
        logger.info("开始测试告警系统...")
        
        # 获取活跃告警
        alerts = self.deployment_system.alert_system.get_active_alerts()
        
        if alerts:
            logger.info(f"活跃告警数量: {len(alerts)}")
            for alert in alerts:
                logger.info(f"  告警: {alert['message']}")
        else:
            logger.info("无活跃告警")
        
        # 记录结果
        self.production_results.append({
            'timestamp': time.time(),
            'type': 'alert_system',
            'alerts': alerts
        })
    
    def _test_system_stability(self):
        """测试系统稳定性"""
        logger.info("开始测试系统稳定性...")
        
        # 持续运行一段时间
        test_duration = 10  # 10秒
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            # 获取系统状态
            status = self.deployment_system.get_system_status()
            
            # 检查系统健康状态
            if not status['is_running']:
                logger.error("系统意外停止")
                break
            
            # 检查队列状态
            queue_sizes = status['queue_sizes']
            if queue_sizes['data_queue'] > 900:
                logger.warning("数据队列接近满载")
            if queue_sizes['result_queue'] > 900:
                logger.warning("结果队列接近满载")
            
            time.sleep(1)
        
        logger.info("系统稳定性测试完成")
        
        # 记录结果
        self.production_results.append({
            'timestamp': time.time(),
            'type': 'stability_test',
            'duration': test_duration,
            'final_status': status
        })
    
    def _generate_production_report(self):
        """生成生产报告"""
        logger.info("=" * 60)
        logger.info("V11 Phase 4 生产部署报告")
        logger.info("=" * 60)
        
        # 系统状态统计
        total_tests = len(self.production_results)
        successful_tests = len([r for r in self.production_results if r.get('status', {}).get('is_running', False)])
        
        logger.info(f"生产测试统计:")
        logger.info(f"  总测试数: {total_tests}")
        logger.info(f"  成功测试数: {successful_tests}")
        logger.info(f"  成功率: {successful_tests/total_tests*100:.1f}%")
        
        # 性能分析
        performance_tests = [r for r in self.production_results if r['type'] == 'performance_monitoring']
        if performance_tests:
            latest_performance = performance_tests[-1]['status']
            logger.info("最新性能指标:")
            for key, value in latest_performance.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value}")
                elif isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        logger.info(f"    {sub_key}: {sub_value}")
        
        # 告警分析
        alert_tests = [r for r in self.production_results if r['type'] == 'alert_system']
        if alert_tests:
            total_alerts = sum(len(r['alerts']) for r in alert_tests)
            logger.info(f"总告警数: {total_alerts}")
        
        # 稳定性分析
        stability_tests = [r for r in self.production_results if r['type'] == 'stability_test']
        if stability_tests:
            logger.info(f"稳定性测试: {len(stability_tests)} 次")
        
        # 保存测试结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"v11_phase4_production_results_{timestamp}.json"
        
        test_results = {
            'timestamp': timestamp,
            'config': self.config,
            'production_results': self.production_results,
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': successful_tests/total_tests*100 if total_tests > 0 else 0
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"生产测试结果已保存到: {results_file}")
        logger.info("V11 Phase 4 生产部署测试完成！")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("V11 Phase 4 生产部署系统")
    logger.info("=" * 60)
    
    # 创建测试器
    tester = V11Phase4ProductionTester()
    
    # 运行测试
    tester.test_production_deployment()
    
    logger.info("V11 Phase 4 生产部署测试完成！")


if __name__ == "__main__":
    main()
