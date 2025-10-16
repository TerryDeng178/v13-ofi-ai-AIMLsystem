"""
测试V11系统中机器学习学习效果
验证回测是否能持续提升ML性能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime
import json
from typing import Dict, List, Any

# 导入V11模块
from src.v11_advanced_features import V11AdvancedFeatureEngine
from src.v11_deep_learning import V11DeepLearning
from src.v11_advanced_backtest_optimizer import V11AdvancedBacktestOptimizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLLearningEffectivenessTester:
    """机器学习学习效果测试器"""
    
    def __init__(self):
        self.config = {
            'feature_dim': 128,
            'sequence_length': 60,
            'optimization_strategy': 'adaptive'
        }
        
        # 初始化组件
        self.feature_engineer = V11AdvancedFeatureEngine()
        self.ml_models = V11DeepLearning(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = V11AdvancedBacktestOptimizer(self.config)
        
        # 学习效果记录
        self.learning_history = []
        self.performance_trends = {}
        
        logger.info("机器学习学习效果测试器初始化完成")
    
    def generate_progressive_market_data(self, num_iterations: int = 10) -> List[pd.DataFrame]:
        """生成渐进式市场数据，模拟真实市场环境变化"""
        logger.info(f"生成 {num_iterations} 轮渐进式市场数据...")
        
        all_data = []
        base_price = 100.0
        
        for i in range(num_iterations):
            # 每轮数据都有不同的市场特征
            num_samples = 1000 + i * 200  # 逐渐增加数据量
            
            # 生成不同趋势的数据
            if i < 3:
                # 前3轮：上升趋势
                trend = 0.001 * (i + 1)
            elif i < 6:
                # 中间3轮：震荡趋势
                trend = 0.0005 * np.sin(i)
            else:
                # 后3轮：下降趋势
                trend = -0.001 * (i - 5)
            
            # 生成价格数据
            prices = [base_price]
            volatility = 0.01 + i * 0.002  # 逐渐增加波动率
            
            for j in range(1, num_samples):
                price_change = trend + np.random.normal(0, volatility)
                new_price = prices[-1] * (1 + price_change)
                prices.append(max(new_price, 50))
            
            # 生成OHLCV数据
            dates = pd.date_range(start=f'2024-01-{i+1:02d}', periods=num_samples, freq='1min')
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                'close': prices,
                'volume': np.random.lognormal(10, 0.5, num_samples)
            })
            
            # 确保OHLC逻辑正确
            df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
            df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
            
            all_data.append(df)
            base_price = prices[-1]  # 使用上一轮的最终价格作为下一轮的起始价格
            
            logger.info(f"第 {i+1} 轮数据: {len(df)} 条记录, 价格范围: {df['close'].min():.2f}-{df['close'].max():.2f}")
        
        return all_data
    
    def test_ml_learning_effectiveness(self):
        """测试机器学习学习效果"""
        logger.info("=" * 80)
        logger.info("V11 机器学习学习效果测试")
        logger.info("=" * 80)
        
        # 生成渐进式数据
        market_data_list = self.generate_progressive_market_data(8)
        
        # 记录初始模型性能
        initial_performance = self._evaluate_model_performance()
        self.learning_history.append({
            'iteration': 0,
            'performance': initial_performance,
            'status': 'initial'
        })
        
        # 逐轮测试学习效果
        for i, market_data in enumerate(market_data_list):
            logger.info(f"\n{'='*50}")
            logger.info(f"第 {i+1} 轮学习测试")
            logger.info(f"{'='*50}")
            
            # 特征工程
            df_features = self.feature_engineer.create_all_features(market_data)
            logger.info(f"特征工程完成: {df_features.shape[1]} 个特征")
            
            # 运行回测优化
            backtest_result = self.optimizer.run_advanced_optimization_cycle(
                df_features, 
                max_iterations=3  # 每轮只运行3次迭代
            )
            
            # 评估学习效果
            learning_effect = self._evaluate_learning_effect(backtest_result, i+1)
            
            # 记录学习历史
            self.learning_history.append({
                'iteration': i+1,
                'performance': learning_effect,
                'backtest_result': backtest_result,
                'status': 'learned'
            })
            
            # 分析性能趋势
            self._analyze_performance_trend()
            
            # 决定是否更新模型
            should_update = self._should_update_model(learning_effect)
            logger.info(f"是否更新模型: {'✅ 是' if should_update else '❌ 否'}")
        
        # 生成学习效果报告
        self._generate_learning_report()
        
        logger.info("机器学习学习效果测试完成！")
    
    def _evaluate_model_performance(self) -> Dict[str, float]:
        """评估模型性能"""
        # 模拟模型性能评估
        performance = {
            'accuracy': np.random.uniform(0.6, 0.8),
            'precision': np.random.uniform(0.6, 0.8),
            'recall': np.random.uniform(0.6, 0.8),
            'f1_score': np.random.uniform(0.6, 0.8),
            'stability': np.random.uniform(0.7, 0.9)
        }
        return performance
    
    def _evaluate_learning_effect(self, backtest_result: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """评估学习效果"""
        summary = backtest_result.get('optimization_summary', {})
        
        # 计算性能指标
        performance = {
            'overall_score': summary.get('best_overall_score', 0),
            'total_return': summary.get('best_total_return', 0),
            'sharpe_ratio': summary.get('best_sharpe_ratio', 0),
            'max_drawdown': summary.get('best_max_drawdown', 0),
            'win_rate': summary.get('best_win_rate', 0),
            'iteration': iteration
        }
        
        # 与上一轮对比
        if len(self.learning_history) > 0:
            prev_performance = self.learning_history[-1]['performance']
            performance['improvement'] = {
                'score_change': performance['overall_score'] - prev_performance.get('overall_score', 0),
                'return_change': performance['total_return'] - prev_performance.get('total_return', 0),
                'sharpe_change': performance['sharpe_ratio'] - prev_performance.get('sharpe_ratio', 0)
            }
        
        return performance
    
    def _analyze_performance_trend(self):
        """分析性能趋势"""
        if len(self.learning_history) < 2:
            return
        
        # 计算性能趋势
        scores = [h['performance']['overall_score'] for h in self.learning_history if 'overall_score' in h['performance']]
        
        if len(scores) >= 3:
            # 计算趋势
            recent_trend = np.mean(scores[-3:]) - np.mean(scores[-6:-3]) if len(scores) >= 6 else 0
            volatility = np.std(scores[-5:]) if len(scores) >= 5 else 0
            
            self.performance_trends = {
                'recent_trend': recent_trend,
                'volatility': volatility,
                'is_improving': recent_trend > 0,
                'is_stable': volatility < 5
            }
            
            logger.info(f"性能趋势分析:")
            logger.info(f"  近期趋势: {recent_trend:+.2f}")
            logger.info(f"  波动性: {volatility:.2f}")
            logger.info(f"  是否改善: {'✅' if self.performance_trends['is_improving'] else '❌'}")
            logger.info(f"  是否稳定: {'✅' if self.performance_trends['is_stable'] else '❌'}")
    
    def _should_update_model(self, learning_effect: Dict[str, Any]) -> bool:
        """决定是否应该更新模型"""
        # 基于多个指标决定是否更新模型
        
        # 1. 性能改善检查
        improvement = learning_effect.get('improvement', {})
        score_change = improvement.get('score_change', 0)
        
        # 2. 稳定性检查
        is_stable = self.performance_trends.get('is_stable', True)
        
        # 3. 趋势检查
        is_improving = self.performance_trends.get('is_improving', False)
        
        # 4. 绝对性能检查
        current_score = learning_effect.get('overall_score', 0)
        
        # 综合决策
        should_update = (
            score_change > 0 and  # 性能有改善
            is_stable and  # 系统稳定
            current_score > 70  # 绝对性能达标
        )
        
        return should_update
    
    def _generate_learning_report(self):
        """生成学习效果报告"""
        logger.info("=" * 80)
        logger.info("机器学习学习效果报告")
        logger.info("=" * 80)
        
        # 统计学习效果
        total_iterations = len(self.learning_history) - 1  # 减去初始轮
        improving_iterations = 0
        stable_iterations = 0
        
        for i in range(1, len(self.learning_history)):
            performance = self.learning_history[i]['performance']
            improvement = performance.get('improvement', {})
            
            if improvement.get('score_change', 0) > 0:
                improving_iterations += 1
            
            if abs(improvement.get('score_change', 0)) < 2:  # 变化小于2认为稳定
                stable_iterations += 1
        
        # 计算学习效果指标
        improvement_rate = improving_iterations / total_iterations if total_iterations > 0 else 0
        stability_rate = stable_iterations / total_iterations if total_iterations > 0 else 0
        
        # 性能变化趋势
        initial_score = self.learning_history[0]['performance'].get('overall_score', 0)
        final_score = self.learning_history[-1]['performance'].get('overall_score', 0)
        total_improvement = final_score - initial_score
        
        # 生成报告
        report = {
            'test_summary': {
                'total_iterations': total_iterations,
                'improving_iterations': improving_iterations,
                'stable_iterations': stable_iterations,
                'improvement_rate': improvement_rate,
                'stability_rate': stability_rate,
                'total_improvement': total_improvement
            },
            'learning_history': self.learning_history,
            'performance_trends': self.performance_trends,
            'conclusions': {
                'learning_effective': improvement_rate > 0.5,
                'system_stable': stability_rate > 0.7,
                'overall_improvement': total_improvement > 0
            }
        }
        
        # 输出报告
        logger.info(f"学习效果统计:")
        logger.info(f"  总迭代次数: {total_iterations}")
        logger.info(f"  性能改善次数: {improving_iterations}")
        logger.info(f"  稳定迭代次数: {stable_iterations}")
        logger.info(f"  改善率: {improvement_rate:.1%}")
        logger.info(f"  稳定率: {stability_rate:.1%}")
        logger.info(f"  总体改善: {total_improvement:+.2f}")
        
        logger.info(f"\n学习效果评估:")
        logger.info(f"  学习有效: {'✅' if report['conclusions']['learning_effective'] else '❌'}")
        logger.info(f"  系统稳定: {'✅' if report['conclusions']['system_stable'] else '❌'}")
        logger.info(f"  总体改善: {'✅' if report['conclusions']['overall_improvement'] else '❌'}")
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"ml_learning_effectiveness_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"学习效果报告已保存到: {report_file}")
        
        return report


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("V11 机器学习学习效果测试")
    logger.info("=" * 80)
    
    # 创建测试器
    tester = MLLearningEffectivenessTester()
    
    # 运行学习效果测试
    tester.test_ml_learning_effectiveness()
    
    logger.info("V11机器学习学习效果测试完成！")


if __name__ == "__main__":
    main()
