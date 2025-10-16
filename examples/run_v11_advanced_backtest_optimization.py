"""
V11 高级回测优化系统测试
增强版回测优化，能够更好地优化到真实交易状态
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
from typing import Dict, List, Any

# 导入V11模块
from src.v11_advanced_backtest_optimizer import V11AdvancedBacktestOptimizer
from src.v11_advanced_features import V11AdvancedFeatureEngine
from src.v11_deep_learning import V11DeepLearning
from src.v11_signal_optimizer import V11SignalOptimizer
from src.v11_risk_manager import V11RiskManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11AdvancedBacktestOptimizationTester:
    """V11高级回测优化系统测试器"""
    
    def __init__(self):
        self.config = {
            'max_memory_usage': 0.8,
            'max_gpu_usage': 0.8,
            'performance_threshold': 0.6,
            'alert_threshold': 0.5,
            'feature_dim': 128,
            'sequence_length': 60,
            'optimization_strategy': 'adaptive'  # adaptive, grid_search, random_search
        }
        
        # 初始化组件
        self.feature_engineer = V11AdvancedFeatureEngine()
        self.ml_models = V11DeepLearning(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.signal_optimizer = V11SignalOptimizer()
        self.risk_manager = V11RiskManager()
        self.advanced_optimizer = V11AdvancedBacktestOptimizer(self.config)
        
        # 测试数据
        self.test_data = None
        self.optimization_results = []
        
        logger.info("V11高级回测优化系统测试器初始化完成")
    
    def generate_enhanced_market_data(self, num_samples: int = 3000) -> pd.DataFrame:
        """生成增强的模拟市场数据"""
        logger.info(f"生成 {num_samples} 条增强模拟市场数据...")
        
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=num_samples, freq='1min')
        
        # 生成更真实的趋势数据
        price_base = 100.0
        prices = [price_base]
        volatility = 0.015
        
        # 添加多个趋势周期
        for i in range(1, num_samples):
            # 短期趋势 (1小时周期)
            short_trend = np.sin(i * 0.01) * 0.002
            
            # 中期趋势 (4小时周期)
            medium_trend = np.sin(i * 0.0025) * 0.005
            
            # 长期趋势 (1天周期)
            long_trend = np.sin(i * 0.0005) * 0.008
            
            # 随机噪声
            noise = np.random.normal(0, volatility)
            
            # 合成价格变化
            price_change = short_trend + medium_trend + long_trend + noise
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 50))  # 防止价格过低
        
        # 生成OHLCV数据
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'close': prices
        })
        
        # 生成更真实的高低价
        high_multiplier = 1 + np.abs(np.random.normal(0, 0.003, num_samples))
        low_multiplier = 1 - np.abs(np.random.normal(0, 0.003, num_samples))
        
        df['high'] = df['close'] * high_multiplier
        df['low'] = df['close'] * low_multiplier
        
        # 确保OHLC逻辑正确
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        # 生成更真实的成交量
        base_volume = 5000
        volume_multiplier = 1 + np.sin(i * 0.01) * 0.5  # 成交量也有周期性
        df['volume'] = np.random.lognormal(np.log(base_volume), 0.3, num_samples) * volume_multiplier
        
        self.test_data = df
        logger.info(f"增强模拟市场数据生成完成: {len(df)} 条记录")
        logger.info(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
        logger.info(f"价格波动率: {df['close'].pct_change().std():.4f}")
        logger.info(f"平均成交量: {df['volume'].mean():.0f}")
        
        return df
    
    def test_advanced_backtest_optimization_system(self):
        """测试高级回测优化系统"""
        logger.info("=" * 80)
        logger.info("V11 高级回测优化系统测试")
        logger.info("=" * 80)
        
        # 生成测试数据
        df = self.generate_enhanced_market_data(3000)
        
        # 特征工程
        logger.info("步骤1: 特征工程...")
        df_features = self.feature_engineer.create_all_features(df)
        logger.info(f"特征工程完成: {df_features.shape[1]} 个特征")
        
        # 运行高级回测优化循环
        logger.info("步骤2: 运行高级回测优化循环...")
        final_report = self.advanced_optimizer.run_advanced_optimization_cycle(
            df_features, 
            max_iterations=20
        )
        
        # 分析优化结果
        logger.info("步骤3: 分析高级优化结果...")
        self._analyze_advanced_optimization_results(final_report)
        
        # 生成优化建议
        logger.info("步骤4: 生成高级优化建议...")
        self._generate_advanced_optimization_recommendations(final_report)
        
        # 保存测试结果
        self._save_advanced_test_results(final_report)
        
        logger.info("V11高级回测优化系统测试完成！")
    
    def _analyze_advanced_optimization_results(self, final_report: Dict[str, Any]):
        """分析高级优化结果"""
        logger.info("=" * 80)
        logger.info("高级优化结果分析")
        logger.info("=" * 80)
        
        if not final_report:
            logger.warning("没有优化结果可分析")
            return
        
        summary = final_report.get('optimization_summary', {})
        progression = final_report.get('performance_progression', [])
        
        logger.info(f"优化摘要:")
        logger.info(f"  总迭代次数: {summary.get('total_iterations', 0)}")
        logger.info(f"  最佳迭代: {summary.get('best_iteration', 0)}")
        logger.info(f"  最佳综合评分: {summary.get('best_overall_score', 0):.2f}")
        logger.info(f"  交易准备度: {'✅ 已准备' if summary.get('trading_ready', False) else '❌ 未准备'}")
        logger.info(f"  优化策略: {summary.get('optimization_strategy', 'unknown')}")
        
        if progression:
            logger.info("性能进展:")
            for i, perf in enumerate(progression):
                logger.info(f"  迭代 {perf['iteration']}: "
                          f"评分={perf['overall_score']:.2f}, "
                          f"年化收益={perf['total_return']:.1%}, "
                          f"夏普={perf['sharpe_ratio']:.2f}, "
                          f"回撤={perf['max_drawdown']:.1%}, "
                          f"胜率={perf['win_rate']:.1%}")
        
        # 最佳参数分析
        best_params = final_report.get('best_parameters', {})
        if best_params:
            logger.info("最佳参数:")
            logger.info(f"  仓位大小: {best_params.get('position_size', 0):.3f}")
            logger.info(f"  止损: {best_params.get('stop_loss', 0):.3f}")
            logger.info(f"  止盈: {best_params.get('take_profit', 0):.3f}")
            logger.info(f"  最大仓位: {best_params.get('max_positions', 0)}")
            logger.info(f"  RSI周期: {best_params.get('rsi_period', 0)}")
            logger.info(f"  RSI超卖: {best_params.get('rsi_oversold', 0)}")
            logger.info(f"  RSI超买: {best_params.get('rsi_overbought', 0)}")
            logger.info(f"  ML阈值: {best_params.get('ml_threshold', 0):.3f}")
            logger.info(f"  置信度阈值: {best_params.get('confidence_threshold', 0):.3f}")
            logger.info(f"  最大日损失: {best_params.get('max_daily_loss', 0):.1%}")
            logger.info(f"  最大回撤限制: {best_params.get('max_drawdown_limit', 0):.1%}")
        
        # 参数影响分析
        param_analysis = final_report.get('parameter_analysis', {})
        if param_analysis and not param_analysis.get('insufficient_data', False):
            logger.info("参数影响分析:")
            for param, analysis in param_analysis.items():
                if isinstance(analysis, dict):
                    correlation = analysis.get('correlation', 0)
                    trend = analysis.get('trend', 'unknown')
                    strength = analysis.get('strength', 0)
                    logger.info(f"  {param}: 相关性={correlation:.3f}, 趋势={trend}, 强度={strength:.3f}")
    
    def _generate_advanced_optimization_recommendations(self, final_report: Dict[str, Any]):
        """生成高级优化建议"""
        logger.info("=" * 80)
        logger.info("高级优化建议")
        logger.info("=" * 80)
        
        recommendations = final_report.get('optimization_recommendations', [])
        
        if recommendations:
            logger.info("系统建议:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
        else:
            logger.info("暂无优化建议")
        
        # 基于最终结果生成额外建议
        summary = final_report.get('optimization_summary', {})
        best_score = summary.get('best_overall_score', 0)
        
        if best_score < 75:
            logger.info("额外建议:")
            logger.info("  1. 继续增加优化迭代次数到50轮以上")
            logger.info("  2. 尝试不同的优化策略 (grid_search, random_search)")
            logger.info("  3. 扩大参数搜索范围")
            logger.info("  4. 考虑增加更多技术指标特征")
            logger.info("  5. 优化模型架构和超参数")
        
        if not summary.get('trading_ready', False):
            logger.info("交易准备建议:")
            logger.info("  1. 重点优化风险控制参数")
            logger.info("  2. 提高信号质量和准确性")
            logger.info("  3. 增强模型稳定性和鲁棒性")
            logger.info("  4. 进行更长时间的回测验证")
            logger.info("  5. 考虑多资产组合优化")
        
        # 基于参数影响分析的建议
        param_analysis = final_report.get('parameter_analysis', {})
        if param_analysis and not param_analysis.get('insufficient_data', False):
            logger.info("参数优化建议:")
            for param, analysis in param_analysis.items():
                if isinstance(analysis, dict):
                    correlation = analysis.get('correlation', 0)
                    if abs(correlation) > 0.3:
                        trend = analysis.get('trend', 'unknown')
                        if trend == 'positive':
                            logger.info(f"  - 增加 {param} 参数值可能提升性能")
                        elif trend == 'negative':
                            logger.info(f"  - 减少 {param} 参数值可能提升性能")
    
    def _save_advanced_test_results(self, final_report: Dict[str, Any]):
        """保存高级测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"v11_advanced_backtest_optimization_results_{timestamp}.json"
        
        test_results = {
            'timestamp': timestamp,
            'config': self.config,
            'test_data_info': {
                'samples': len(self.test_data) if self.test_data is not None else 0,
                'features': self.test_data.shape[1] if self.test_data is not None else 0,
                'price_range': {
                    'min': self.test_data['close'].min() if self.test_data is not None else 0,
                    'max': self.test_data['close'].max() if self.test_data is not None else 0
                },
                'volatility': self.test_data['close'].pct_change().std() if self.test_data is not None else 0
            },
            'optimization_results': final_report,
            'summary': {
                'total_iterations': final_report.get('optimization_summary', {}).get('total_iterations', 0),
                'best_score': final_report.get('optimization_summary', {}).get('best_overall_score', 0),
                'trading_ready': final_report.get('optimization_summary', {}).get('trading_ready', False),
                'optimization_strategy': final_report.get('optimization_summary', {}).get('optimization_strategy', 'unknown')
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"高级测试结果已保存到: {results_file}")
    
    def run_multi_strategy_optimization(self, strategies: List[str] = ['adaptive', 'random_search']):
        """运行多策略优化"""
        logger.info("=" * 80)
        logger.info(f"V11 多策略优化测试 (策略: {strategies})")
        logger.info("=" * 80)
        
        all_results = []
        
        for strategy in strategies:
            logger.info(f"开始 {strategy} 策略优化...")
            
            # 更新配置
            self.config['optimization_strategy'] = strategy
            self.advanced_optimizer = V11AdvancedBacktestOptimizer(self.config)
            
            # 生成测试数据
            df = self.generate_enhanced_market_data(2500)
            df_features = self.feature_engineer.create_all_features(df)
            
            # 运行优化
            final_report = self.advanced_optimizer.run_advanced_optimization_cycle(
                df_features, 
                max_iterations=15
            )
            
            all_results.append({
                'strategy': strategy,
                'report': final_report
            })
            
            # 分析结果
            summary = final_report.get('optimization_summary', {})
            logger.info(f"{strategy} 策略结果:")
            logger.info(f"  最佳评分: {summary.get('best_overall_score', 0):.2f}")
            logger.info(f"  交易准备: {'✅' if summary.get('trading_ready', False) else '❌'}")
            logger.info(f"  迭代次数: {summary.get('total_iterations', 0)}")
            
            # 如果达到交易标准，可以提前结束
            if summary.get('trading_ready', False):
                logger.info(f"🎉 {strategy} 策略已达到交易标准！")
                break
        
        # 保存多策略优化结果
        self._save_multi_strategy_results(all_results)
        
        return all_results
    
    def _save_multi_strategy_results(self, all_results: List[Dict[str, Any]]):
        """保存多策略优化结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"v11_multi_strategy_optimization_results_{timestamp}.json"
        
        # 找出最佳策略
        best_strategy = None
        best_score = -np.inf
        
        for result in all_results:
            summary = result['report'].get('optimization_summary', {})
            score = summary.get('best_overall_score', 0)
            if score > best_score:
                best_score = score
                best_strategy = result['strategy']
        
        test_results = {
            'timestamp': timestamp,
            'strategies_tested': [r['strategy'] for r in all_results],
            'best_strategy': best_strategy,
            'best_score': best_score,
            'results': all_results,
            'comparison': {
                'strategy_scores': {
                    r['strategy']: r['report'].get('optimization_summary', {}).get('best_overall_score', 0)
                    for r in all_results
                }
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"多策略优化结果已保存到: {results_file}")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("V11 高级回测优化系统")
    logger.info("=" * 80)
    
    # 创建测试器
    tester = V11AdvancedBacktestOptimizationTester()
    
    # 运行单策略优化测试
    # tester.test_advanced_backtest_optimization_system()
    
    # 运行多策略优化测试
    tester.run_multi_strategy_optimization(['adaptive', 'random_search'])
    
    logger.info("V11高级回测优化系统测试完成！")


if __name__ == "__main__":
    main()
