"""
V11 回测优化系统测试
持续优化交易指标、技术指标、机器学习指标
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
from src.v11_backtest_optimizer import V11BacktestOptimizer
from src.v11_advanced_features import V11AdvancedFeatureEngine
from src.v11_deep_learning import V11DeepLearning
from src.v11_signal_optimizer import V11SignalOptimizer
from src.v11_risk_manager import V11RiskManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11BacktestOptimizationTester:
    """V11回测优化系统测试器"""
    
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
        self.backtest_optimizer = V11BacktestOptimizer(self.config)
        
        # 测试数据
        self.test_data = None
        self.optimization_results = []
        
        logger.info("V11回测优化系统测试器初始化完成")
    
    def generate_realistic_market_data(self, num_samples: int = 2000) -> pd.DataFrame:
        """生成真实的模拟市场数据"""
        logger.info(f"生成 {num_samples} 条真实模拟市场数据...")
        
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=num_samples, freq='1min')
        
        # 生成更真实的价格数据
        price_base = 100.0
        prices = [price_base]
        volatility = 0.02
        
        for i in range(1, num_samples):
            # 添加趋势和波动
            trend = np.sin(i * 0.001) * 0.001  # 长期趋势
            noise = np.random.normal(0, volatility)
            price_change = trend + noise
            
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 50))  # 防止价格过低
        
        # 生成OHLCV数据
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'close': prices
        })
        
        # 生成高低价
        df['high'] = df['close'] * (1 + np.abs(np.random.normal(0, 0.005, num_samples)))
        df['low'] = df['close'] * (1 - np.abs(np.random.normal(0, 0.005, num_samples)))
        
        # 确保OHLC逻辑正确
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        # 生成成交量
        df['volume'] = np.random.lognormal(8, 0.5, num_samples)
        
        self.test_data = df
        logger.info(f"真实模拟市场数据生成完成: {len(df)} 条记录")
        logger.info(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
        logger.info(f"平均成交量: {df['volume'].mean():.0f}")
        
        return df
    
    def test_backtest_optimization_system(self):
        """测试回测优化系统"""
        logger.info("=" * 80)
        logger.info("V11 回测优化系统测试")
        logger.info("=" * 80)
        
        # 生成测试数据
        df = self.generate_realistic_market_data(2000)
        
        # 特征工程
        logger.info("步骤1: 特征工程...")
        df_features = self.feature_engineer.create_all_features(df)
        logger.info(f"特征工程完成: {df_features.shape[1]} 个特征")
        
        # 运行回测优化循环
        logger.info("步骤2: 运行回测优化循环...")
        final_report = self.backtest_optimizer.run_optimization_cycle(
            df_features, 
            max_iterations=10
        )
        
        # 分析优化结果
        logger.info("步骤3: 分析优化结果...")
        self._analyze_optimization_results(final_report)
        
        # 生成优化建议
        logger.info("步骤4: 生成优化建议...")
        self._generate_optimization_recommendations(final_report)
        
        # 保存测试结果
        self._save_test_results(final_report)
        
        logger.info("V11回测优化系统测试完成！")
    
    def _analyze_optimization_results(self, final_report: Dict[str, Any]):
        """分析优化结果"""
        logger.info("=" * 60)
        logger.info("优化结果分析")
        logger.info("=" * 60)
        
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
        
        if progression:
            logger.info("性能进展:")
            for i, perf in enumerate(progression):
                logger.info(f"  迭代 {perf['iteration']}: "
                          f"评分={perf['overall_score']:.2f}, "
                          f"收益={perf['total_return']:.3f}, "
                          f"夏普={perf['sharpe_ratio']:.2f}, "
                          f"回撤={perf['max_drawdown']:.3f}")
        
        # 最佳参数分析
        best_params = final_report.get('best_parameters', {})
        if best_params:
            logger.info("最佳参数:")
            logger.info(f"  仓位大小: {best_params.get('position_size', 0):.3f}")
            logger.info(f"  止损: {best_params.get('stop_loss', 0):.3f}")
            logger.info(f"  止盈: {best_params.get('take_profit', 0):.3f}")
            logger.info(f"  RSI超卖: {best_params.get('rsi_oversold', 0)}")
            logger.info(f"  RSI超买: {best_params.get('rsi_overbought', 0)}")
            logger.info(f"  ML阈值: {best_params.get('ml_threshold', 0):.3f}")
            logger.info(f"  置信度阈值: {best_params.get('confidence_threshold', 0):.3f}")
    
    def _generate_optimization_recommendations(self, final_report: Dict[str, Any]):
        """生成优化建议"""
        logger.info("=" * 60)
        logger.info("优化建议")
        logger.info("=" * 60)
        
        recommendations = final_report.get('optimization_recommendations', [])
        
        if recommendations:
            logger.info("系统建议:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
        else:
            logger.info("暂无优化建议")
        
        # 基于最终结果生成额外建议
        summary = final_report.get('optimization_summary', {})
        if summary.get('best_overall_score', 0) < 75:
            logger.info("额外建议:")
            logger.info("  1. 继续增加优化迭代次数")
            logger.info("  2. 调整参数搜索范围")
            logger.info("  3. 考虑增加更多特征")
            logger.info("  4. 优化模型架构")
        
        if not summary.get('trading_ready', False):
            logger.info("交易准备建议:")
            logger.info("  1. 重点优化风险控制")
            logger.info("  2. 提高信号质量")
            logger.info("  3. 增强模型稳定性")
            logger.info("  4. 进行更长时间的回测验证")
    
    def _save_test_results(self, final_report: Dict[str, Any]):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"v11_backtest_optimization_results_{timestamp}.json"
        
        test_results = {
            'timestamp': timestamp,
            'config': self.config,
            'test_data_info': {
                'samples': len(self.test_data) if self.test_data is not None else 0,
                'features': self.test_data.shape[1] if self.test_data is not None else 0
            },
            'optimization_results': final_report,
            'summary': {
                'total_iterations': final_report.get('optimization_summary', {}).get('total_iterations', 0),
                'best_score': final_report.get('optimization_summary', {}).get('best_overall_score', 0),
                'trading_ready': final_report.get('optimization_summary', {}).get('trading_ready', False)
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"测试结果已保存到: {results_file}")
    
    def run_continuous_optimization(self, max_cycles: int = 3):
        """运行持续优化"""
        logger.info("=" * 80)
        logger.info(f"V11 持续优化测试 (最大 {max_cycles} 轮)")
        logger.info("=" * 80)
        
        all_results = []
        
        for cycle in range(max_cycles):
            logger.info(f"开始第 {cycle + 1} 轮优化...")
            
            # 生成新的测试数据
            df = self.generate_realistic_market_data(2000 + cycle * 500)
            
            # 特征工程
            df_features = self.feature_engineer.create_all_features(df)
            
            # 运行优化
            final_report = self.backtest_optimizer.run_optimization_cycle(
                df_features, 
                max_iterations=8
            )
            
            all_results.append({
                'cycle': cycle + 1,
                'report': final_report
            })
            
            # 分析结果
            summary = final_report.get('optimization_summary', {})
            logger.info(f"第 {cycle + 1} 轮结果:")
            logger.info(f"  最佳评分: {summary.get('best_overall_score', 0):.2f}")
            logger.info(f"  交易准备: {'✅' if summary.get('trading_ready', False) else '❌'}")
            
            # 如果达到交易标准，可以提前结束
            if summary.get('trading_ready', False):
                logger.info("🎉 已达到交易标准，提前结束优化！")
                break
        
        # 保存持续优化结果
        self._save_continuous_optimization_results(all_results)
        
        return all_results


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("V11 回测优化系统")
    logger.info("=" * 80)
    
    # 创建测试器
    tester = V11BacktestOptimizationTester()
    
    # 运行单次优化测试
    tester.test_backtest_optimization_system()
    
    # 运行持续优化测试
    # tester.run_continuous_optimization(max_cycles=2)
    
    logger.info("V11回测优化系统测试完成！")


if __name__ == "__main__":
    main()
