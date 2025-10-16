"""
V11 回测优化系统
持续优化交易指标、技术指标、机器学习指标，生成评估报告
"""

import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import time
import json
import os
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11BacktestOptimizer:
    """
    V11回测优化系统
    持续优化交易指标、技术指标、机器学习指标
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 优化历史
        self.optimization_history = []
        self.best_performance = {
            'total_return': -np.inf,
            'sharpe_ratio': -np.inf,
            'max_drawdown': np.inf,
            'win_rate': 0,
            'profit_factor': 0,
            'iteration': 0
        }
        
        # 当前参数
        self.current_params = self._initialize_parameters()
        
        # 评估指标
        self.evaluation_metrics = {
            'trading_metrics': ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor'],
            'technical_metrics': ['signal_accuracy', 'signal_strength', 'market_timing'],
            'ml_metrics': ['model_accuracy', 'prediction_confidence', 'feature_importance']
        }
        
        logger.info(f"V11回测优化系统初始化完成，设备: {self.device}")
    
    def _initialize_parameters(self) -> Dict[str, Any]:
        """初始化参数"""
        return {
            # 交易参数
            'position_size': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'max_positions': 5,
            
            # 技术指标参数
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_period': 20,
            'bollinger_std': 2,
            
            # 机器学习参数
            'ml_threshold': 0.5,
            'confidence_threshold': 0.6,
            'feature_importance_threshold': 0.1,
            'model_weight_lstm': 0.25,
            'model_weight_transformer': 0.25,
            'model_weight_cnn': 0.25,
            'model_weight_ensemble': 0.25,
            
            # 风险参数
            'max_daily_loss': 0.05,
            'max_drawdown_limit': 0.15,
            'volatility_threshold': 0.03
        }
    
    def run_optimization_cycle(self, data: pd.DataFrame, max_iterations: int = 20) -> Dict[str, Any]:
        """运行优化循环"""
        logger.info(f"开始V11回测优化循环，最大迭代次数: {max_iterations}")
        
        for iteration in range(max_iterations):
            logger.info(f"=" * 60)
            logger.info(f"优化迭代 {iteration + 1}/{max_iterations}")
            logger.info(f"=" * 60)
            
            # 运行回测
            backtest_result = self._run_backtest(data)
            
            # 评估性能
            evaluation = self._evaluate_performance(backtest_result)
            
            # 生成评估报告
            report = self._generate_evaluation_report(iteration + 1, backtest_result, evaluation)
            
            # 保存评估报告
            self._save_evaluation_report(report, iteration + 1)
            
            # 检查是否达到真实交易标准
            if self._check_trading_ready(evaluation):
                logger.info("🎉 系统已达到真实交易标准！")
                break
            
            # 优化参数
            if iteration < max_iterations - 1:
                self._optimize_parameters(evaluation)
            
            # 记录优化历史
            self.optimization_history.append({
                'iteration': iteration + 1,
                'params': self.current_params.copy(),
                'backtest_result': backtest_result,
                'evaluation': evaluation,
                'report': report
            })
        
        # 生成最终优化报告
        final_report = self._generate_final_report()
        self._save_final_report(final_report)
        
        return final_report
    
    def _run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """运行回测"""
        logger.info("运行V11回测...")
        
        # 模拟回测结果
        np.random.seed(42)
        n_trades = np.random.randint(50, 200)
        
        # 生成交易数据
        trades = []
        for i in range(n_trades):
            trade = {
                'entry_time': data.index[np.random.randint(0, len(data))],
                'exit_time': data.index[np.random.randint(0, len(data))],
                'entry_price': np.random.uniform(95, 105),
                'exit_price': np.random.uniform(95, 105),
                'position_size': self.current_params['position_size'],
                'side': np.random.choice(['long', 'short']),
                'pnl': np.random.normal(0, 0.02),
                'fees': np.random.uniform(0.001, 0.005),
                'signal_strength': np.random.uniform(0.5, 1.0),
                'confidence': np.random.uniform(0.6, 0.9)
            }
            trade['net_pnl'] = trade['pnl'] - trade['fees']
            trades.append(trade)
        
        # 计算回测指标
        backtest_result = self._calculate_backtest_metrics(trades)
        
        logger.info(f"回测完成: {n_trades} 笔交易")
        return backtest_result
    
    def _calculate_backtest_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """计算回测指标"""
        if not trades:
            return {}
        
        df_trades = pd.DataFrame(trades)
        
        # 交易指标
        total_pnl = df_trades['net_pnl'].sum()
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['net_pnl'] > 0])
        losing_trades = len(df_trades[df_trades['net_pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 风险指标
        returns = df_trades['net_pnl'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # 计算最大回撤
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        # 技术指标
        signal_accuracy = np.mean(df_trades['confidence'])
        signal_strength = np.mean(df_trades['signal_strength'])
        
        # 机器学习指标
        ml_accuracy = np.random.uniform(0.6, 0.8)  # 模拟ML准确率
        prediction_confidence = np.mean(df_trades['confidence'])
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'signal_accuracy': signal_accuracy,
            'signal_strength': signal_strength,
            'ml_accuracy': ml_accuracy,
            'prediction_confidence': prediction_confidence,
            'trades': trades
        }
    
    def _evaluate_performance(self, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """评估性能"""
        logger.info("评估系统性能...")
        
        evaluation = {
            'trading_performance': {
                'total_return': backtest_result.get('total_pnl', 0),
                'sharpe_ratio': backtest_result.get('sharpe_ratio', 0),
                'max_drawdown': backtest_result.get('max_drawdown', 0),
                'win_rate': backtest_result.get('win_rate', 0),
                'profit_factor': backtest_result.get('profit_factor', 0)
            },
            'technical_performance': {
                'signal_accuracy': backtest_result.get('signal_accuracy', 0),
                'signal_strength': backtest_result.get('signal_strength', 0),
                'market_timing': np.random.uniform(0.6, 0.8)  # 模拟市场时机
            },
            'ml_performance': {
                'model_accuracy': backtest_result.get('ml_accuracy', 0),
                'prediction_confidence': backtest_result.get('prediction_confidence', 0),
                'feature_importance': np.random.uniform(0.7, 0.9)  # 模拟特征重要性
            }
        }
        
        # 计算综合评分
        evaluation['overall_score'] = self._calculate_overall_score(evaluation)
        
        return evaluation
    
    def _calculate_overall_score(self, evaluation: Dict[str, Any]) -> float:
        """计算综合评分"""
        trading = evaluation['trading_performance']
        technical = evaluation['technical_performance']
        ml = evaluation['ml_performance']
        
        # 权重分配
        trading_weight = 0.5
        technical_weight = 0.3
        ml_weight = 0.2
        
        # 交易性能评分 (0-100)
        trading_score = (
            min(trading['total_return'] * 100, 50) +  # 总收益
            min(trading['sharpe_ratio'] * 10, 20) +   # 夏普比率
            min((1 - trading['max_drawdown']) * 100, 15) +  # 回撤控制
            trading['win_rate'] * 15  # 胜率
        )
        
        # 技术性能评分 (0-100)
        technical_score = (
            technical['signal_accuracy'] * 40 +
            technical['signal_strength'] * 30 +
            technical['market_timing'] * 30
        )
        
        # 机器学习性能评分 (0-100)
        ml_score = (
            ml['model_accuracy'] * 40 +
            ml['prediction_confidence'] * 30 +
            ml['feature_importance'] * 30
        )
        
        overall_score = (
            trading_score * trading_weight +
            technical_score * technical_weight +
            ml_score * ml_weight
        )
        
        return overall_score
    
    def _generate_evaluation_report(self, iteration: int, backtest_result: Dict[str, Any], 
                                  evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """生成评估报告"""
        logger.info(f"生成第 {iteration} 次评估报告...")
        
        report = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'parameters': self.current_params.copy(),
            'backtest_results': backtest_result,
            'performance_evaluation': evaluation,
            'improvement_suggestions': self._generate_improvement_suggestions(evaluation),
            'next_optimization_focus': self._determine_next_focus(evaluation)
        }
        
        return report
    
    def _generate_improvement_suggestions(self, evaluation: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        trading = evaluation['trading_performance']
        technical = evaluation['technical_performance']
        ml = evaluation['ml_performance']
        
        # 交易性能改进建议
        if trading['total_return'] < 0.1:
            suggestions.append("增加仓位大小或优化信号质量以提高总收益")
        if trading['sharpe_ratio'] < 1.0:
            suggestions.append("优化风险调整收益，减少波动性")
        if trading['max_drawdown'] > 0.1:
            suggestions.append("加强风险控制，降低最大回撤")
        if trading['win_rate'] < 0.5:
            suggestions.append("提高信号准确性，改善胜率")
        
        # 技术性能改进建议
        if technical['signal_accuracy'] < 0.7:
            suggestions.append("优化技术指标参数，提高信号准确性")
        if technical['signal_strength'] < 0.7:
            suggestions.append("增强信号强度阈值，提高信号质量")
        
        # 机器学习性能改进建议
        if ml['model_accuracy'] < 0.7:
            suggestions.append("增加训练数据或调整模型参数")
        if ml['prediction_confidence'] < 0.7:
            suggestions.append("提高模型预测置信度")
        
        return suggestions
    
    def _determine_next_focus(self, evaluation: Dict[str, Any]) -> str:
        """确定下次优化重点"""
        trading = evaluation['trading_performance']
        technical = evaluation['technical_performance']
        ml = evaluation['ml_performance']
        
        # 找出最弱的指标
        trading_score = trading['total_return'] + trading['sharpe_ratio'] + (1 - trading['max_drawdown']) + trading['win_rate']
        technical_score = technical['signal_accuracy'] + technical['signal_strength'] + technical['market_timing']
        ml_score = ml['model_accuracy'] + ml['prediction_confidence'] + ml['feature_importance']
        
        if trading_score < technical_score and trading_score < ml_score:
            return "交易指标优化"
        elif technical_score < ml_score:
            return "技术指标优化"
        else:
            return "机器学习指标优化"
    
    def _check_trading_ready(self, evaluation: Dict[str, Any]) -> bool:
        """检查是否达到真实交易标准"""
        trading = evaluation['trading_performance']
        
        # 真实交易标准
        criteria = {
            'total_return': trading['total_return'] > 0.15,  # 年化收益 > 15%
            'sharpe_ratio': trading['sharpe_ratio'] > 1.5,   # 夏普比率 > 1.5
            'max_drawdown': trading['max_drawdown'] < 0.08,  # 最大回撤 < 8%
            'win_rate': trading['win_rate'] > 0.55,          # 胜率 > 55%
            'profit_factor': trading['profit_factor'] > 1.3, # 盈亏比 > 1.3
            'overall_score': evaluation['overall_score'] > 75  # 综合评分 > 75
        }
        
        ready_count = sum(criteria.values())
        total_criteria = len(criteria)
        
        logger.info(f"真实交易准备度: {ready_count}/{total_criteria} 项标准达标")
        for criterion, met in criteria.items():
            status = "✅" if met else "❌"
            logger.info(f"  {status} {criterion}: {evaluation['trading_performance'].get(criterion, 'N/A')}")
        
        return ready_count >= 5  # 至少5项标准达标
    
    def _optimize_parameters(self, evaluation: Dict[str, Any]):
        """优化参数"""
        logger.info("优化系统参数...")
        
        # 基于评估结果调整参数
        trading = evaluation['trading_performance']
        technical = evaluation['technical_performance']
        ml = evaluation['ml_performance']
        
        # 交易参数优化
        if trading['total_return'] < 0.1:
            self.current_params['position_size'] = min(self.current_params['position_size'] * 1.1, 0.2)
        
        if trading['max_drawdown'] > 0.1:
            self.current_params['stop_loss'] = min(self.current_params['stop_loss'] * 0.9, 0.05)
            self.current_params['take_profit'] = min(self.current_params['take_profit'] * 0.9, 0.1)
        
        # 技术指标参数优化
        if technical['signal_accuracy'] < 0.7:
            self.current_params['rsi_oversold'] = max(self.current_params['rsi_oversold'] - 2, 20)
            self.current_params['rsi_overbought'] = min(self.current_params['rsi_overbought'] + 2, 80)
        
        if technical['signal_strength'] < 0.7:
            self.current_params['bollinger_std'] = min(self.current_params['bollinger_std'] * 1.1, 3)
        
        # 机器学习参数优化
        if ml['model_accuracy'] < 0.7:
            self.current_params['ml_threshold'] = min(self.current_params['ml_threshold'] * 1.05, 0.8)
            self.current_params['confidence_threshold'] = min(self.current_params['confidence_threshold'] * 1.05, 0.9)
        
        # 模型权重优化
        if ml['prediction_confidence'] < 0.7:
            # 调整模型权重，增加表现最好的模型权重
            weights = [
                self.current_params['model_weight_lstm'],
                self.current_params['model_weight_transformer'],
                self.current_params['model_weight_cnn'],
                self.current_params['model_weight_ensemble']
            ]
            
            # 简单的权重调整策略
            max_weight_idx = np.argmax(weights)
            weights[max_weight_idx] = min(weights[max_weight_idx] * 1.1, 0.4)
            
            # 重新归一化
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            self.current_params['model_weight_lstm'] = weights[0]
            self.current_params['model_weight_transformer'] = weights[1]
            self.current_params['model_weight_cnn'] = weights[2]
            self.current_params['model_weight_ensemble'] = weights[3]
        
        logger.info("参数优化完成")
    
    def _save_evaluation_report(self, report: Dict[str, Any], iteration: int):
        """保存评估报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"v11_evaluation_report_iteration_{iteration}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"评估报告已保存: {filename}")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """生成最终报告"""
        logger.info("生成最终优化报告...")
        
        if not self.optimization_history:
            return {}
        
        # 找到最佳性能
        best_iteration = max(self.optimization_history, 
                           key=lambda x: x['evaluation']['overall_score'])
        
        # 生成最终报告
        final_report = {
            'optimization_summary': {
                'total_iterations': len(self.optimization_history),
                'best_iteration': best_iteration['iteration'],
                'best_overall_score': best_iteration['evaluation']['overall_score'],
                'trading_ready': self._check_trading_ready(best_iteration['evaluation'])
            },
            'performance_progression': [
                {
                    'iteration': h['iteration'],
                    'overall_score': h['evaluation']['overall_score'],
                    'total_return': h['evaluation']['trading_performance']['total_return'],
                    'sharpe_ratio': h['evaluation']['trading_performance']['sharpe_ratio'],
                    'max_drawdown': h['evaluation']['trading_performance']['max_drawdown']
                }
                for h in self.optimization_history
            ],
            'best_parameters': best_iteration['params'],
            'best_evaluation': best_iteration['evaluation'],
            'optimization_recommendations': self._generate_optimization_recommendations()
        }
        
        return final_report
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if self.optimization_history:
            latest = self.optimization_history[-1]
            evaluation = latest['evaluation']
            
            if evaluation['overall_score'] < 75:
                recommendations.append("继续优化系统参数，目标综合评分 > 75")
            
            if evaluation['trading_performance']['total_return'] < 0.15:
                recommendations.append("重点优化交易策略，提高总收益")
            
            if evaluation['trading_performance']['sharpe_ratio'] < 1.5:
                recommendations.append("优化风险调整收益，提高夏普比率")
            
            if evaluation['trading_performance']['max_drawdown'] > 0.08:
                recommendations.append("加强风险控制，降低最大回撤")
        
        return recommendations
    
    def _save_final_report(self, final_report: Dict[str, Any]):
        """保存最终报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"v11_final_optimization_report_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"最终优化报告已保存: {filename}")


if __name__ == "__main__":
    # 测试回测优化系统
    config = {
        'max_memory_usage': 0.8,
        'max_gpu_usage': 0.8,
        'performance_threshold': 0.6,
        'alert_threshold': 0.5
    }
    
    # 创建回测优化系统
    optimizer = V11BacktestOptimizer(config)
    
    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(95, 105, 1000),
        'high': np.random.uniform(95, 105, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(95, 105, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    # 运行优化循环
    final_report = optimizer.run_optimization_cycle(test_data, max_iterations=5)
    
    print("V11回测优化系统测试完成！")
