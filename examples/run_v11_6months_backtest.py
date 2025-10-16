#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 6个月专业回测脚本
专门用于生成交易面评估报告
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import glob
from typing import Dict, List, Any
import json
import warnings
warnings.filterwarnings('ignore')

# V11模块导入
from src.v11_advanced_features import V11AdvancedFeatureEngine
from src.v11_deep_learning import V11DeepLearning
from src.v11_signal_optimizer import V11SignalOptimizer
from src.v11_risk_manager import V11RiskManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11_6MonthsBacktester:
    """V11 6个月专业回测器"""
    
    def __init__(self):
        self.data_dir = "data/binance"
        self.results_dir = "results/v11_6months_backtest"
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 初始化V11组件
        self.feature_engine = V11AdvancedFeatureEngine()
        self.deep_learning = V11DeepLearning()
        self.signal_optimizer = V11SignalOptimizer()
        self.risk_manager = V11RiskManager()
        
        # 回测参数
        self.backtest_params = {
            'initial_capital': 100000,  # 初始资金10万美元
            'commission_rate': 0.001,   # 手续费0.1%
            'slippage_rate': 0.0005,    # 滑点0.05%
            'max_position_size': 0.2,   # 最大仓位20%
            'stop_loss_rate': 0.02,     # 止损2%
            'take_profit_rate': 0.04,   # 止盈4%
        }
        
        logger.info("V11 6个月专业回测器初始化完成")
    
    def load_6months_data(self) -> pd.DataFrame:
        """加载6个月币安数据"""
        logger.info("加载6个月币安数据...")
        
        # 查找最新的6个月数据文件
        data_files = glob.glob(f"{self.data_dir}/ETHUSDT_1m_6months_*.csv")
        if not data_files:
            logger.error("未找到6个月币安数据文件")
            return pd.DataFrame()
        
        # 选择最新的文件
        latest_file = max(data_files, key=os.path.getctime)
        logger.info(f"使用数据文件: {latest_file}")
        
        # 加载数据
        df = pd.read_csv(latest_file)
        
        # 数据类型转换
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"数据加载完成: {len(df)} 条记录")
        logger.info(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        logger.info(f"价格范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备V11特征"""
        logger.info("准备V11高级特征...")
        
        # 创建V11特征
        df_features = self.feature_engine.create_all_features(df)
        
        logger.info(f"特征工程完成: {len(df_features.columns)} 个特征")
        
        return df_features
    
    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        logger.info("生成V11交易信号...")
        
        # 基础信号生成（简化版）
        df['price_change'] = df['close'].pct_change()
        df['price_ma_5'] = df['close'].rolling(5).mean()
        df['price_ma_20'] = df['close'].rolling(20).mean()
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # 生成买入信号
        buy_condition = (
            (df['close'] > df['price_ma_5']) & 
            (df['price_ma_5'] > df['price_ma_20']) & 
            (df['rsi_14'] < 70) & 
            (df['rsi_14'] > 30)
        )
        
        # 生成卖出信号
        sell_condition = (
            (df['close'] < df['price_ma_5']) & 
            (df['price_ma_5'] < df['price_ma_20']) & 
            (df['rsi_14'] > 70)
        )
        
        # 信号编码
        df['signal'] = 0
        df.loc[buy_condition, 'signal'] = 1  # 买入信号
        df.loc[sell_condition, 'signal'] = -1  # 卖出信号
        
        # 信号强度
        df['signal_strength'] = np.abs(df['rsi_14'] - 50) / 50
        
        logger.info(f"交易信号生成完成")
        logger.info(f"买入信号: {(df['signal'] == 1).sum()} 个")
        logger.info(f"卖出信号: {(df['signal'] == -1).sum()} 个")
        
        return df
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """运行回测"""
        logger.info("运行V11回测...")
        
        # 初始化回测变量
        capital = self.backtest_params['initial_capital']
        position = 0  # 当前仓位
        trades = []  # 交易记录
        portfolio_values = []  # 组合价值记录
        
        # 回测主循环
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            signal_strength = df['signal_strength'].iloc[i]
            
            # 计算当前组合价值
            portfolio_value = capital + position * current_price
            portfolio_values.append(portfolio_value)
            
            # 信号处理
            if signal == 1 and position == 0:  # 买入信号且无仓位
                # 计算仓位大小
                position_size = min(
                    self.backtest_params['max_position_size'],
                    signal_strength
                )
                
                # 计算交易数量
                trade_amount = capital * position_size
                position = trade_amount / current_price
                capital -= trade_amount
                
                # 记录交易
                trade = {
                    'timestamp': df['timestamp'].iloc[i],
                    'type': 'BUY',
                    'price': current_price,
                    'amount': trade_amount,
                    'position': position,
                    'signal_strength': signal_strength
                }
                trades.append(trade)
                
            elif signal == -1 and position > 0:  # 卖出信号且有仓位
                # 卖出所有仓位
                trade_amount = position * current_price
                capital += trade_amount
                
                # 记录交易
                trade = {
                    'timestamp': df['timestamp'].iloc[i],
                    'type': 'SELL',
                    'price': current_price,
                    'amount': trade_amount,
                    'position': 0,
                    'signal_strength': signal_strength
                }
                trades.append(trade)
                
                position = 0
            
            # 止损止盈检查
            if position > 0:
                # 计算当前收益率
                entry_price = trades[-1]['price'] if trades else current_price
                current_return = (current_price - entry_price) / entry_price
                
                # 止损
                if current_return <= -self.backtest_params['stop_loss_rate']:
                    trade_amount = position * current_price
                    capital += trade_amount
                    
                    trade = {
                        'timestamp': df['timestamp'].iloc[i],
                        'type': 'STOP_LOSS',
                        'price': current_price,
                        'amount': trade_amount,
                        'position': 0,
                        'signal_strength': signal_strength
                    }
                    trades.append(trade)
                    position = 0
                
                # 止盈
                elif current_return >= self.backtest_params['take_profit_rate']:
                    trade_amount = position * current_price
                    capital += trade_amount
                    
                    trade = {
                        'timestamp': df['timestamp'].iloc[i],
                        'type': 'TAKE_PROFIT',
                        'price': current_price,
                        'amount': trade_amount,
                        'position': 0,
                        'signal_strength': signal_strength
                    }
                    trades.append(trade)
                    position = 0
        
        # 计算最终组合价值
        final_price = df['close'].iloc[-1]
        final_portfolio_value = capital + position * final_price
        portfolio_values.append(final_portfolio_value)
        
        logger.info(f"回测完成: {len(trades)} 笔交易")
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'final_capital': capital,
            'final_position': position,
            'final_portfolio_value': final_portfolio_value
        }
    
    def calculate_performance_metrics(self, df: pd.DataFrame, backtest_results: Dict) -> Dict:
        """计算性能指标"""
        logger.info("计算性能指标...")
        
        trades = backtest_results['trades']
        portfolio_values = backtest_results['portfolio_values']
        
        # 基础收益指标
        initial_capital = self.backtest_params['initial_capital']
        final_value = backtest_results['final_portfolio_value']
        total_return = (final_value - initial_capital) / initial_capital
        
        # 时间跨度
        time_span = (df['timestamp'].max() - df['timestamp'].min()).days
        annual_return = (1 + total_return) ** (365 / time_span) - 1
        
        # 波动率计算
        returns = pd.Series(portfolio_values).pct_change().dropna()
        volatility = returns.std() * np.sqrt(365)  # 年化波动率
        
        # 夏普比率
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        peak = pd.Series(portfolio_values).expanding().max()
        drawdown = (pd.Series(portfolio_values) - peak) / peak
        max_drawdown = drawdown.min()
        
        # 交易统计
        if trades:
            # 计算每笔交易的收益
            trade_returns = []
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT']]
            
            for i, buy_trade in enumerate(buy_trades):
                if i < len(sell_trades):
                    sell_trade = sell_trades[i]
                    trade_return = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                    trade_returns.append(trade_return)
            
            win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) if trade_returns else 0
            avg_trade_return = np.mean(trade_returns) if trade_returns else 0
            profit_factor = sum([r for r in trade_returns if r > 0]) / abs(sum([r for r in trade_returns if r < 0])) if trade_returns else 0
        else:
            win_rate = 0
            avg_trade_return = 0
            profit_factor = 0
        
        # 性能指标
        performance_metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'avg_trade_return': avg_trade_return,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'time_span_days': time_span
        }
        
        logger.info(f"性能指标计算完成")
        return performance_metrics
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_trading_report(self, df: pd.DataFrame, backtest_results: Dict, performance_metrics: Dict) -> Dict:
        """生成交易报告"""
        logger.info("生成交易报告...")
        
        # 基础信息
        report = {
            'backtest_info': {
                'start_date': df['timestamp'].min().strftime('%Y-%m-%d'),
                'end_date': df['timestamp'].max().strftime('%Y-%m-%d'),
                'total_days': (df['timestamp'].max() - df['timestamp'].min()).days,
                'total_records': len(df),
                'data_source': 'Binance ETHUSDT 1m'
            },
            'market_analysis': {
                'start_price': df['close'].iloc[0],
                'end_price': df['close'].iloc[-1],
                'min_price': df['close'].min(),
                'max_price': df['close'].max(),
                'avg_price': df['close'].mean(),
                'price_volatility': df['close'].std(),
                'total_market_return': (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
            },
            'strategy_performance': performance_metrics,
            'trading_statistics': {
                'total_trades': len(backtest_results['trades']),
                'buy_trades': len([t for t in backtest_results['trades'] if t['type'] == 'BUY']),
                'sell_trades': len([t for t in backtest_results['trades'] if t['type'] == 'SELL']),
                'stop_loss_trades': len([t for t in backtest_results['trades'] if t['type'] == 'STOP_LOSS']),
                'take_profit_trades': len([t for t in backtest_results['trades'] if t['type'] == 'TAKE_PROFIT']),
                'avg_trade_size': np.mean([t['amount'] for t in backtest_results['trades']]) if backtest_results['trades'] else 0,
                'max_trade_size': max([t['amount'] for t in backtest_results['trades']]) if backtest_results['trades'] else 0,
                'min_trade_size': min([t['amount'] for t in backtest_results['trades']]) if backtest_results['trades'] else 0
            },
            'risk_metrics': {
                'max_drawdown': performance_metrics['max_drawdown'],
                'volatility': performance_metrics['volatility'],
                'sharpe_ratio': performance_metrics['sharpe_ratio'],
                'var_95': np.percentile([t.get('return', 0) for t in backtest_results['trades']], 5) if backtest_results['trades'] else 0,
                'cvar_95': np.mean([t.get('return', 0) for t in backtest_results['trades'] if t.get('return', 0) <= np.percentile([t.get('return', 0) for t in backtest_results['trades']], 5)]) if backtest_results['trades'] else 0
            },
            'recommendations': self._generate_recommendations(performance_metrics)
        }
        
        logger.info(f"交易报告生成完成")
        return report
    
    def _generate_recommendations(self, performance_metrics: Dict) -> List[str]:
        """生成交易建议"""
        recommendations = []
        
        # 基于夏普比率的建议
        if performance_metrics['sharpe_ratio'] > 1.5:
            recommendations.append("✅ 夏普比率优秀，策略风险调整后收益良好")
        elif performance_metrics['sharpe_ratio'] > 1.0:
            recommendations.append("🟡 夏普比率良好，策略表现中等")
        else:
            recommendations.append("🔴 夏普比率偏低，建议优化风险控制")
        
        # 基于最大回撤的建议
        if performance_metrics['max_drawdown'] > -0.1:
            recommendations.append("✅ 最大回撤控制良好，风险可控")
        elif performance_metrics['max_drawdown'] > -0.2:
            recommendations.append("🟡 最大回撤中等，需要改进风险控制")
        else:
            recommendations.append("🔴 最大回撤过大，需要严格控制仓位")
        
        # 基于胜率的建议
        if performance_metrics['win_rate'] > 0.6:
            recommendations.append("✅ 胜率优秀，信号质量良好")
        elif performance_metrics['win_rate'] > 0.5:
            recommendations.append("🟡 胜率中等，可以优化信号生成")
        else:
            recommendations.append("🔴 胜率偏低，建议重新审视信号策略")
        
        # 基于总收益的建议
        if performance_metrics['total_return'] > 0.2:
            recommendations.append("✅ 总收益优秀，策略表现良好")
        elif performance_metrics['total_return'] > 0:
            recommendations.append("🟡 总收益为正，策略基本有效")
        else:
            recommendations.append("🔴 总收益为负，需要重新评估策略")
        
        return recommendations
    
    def save_results(self, report: Dict):
        """保存结果"""
        logger.info("保存回测结果...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整报告
        report_file = f"{self.results_dir}/v11_6months_backtest_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"回测报告已保存到: {report_file}")
    
    def print_report_summary(self, report: Dict):
        """打印报告摘要"""
        logger.info("=" * 80)
        logger.info("V11 6个月回测交易面评估报告")
        logger.info("=" * 80)
        
        # 基础信息
        backtest_info = report['backtest_info']
        logger.info("📊 回测基础信息:")
        logger.info(f"  时间范围: {backtest_info['start_date']} ~ {backtest_info['end_date']}")
        logger.info(f"  回测天数: {backtest_info['total_days']} 天")
        logger.info(f"  数据记录: {backtest_info['total_records']} 条")
        logger.info(f"  数据源: {backtest_info['data_source']}")
        
        # 市场分析
        market_analysis = report['market_analysis']
        logger.info("\n📈 市场表现分析:")
        logger.info(f"  起始价格: ${market_analysis['start_price']:.2f}")
        logger.info(f"  结束价格: ${market_analysis['end_price']:.2f}")
        logger.info(f"  最高价格: ${market_analysis['max_price']:.2f}")
        logger.info(f"  最低价格: ${market_analysis['min_price']:.2f}")
        logger.info(f"  市场总收益: {market_analysis['total_market_return']:.2%}")
        
        # 策略表现
        strategy_performance = report['strategy_performance']
        logger.info("\n🎯 策略表现:")
        logger.info(f"  总收益率: {strategy_performance['total_return']:.2%}")
        logger.info(f"  年化收益率: {strategy_performance['annual_return']:.2%}")
        logger.info(f"  夏普比率: {strategy_performance['sharpe_ratio']:.2f}")
        logger.info(f"  最大回撤: {strategy_performance['max_drawdown']:.2%}")
        logger.info(f"  胜率: {strategy_performance['win_rate']:.2%}")
        logger.info(f"  总交易次数: {strategy_performance['total_trades']}")
        
        # 交易统计
        trading_stats = report['trading_statistics']
        logger.info("\n💼 交易统计:")
        logger.info(f"  买入交易: {trading_stats['buy_trades']} 笔")
        logger.info(f"  卖出交易: {trading_stats['sell_trades']} 笔")
        logger.info(f"  止损交易: {trading_stats['stop_loss_trades']} 笔")
        logger.info(f"  止盈交易: {trading_stats['take_profit_trades']} 笔")
        logger.info(f"  平均交易金额: ${trading_stats['avg_trade_size']:.2f}")
        
        # 风险指标
        risk_metrics = report['risk_metrics']
        logger.info("\n⚠️ 风险指标:")
        logger.info(f"  年化波动率: {risk_metrics['volatility']:.2%}")
        logger.info(f"  最大回撤: {risk_metrics['max_drawdown']:.2%}")
        logger.info(f"  夏普比率: {risk_metrics['sharpe_ratio']:.2f}")
        
        # 交易建议
        recommendations = report['recommendations']
        logger.info("\n💡 交易建议:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec}")
        
        logger.info("=" * 80)
    
    def run_full_backtest(self):
        """运行完整回测"""
        logger.info("=" * 80)
        logger.info("V11 6个月专业回测")
        logger.info("=" * 80)
        
        try:
            # 1. 加载数据
            df = self.load_6months_data()
            if df.empty:
                return False
            
            # 2. 准备特征
            df_features = self.prepare_features(df)
            
            # 3. 生成交易信号
            df_with_signals = self.generate_trading_signals(df_features)
            
            # 4. 运行回测
            backtest_results = self.run_backtest(df_with_signals)
            
            # 5. 计算性能指标
            performance_metrics = self.calculate_performance_metrics(df_with_signals, backtest_results)
            
            # 6. 生成交易报告
            report = self.generate_trading_report(df_with_signals, backtest_results, performance_metrics)
            
            # 7. 保存结果
            self.save_results(report)
            
            # 8. 打印报告摘要
            self.print_report_summary(report)
            
            logger.info("=" * 80)
            logger.info("✅ V11 6个月回测完成！")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"回测过程中出现错误: {e}")
            return False


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("V11 6个月专业回测系统")
    logger.info("=" * 80)
    
    # 创建回测器
    backtester = V11_6MonthsBacktester()
    
    # 运行完整回测
    success = backtester.run_full_backtest()
    
    if success:
        logger.info("🎉 V11 6个月回测成功完成！交易面评估报告已生成。")
    else:
        logger.error("❌ V11 6个月回测失败，请检查数据和配置。")


if __name__ == "__main__":
    main()
