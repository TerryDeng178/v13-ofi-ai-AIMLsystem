"""
V12 新鲜数据回测框架
确保每次回测都使用全新的数据，避免数据泄露和过拟合
"""

import numpy as np
import pandas as pd
import json
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable

# 导入V12组件
from .v12_realistic_data_simulator import V12RealisticDataSimulator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V12FreshDataBacktestFramework:
    """V12新鲜数据回测框架"""
    
    def __init__(self, config: Dict):
        """
        初始化回测框架
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.backtest_results_dir = config.get('backtest_results_dir', 'backtest_results')
        self.ensure_results_dir()
        
        # 数据生成配置
        self.data_config = config.get('data', {})
        self.backtest_config = config.get('backtest', {})
        
        logger.info("V12新鲜数据回测框架初始化完成")
    
    def ensure_results_dir(self):
        """确保结果目录存在"""
        os.makedirs(self.backtest_results_dir, exist_ok=True)
    
    def generate_fresh_data(self, custom_seed: int = None) -> pd.DataFrame:
        """
        生成全新的回测数据
        
        Args:
            custom_seed: 自定义随机种子，如果为None则使用时间戳
        
        Returns:
            全新的回测数据
        """
        if custom_seed is None:
            # 使用当前时间戳确保数据唯一性
            random_seed = int(time.time() * 1000000) % 1000000
        else:
            random_seed = custom_seed
        
        logger.info(f"生成全新数据 - 随机种子: {random_seed}")
        
        # 创建新的数据模拟器实例
        data_simulator = V12RealisticDataSimulator(seed=random_seed)
        
        # 生成完整数据集
        fresh_data = data_simulator.generate_complete_dataset()
        
        logger.info(f"数据生成完成 - 形状: {fresh_data.shape}")
        logger.info(f"价格范围: {fresh_data['price'].min():.2f} - {fresh_data['price'].max():.2f}")
        logger.info(f"OFI Z-score范围: {fresh_data['ofi_z'].min():.4f} - {fresh_data['ofi_z'].max():.4f}")
        
        return fresh_data
    
    def run_backtest(self, 
                    backtest_name: str,
                    signal_generator: Callable,
                    risk_manager: Any,
                    fresh_data: pd.DataFrame = None,
                    custom_seed: int = None) -> Dict:
        """
        运行回测
        
        Args:
            backtest_name: 回测名称
            signal_generator: 信号生成函数
            risk_manager: 风险管理器
            fresh_data: 自定义数据，如果为None则生成新数据
            custom_seed: 自定义随机种子
        
        Returns:
            回测结果
        """
        logger.info(f"开始运行回测: {backtest_name}")
        
        # 生成或使用提供的数据
        if fresh_data is None:
            backtest_data = self.generate_fresh_data(custom_seed)
        else:
            backtest_data = fresh_data.copy()
            logger.info("使用提供的自定义数据")
        
        # 运行回测逻辑
        results = self._execute_backtest_logic(
            backtest_name, signal_generator, risk_manager, backtest_data
        )
        
        # 保存结果
        self._save_backtest_results(backtest_name, results)
        
        return results
    
    def _execute_backtest_logic(self, 
                               backtest_name: str,
                               signal_generator: Callable,
                               risk_manager: Any,
                               backtest_data: pd.DataFrame) -> Dict:
        """执行回测逻辑"""
        
        logger.info("执行回测逻辑...")
        
        # 初始化统计
        trades = []
        signals_generated = 0
        signals_filtered = 0
        daily_trades = 0
        last_trade_day = None
        
        # 回测参数
        max_daily_trades = self.backtest_config.get('max_daily_trades', 50)
        
        # 遍历数据
        for i, row in backtest_data.iterrows():
            current_time = row['timestamp']
            current_day = current_time.date()
            
            # 重置日交易计数
            if last_trade_day != current_day:
                daily_trades = 0
                risk_manager.daily_pnl = 0
                last_trade_day = current_day
            
            # 检查日交易限制
            if daily_trades >= max_daily_trades:
                continue
            
            try:
                # 检查是否需要平仓
                if hasattr(risk_manager, 'should_close_position'):
                    if risk_manager.should_close_position(row['price']):
                        close_trade = risk_manager.close_position(row['price'], current_time)
                        if close_trade:
                            trades.append(close_trade)
                
                # 生成信号
                signal = signal_generator(row)
                
                if signal:
                    signals_generated += 1
                    
                    # 风险管理检查
                    if hasattr(risk_manager, 'can_open_position'):
                        if risk_manager.can_open_position(
                            signal.get('signal_quality', 0.5),
                            signal.get('ai_confidence', 0.5)
                        ):
                            signals_filtered += 1
                            
                            # 执行交易逻辑
                            trade_result = self._execute_trade(
                                signal, row, risk_manager, current_time
                            )
                            
                            if trade_result:
                                trades.append(trade_result)
                                daily_trades += 1
            
            except Exception as e:
                logger.error(f"处理第{i}行数据时出错: {e}")
                continue
        
        # 强制平仓所有未平仓位
        if hasattr(risk_manager, 'current_position') and risk_manager.current_position != 0:
            final_price = backtest_data.iloc[-1]['price']
            close_trade = risk_manager.close_position(final_price, backtest_data.iloc[-1]['timestamp'])
            if close_trade:
                trades.append(close_trade)
        
        # 计算回测结果
        results = self._calculate_backtest_results(
            trades, signals_generated, signals_filtered, backtest_data
        )
        
        results['backtest_name'] = backtest_name
        results['data_info'] = {
            'data_shape': backtest_data.shape,
            'price_range': [backtest_data['price'].min(), backtest_data['price'].max()],
            'ofi_z_range': [backtest_data['ofi_z'].min(), backtest_data['ofi_z'].max()],
            'cvd_z_range': [backtest_data['cvd_z'].min(), backtest_data['cvd_z'].max()]
        }
        
        return results
    
    def _execute_trade(self, signal: Dict, row: pd.Series, risk_manager: Any, timestamp: datetime) -> Dict:
        """执行交易"""
        try:
            # 计算仓位大小
            if hasattr(risk_manager, 'calculate_position_size'):
                position_size = risk_manager.calculate_position_size(
                    signal.get('signal_quality', 0.5),
                    signal.get('ai_confidence', 0.5),
                    signal['price']
                )
            else:
                position_size = 1.0
            
            # 开仓
            if hasattr(risk_manager, 'open_position'):
                risk_manager.open_position(
                    signal['action'],
                    signal['price'],
                    position_size,
                    signal
                )
            
            # 记录交易
            trade_result = {
                'timestamp': timestamp,
                'action': signal['action'],
                'price': signal['price'],
                'quantity': position_size,
                'signal_quality': signal.get('signal_quality', 0.5),
                'ai_confidence': signal.get('ai_confidence', 0.5),
                'signal_strength': signal.get('signal_strength', 0.5),
                'trade_type': 'open'
            }
            
            # 添加其他信号信息
            for key in ['ofi_z', 'cvd_z', 'rsi', 'volatility']:
                if key in signal:
                    trade_result[key] = signal[key]
            
            return trade_result
            
        except Exception as e:
            logger.error(f"执行交易时出错: {e}")
            return None
    
    def _calculate_backtest_results(self, 
                                   trades: List[Dict], 
                                   signals_generated: int, 
                                   signals_filtered: int,
                                   backtest_data: pd.DataFrame) -> Dict:
        """计算回测结果"""
        
        if not trades:
            return {
                'timestamp': datetime.now().isoformat(),
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'signals_generated': signals_generated,
                'signals_filtered': signals_filtered,
                'data_points_processed': len(backtest_data)
            }
        
        trades_df = pd.DataFrame(trades)
        
        # 分离开仓和平仓交易
        open_trades = trades_df[trades_df['trade_type'] == 'open']
        close_trades = trades_df[trades_df['trade_type'].isnull()]
        
        # 计算PnL
        total_pnl = 0
        winning_trades = 0
        
        if len(close_trades) > 0:
            total_pnl = close_trades['pnl'].sum()
            winning_trades = len(close_trades[close_trades['pnl'] > 0])
        
        win_rate = (winning_trades / max(len(close_trades), 1)) * 100
        
        # 计算风险指标
        if len(close_trades) > 0:
            returns = close_trades['pnl'].values
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            max_drawdown = np.min(np.cumsum(returns) - np.maximum.accumulate(np.cumsum(returns)))
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # 计算平均指标
        avg_signal_quality = open_trades['signal_quality'].mean() if len(open_trades) > 0 else 0
        avg_ai_confidence = open_trades['ai_confidence'].mean() if len(open_trades) > 0 else 0
        avg_signal_strength = open_trades['signal_strength'].mean() if len(open_trades) > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_trades': len(trades),
            'open_trades': len(open_trades),
            'close_trades': len(close_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_signal_quality': avg_signal_quality,
            'avg_ai_confidence': avg_ai_confidence,
            'avg_signal_strength': avg_signal_strength,
            'signals_generated': signals_generated,
            'signals_filtered': signals_filtered,
            'signal_generation_rate': signals_generated / len(backtest_data) * 100,
            'signal_filter_rate': signals_filtered / max(signals_generated, 1) * 100,
            'data_points_processed': len(backtest_data)
        }
    
    def _save_backtest_results(self, backtest_name: str, results: Dict):
        """保存回测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.backtest_results_dir}/{backtest_name}_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"回测结果已保存: {report_file}")
    
    def run_multiple_backtests(self, 
                              backtest_configs: List[Dict],
                              num_iterations: int = 3) -> Dict:
        """
        运行多次回测，每次使用新数据
        
        Args:
            backtest_configs: 回测配置列表
            num_iterations: 迭代次数
        
        Returns:
            汇总结果
        """
        logger.info(f"开始运行{num_iterations}次回测，每次使用新数据")
        
        all_results = []
        
        for iteration in range(num_iterations):
            logger.info(f"第{iteration + 1}次回测开始")
            
            iteration_results = {}
            
            for config in backtest_configs:
                backtest_name = f"{config['name']}_iter_{iteration + 1}"
                
                # 每次使用新的随机种子
                fresh_data = self.generate_fresh_data()
                
                results = self.run_backtest(
                    backtest_name=backtest_name,
                    signal_generator=config['signal_generator'],
                    risk_manager=config['risk_manager'],
                    fresh_data=fresh_data
                )
                
                iteration_results[config['name']] = results
            
            all_results.append(iteration_results)
        
        # 计算汇总统计
        summary = self._calculate_summary_statistics(all_results)
        
        return {
            'individual_results': all_results,
            'summary_statistics': summary,
            'total_iterations': num_iterations
        }
    
    def _calculate_summary_statistics(self, all_results: List[Dict]) -> Dict:
        """计算汇总统计"""
        summary = {}
        
        # 获取所有回测名称
        all_names = set()
        for iteration_results in all_results:
            all_names.update(iteration_results.keys())
        
        for name in all_names:
            name_results = []
            for iteration_results in all_results:
                if name in iteration_results:
                    name_results.append(iteration_results[name])
            
            if name_results:
                summary[name] = {
                    'avg_win_rate': np.mean([r['win_rate'] for r in name_results]),
                    'avg_total_pnl': np.mean([r['total_pnl'] for r in name_results]),
                    'avg_sharpe_ratio': np.mean([r['sharpe_ratio'] for r in name_results]),
                    'avg_max_drawdown': np.mean([r['max_drawdown'] for r in name_results]),
                    'avg_trades': np.mean([r['total_trades'] for r in name_results]),
                    'std_win_rate': np.std([r['win_rate'] for r in name_results]),
                    'std_total_pnl': np.std([r['total_pnl'] for r in name_results]),
                    'consistency_score': 1.0 - np.std([r['win_rate'] for r in name_results]) / 100.0
                }
        
        return summary
