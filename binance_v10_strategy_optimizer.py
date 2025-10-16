#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安V10策略优化器
实现信号过滤、参数调优、回测验证功能
"""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
from collections import deque
import json
from datetime import datetime, timedelta

from binance_v10_simple_advanced import BinanceV10SimpleAdvanced

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SignalFilter:
    """信号过滤器配置"""
    ofi_threshold: float = 1.0  # OFI Z-score阈值
    momentum_threshold: float = 0.001  # 价格动量阈值
    volume_threshold: float = 100.0  # 成交量阈值
    time_filter: bool = True  # 时间过滤
    quality_score_threshold: float = 0.5  # 质量评分阈值

@dataclass
class StrategyParams:
    """策略参数配置"""
    # 信号参数
    ofi_threshold: float = 1.0
    momentum_threshold: float = 0.001
    volume_threshold: float = 100.0
    
    # 风险参数
    max_position_size: float = 1.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    
    # 交易参数
    min_signal_strength: float = 1.5
    max_trades_per_hour: int = 10
    cooldown_period: int = 60  # 秒

@dataclass
class BacktestResult:
    """回测结果"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_duration: float
    best_trade: float
    worst_trade: float

class SignalOptimizer:
    """信号优化器"""
    
    def __init__(self):
        self.signal_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=100)
        self.optimization_results = []
        
    def add_signal(self, signal_data: Dict):
        """添加信号数据"""
        self.signal_history.append({
            'timestamp': time.time(),
            'ofi_zscore': signal_data.get('ofi_zscore', 0),
            'price': signal_data.get('price', 0),
            'volume': signal_data.get('volume', 0),
            'signal_strength': signal_data.get('signal_strength', 0),
            'quality_score': signal_data.get('quality_score', 0)
        })
    
    def calculate_signal_quality(self, signal_data: Dict) -> float:
        """计算信号质量评分"""
        ofi_score = min(abs(signal_data.get('ofi_zscore', 0)) / 3.0, 1.0)
        momentum_score = min(abs(signal_data.get('momentum', 0)) / 0.01, 1.0)
        volume_score = min(signal_data.get('volume', 0) / 1000.0, 1.0)
        
        # 加权平均
        quality_score = (
            ofi_score * 0.4 +
            momentum_score * 0.3 +
            volume_score * 0.3
        )
        
        return quality_score
    
    def filter_signals(self, signal_data: Dict, filter_config: SignalFilter) -> bool:
        """过滤信号"""
        # OFI阈值检查
        if abs(signal_data.get('ofi_zscore', 0)) < filter_config.ofi_threshold:
            return False
        
        # 动量阈值检查
        if abs(signal_data.get('momentum', 0)) < filter_config.momentum_threshold:
            return False
        
        # 成交量阈值检查
        if signal_data.get('volume', 0) < filter_config.volume_threshold:
            return False
        
        # 质量评分检查
        quality_score = self.calculate_signal_quality(signal_data)
        if quality_score < filter_config.quality_score_threshold:
            return False
        
        # 时间过滤（避免频繁交易）
        if filter_config.time_filter and len(self.signal_history) > 0:
            last_signal_time = self.signal_history[-1]['timestamp']
            if time.time() - last_signal_time < 60:  # 1分钟内不重复信号
                return False
        
        return True
    
    def optimize_parameters(self, historical_data: List[Dict]) -> StrategyParams:
        """参数优化"""
        logger.info("开始参数优化...")
        
        best_params = None
        best_performance = -float('inf')
        
        # 参数搜索空间
        ofi_thresholds = [0.5, 1.0, 1.5, 2.0]
        momentum_thresholds = [0.0005, 0.001, 0.002, 0.005]
        stop_loss_pcts = [0.01, 0.02, 0.03, 0.05]
        take_profit_pcts = [0.02, 0.04, 0.06, 0.08]
        
        for ofi_th in ofi_thresholds:
            for mom_th in momentum_thresholds:
                for sl_pct in stop_loss_pcts:
                    for tp_pct in take_profit_pcts:
                        params = StrategyParams(
                            ofi_threshold=ofi_th,
                            momentum_threshold=mom_th,
                            stop_loss_pct=sl_pct,
                            take_profit_pct=tp_pct
                        )
                        
                        # 回测参数
                        result = self.backtest_strategy(historical_data, params)
                        
                        # 计算综合评分
                        performance_score = (
                            result.win_rate * 0.3 +
                            result.sharpe_ratio * 0.3 +
                            (1 - result.max_drawdown) * 0.2 +
                            result.profit_factor * 0.2
                        )
                        
                        if performance_score > best_performance:
                            best_performance = performance_score
                            best_params = params
                        
                        logger.info(f"参数测试: OFI={ofi_th}, 动量={mom_th}, 止损={sl_pct}, 止盈={tp_pct}, 评分={performance_score:.3f}")
        
        logger.info(f"最佳参数: {best_params}")
        logger.info(f"最佳评分: {best_performance:.3f}")
        
        return best_params
    
    def backtest_strategy(self, historical_data: List[Dict], params: StrategyParams) -> BacktestResult:
        """策略回测"""
        trades = []
        position = None
        entry_price = 0
        entry_time = 0
        
        for i, data in enumerate(historical_data):
            current_price = data.get('price', 0)
            ofi_zscore = data.get('ofi_zscore', 0)
            momentum = data.get('momentum', 0)
            volume = data.get('volume', 0)
            
            # 信号检查
            signal_strength = abs(ofi_zscore)
            if signal_strength < params.ofi_threshold:
                continue
            
            if abs(momentum) < params.momentum_threshold:
                continue
            
            if volume < params.volume_threshold:
                continue
            
            # 开仓逻辑
            if position is None:
                if ofi_zscore > params.ofi_threshold:
                    # 买入信号
                    position = 'LONG'
                    entry_price = current_price
                    entry_time = i
                elif ofi_zscore < -params.ofi_threshold:
                    # 卖出信号
                    position = 'SHORT'
                    entry_price = current_price
                    entry_time = i
            
            # 平仓逻辑
            elif position == 'LONG':
                # 止损
                if current_price <= entry_price * (1 - params.stop_loss_pct):
                    pnl = (current_price - entry_price) / entry_price
                    trades.append({
                        'side': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'duration': i - entry_time
                    })
                    position = None
                # 止盈
                elif current_price >= entry_price * (1 + params.take_profit_pct):
                    pnl = (current_price - entry_price) / entry_price
                    trades.append({
                        'side': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'duration': i - entry_time
                    })
                    position = None
            
            elif position == 'SHORT':
                # 止损
                if current_price >= entry_price * (1 + params.stop_loss_pct):
                    pnl = (entry_price - current_price) / entry_price
                    trades.append({
                        'side': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'duration': i - entry_time
                    })
                    position = None
                # 止盈
                elif current_price <= entry_price * (1 - params.take_profit_pct):
                    pnl = (entry_price - current_price) / entry_price
                    trades.append({
                        'side': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'duration': i - entry_time
                    })
                    position = None
        
        # 计算回测结果
        return self._calculate_backtest_metrics(trades)
    
    def _calculate_backtest_metrics(self, trades: List[Dict]) -> BacktestResult:
        """计算回测指标"""
        if not trades:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in trades)
        
        # 计算最大回撤
        cumulative_pnl = np.cumsum([t['pnl'] for t in trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # 计算夏普比率
        returns = [t['pnl'] for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # 计算盈利因子
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # 其他指标
        avg_trade_duration = np.mean([t['duration'] for t in trades])
        best_trade = max(t['pnl'] for t in trades) if trades else 0
        worst_trade = min(t['pnl'] for t in trades) if trades else 0
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            avg_trade_duration=avg_trade_duration,
            best_trade=best_trade,
            worst_trade=worst_trade
        )

class BinanceV10StrategyOptimizer:
    """币安V10策略优化器"""
    
    def __init__(self, api_key: str, secret_key: str, symbol: str = "ETHUSDT"):
        self.trading_system = BinanceV10SimpleAdvanced(api_key, secret_key, symbol)
        self.signal_optimizer = SignalOptimizer()
        self.symbol = symbol
        self.is_running = False
        self.optimization_thread = None
        self.current_params = StrategyParams()
        self.optimization_results = []
        
    def start(self):
        """启动策略优化器"""
        logger.info("启动币安V10策略优化器...")
        
        # 启动交易系统
        self.trading_system.start()
        
        # 启动优化线程
        self.is_running = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
        logger.info("币安V10策略优化器启动成功")
    
    def stop(self):
        """停止策略优化器"""
        logger.info("停止币安V10策略优化器...")
        
        self.is_running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        
        self.trading_system.stop()
        logger.info("币安V10策略优化器已停止")
    
    def _optimization_loop(self):
        """优化主循环"""
        logger.info("策略优化循环开始...")
        
        while self.is_running:
            try:
                # 获取交易状态
                status = self.trading_system.get_trading_status()
                
                # 收集信号数据
                signal_data = {
                    'timestamp': time.time(),
                    'price': status['current_price'],
                    'ofi_zscore': status['ofi_zscore'],
                    'volume': 1000.0,  # 模拟成交量
                    'momentum': np.random.normal(0, 0.001),  # 模拟动量
                    'signal_strength': abs(status['ofi_zscore']),
                    'quality_score': 0.5
                }
                
                # 计算信号质量
                signal_data['quality_score'] = self.signal_optimizer.calculate_signal_quality(signal_data)
                
                # 添加信号历史
                self.signal_optimizer.add_signal(signal_data)
                
                # 信号过滤
                filter_config = SignalFilter(
                    ofi_threshold=self.current_params.ofi_threshold,
                    momentum_threshold=self.current_params.momentum_threshold,
                    volume_threshold=self.current_params.volume_threshold,
                    quality_score_threshold=0.5
                )
                
                if self.signal_optimizer.filter_signals(signal_data, filter_config):
                    logger.info(f"高质量信号: OFI={signal_data['ofi_zscore']:.3f}, 质量={signal_data['quality_score']:.3f}")
                
                # 定期优化参数
                if len(self.signal_optimizer.signal_history) > 100:
                    self._run_parameter_optimization()
                
                time.sleep(1)  # 1秒循环
                
            except Exception as e:
                logger.error(f"优化循环异常: {e}")
                time.sleep(1)
    
    def _run_parameter_optimization(self):
        """运行参数优化"""
        logger.info("开始参数优化...")
        
        # 获取历史数据
        historical_data = []
        for signal in list(self.signal_optimizer.signal_history):
            historical_data.append({
                'price': signal['price'],
                'ofi_zscore': signal['ofi_zscore'],
                'momentum': signal.get('momentum', 0),
                'volume': signal.get('volume', 1000)
            })
        
        # 运行优化
        best_params = self.signal_optimizer.optimize_parameters(historical_data)
        
        if best_params:
            self.current_params = best_params
            logger.info(f"参数优化完成，新参数: {best_params}")
            
            # 保存优化结果
            self.optimization_results.append({
                'timestamp': time.time(),
                'params': best_params,
                'signal_count': len(historical_data)
            })
    
    def get_optimization_status(self) -> Dict:
        """获取优化状态"""
        return {
            "symbol": self.symbol,
            "current_params": {
                "ofi_threshold": self.current_params.ofi_threshold,
                "momentum_threshold": self.current_params.momentum_threshold,
                "stop_loss_pct": self.current_params.stop_loss_pct,
                "take_profit_pct": self.current_params.take_profit_pct
            },
            "signal_history_count": len(self.signal_optimizer.signal_history),
            "optimization_runs": len(self.optimization_results),
            "trading_status": self.trading_system.get_trading_status()
        }
    
    def export_optimization_results(self, filename: str = None):
        """导出优化结果"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
        
        results = {
            "optimization_runs": len(self.optimization_results),
            "current_params": {
                "ofi_threshold": self.current_params.ofi_threshold,
                "momentum_threshold": self.current_params.momentum_threshold,
                "stop_loss_pct": self.current_params.stop_loss_pct,
                "take_profit_pct": self.current_params.take_profit_pct
            },
            "signal_history": list(self.signal_optimizer.signal_history),
            "optimization_results": self.optimization_results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"优化结果已导出到: {filename}")

if __name__ == "__main__":
    # 币安测试网API配置
    API_KEY = "tKs2nQW3t3rcHnUCGOPWLIYsDotF6ot1GRZzo5n3QOsNRqbPTxO06TC8QglxtUYy"
    SECRET_KEY = "I5SqluL8yqvkcSSb0hBBMr80cHmNndxrl6DAufw9t6myjlRBQDvPxUmabHNJpkZJ"
    
    # 创建策略优化器
    optimizer = BinanceV10StrategyOptimizer(API_KEY, SECRET_KEY)
    
    try:
        # 启动优化器
        optimizer.start()
        
        # 运行优化
        logger.info("策略优化器运行中...")
        time.sleep(120)  # 运行2分钟
        
        # 显示状态
        status = optimizer.get_optimization_status()
        logger.info(f"优化状态: {status}")
        
        # 导出结果
        optimizer.export_optimization_results()
        
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"系统异常: {e}")
    finally:
        # 停止优化器
        optimizer.stop()
        logger.info("策略优化器测试完成")
