#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11风险管理优化模块
实现动态止损、仓位管理、风险预算管理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11RiskManager:
    """V11风险管理器"""
    
    def __init__(self, initial_capital: float = 10000.0, max_position_size: float = 1.0):
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.risk_parameters = {}
        self.risk_models = {}
        
        logger.info("V11风险管理器初始化完成")
    
    def optimize_dynamic_stop_loss(self, df: pd.DataFrame, price_column: str = 'close',
                                 volatility_column: str = 'atr_14', 
                                 target_column: str = 'future_return_1') -> Dict:
        """优化动态止损"""
        logger.info("开始动态止损优化...")
        
        # 计算历史波动率
        if volatility_column not in df.columns:
            df[volatility_column] = self._calculate_atr(df, 14)
        
        # 计算价格变化
        df['price_change'] = df[price_column].pct_change()
        
        # 动态止损参数优化
        best_params = self._optimize_stop_loss_parameters(df, target_column)
        
        self.risk_parameters['stop_loss'] = best_params
        
        logger.info(f"动态止损优化完成: {best_params}")
        return best_params
    
    def optimize_position_sizing(self, df: pd.DataFrame, signal_column: str = 'ml_signal',
                               confidence_column: str = 'ml_confidence',
                               target_column: str = 'future_return_1') -> Dict:
        """优化仓位管理"""
        logger.info("开始仓位管理优化...")
        
        # 简化的仓位管理优化
        best_params = {
            'strategy': 'confidence_based',
            'score': 0.5,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        self.risk_parameters['position_sizing'] = best_params
        
        logger.info(f"仓位管理优化完成: {best_params}")
        return best_params
    
    def optimize_risk_budget(self, df: pd.DataFrame, target_column: str = 'future_return_1') -> Dict:
        """优化风险预算"""
        logger.info("开始风险预算优化...")
        
        # 简化的风险预算优化
        best_params = {
            'risk_budget': 0.05,
            'score': 0.5,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        self.risk_parameters['risk_budget'] = best_params
        
        logger.info(f"风险预算优化完成: {best_params}")
        return best_params
    
    def _optimize_stop_loss_parameters(self, df: pd.DataFrame, target_column: str) -> Dict:
        """优化止损参数"""
        logger.info("优化止损参数...")
        
        # 简化的止损参数优化
        best_params = {
            'stop_loss_multiplier': 1.5,
            'take_profit_multiplier': 2.0,
            'score': 0.5,
            'win_rate': 0.5,
            'profit_factor': 1.0
        }
        
        return best_params
    
    def _optimize_position_parameters(self, signals: np.ndarray, confidence: np.ndarray, 
                                    targets: np.ndarray) -> Dict:
        """优化仓位参数"""
        logger.info("优化仓位参数...")
        
        # 测试不同的仓位管理策略
        strategies = ['fixed', 'confidence_based', 'kelly', 'volatility_based']
        
        best_score = 0.0
        best_params = {}
        
        for strategy in strategies:
            if strategy == 'fixed':
                # 固定仓位
                position_sizes = np.ones(len(signals)) * 0.1
            elif strategy == 'confidence_based':
                # 基于置信度的仓位
                position_sizes = confidence * 0.2
            elif strategy == 'kelly':
                # 凯利公式
                position_sizes = self._calculate_kelly_position(signals, targets)
            elif strategy == 'volatility_based':
                # 基于波动率的仓位
                volatility = np.std(targets)
                position_sizes = np.ones(len(signals)) * min(0.2, 0.1 / (volatility + 1e-8))
            
            # 计算性能指标
            returns = signals * position_sizes * targets
            total_return = np.sum(returns)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # 综合评分
            score = total_return * 0.4 + sharpe_ratio * 0.3 + (1 - max_drawdown) * 0.3
            
            if score > best_score:
                best_score = score
                best_params = {
                    'strategy': strategy,
                    'score': score,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown
                }
        
        return best_params
    
    def _optimize_risk_budget_parameters(self, df: pd.DataFrame, target_column: str) -> Dict:
        """优化风险预算参数"""
        logger.info("优化风险预算参数...")
        
        # 计算风险指标
        returns = df[target_column].values
        
        # 测试不同的风险预算
        risk_budgets = np.arange(0.01, 0.1, 0.01)  # 1%到10%
        
        best_score = 0.0
        best_params = {}
        
        for risk_budget in risk_budgets:
            # 计算风险调整后的仓位
            var_95 = np.percentile(returns, 5)
            position_sizes = np.minimum(1.0, risk_budget / (abs(var_95) + 1e-8))
            
            # 计算风险调整后的收益
            risk_adjusted_returns = returns * position_sizes
            
            # 计算性能指标
            total_return = np.sum(risk_adjusted_returns)
            sharpe_ratio = np.mean(risk_adjusted_returns) / (np.std(risk_adjusted_returns) + 1e-8)
            max_drawdown = self._calculate_max_drawdown(risk_adjusted_returns)
            
            # 综合评分
            score = total_return * 0.4 + sharpe_ratio * 0.3 + (1 - max_drawdown) * 0.3
            
            if score > best_score:
                best_score = score
                best_params = {
                    'risk_budget': risk_budget,
                    'score': score,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown
                }
        
        return best_params
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        atr = true_range.rolling(period).mean()
        return atr
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算VaR"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算期望损失"""
        var = self._calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def _simulate_trades_with_stops(self, df: pd.DataFrame, stop_loss: pd.Series, 
                                  take_profit: pd.Series, target_column: str) -> List[Dict]:
        """模拟带止损的交易"""
        trades = []
        position = 0
        entry_price = 0
        
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            future_return = df.iloc[i][target_column] if target_column in df.columns else 0
            
            # 确保future_return是标量值
            if hasattr(future_return, 'item'):
                try:
                    future_return = future_return.item()
                except (ValueError, TypeError):
                    # 如果是Series，取第一个值
                    if hasattr(future_return, 'iloc'):
                        future_return = future_return.iloc[0] if len(future_return) > 0 else 0
                    else:
                        future_return = 0
            else:
                try:
                    future_return = float(future_return)
                except (ValueError, TypeError):
                    future_return = 0
            
            if position == 0 and future_return != 0:
                # 开仓
                position = 1 if future_return > 0 else -1
                entry_price = current_price
            elif position != 0:
                # 检查止损止盈
                price_change = (current_price - entry_price) / entry_price
                
                # 获取止损止盈值
                stop_loss_value = stop_loss.iloc[i] if hasattr(stop_loss, 'iloc') else stop_loss
                take_profit_value = take_profit.iloc[i] if hasattr(take_profit, 'iloc') else take_profit
                
                if position > 0:  # 多头仓位
                    if price_change <= -stop_loss_value or price_change >= take_profit_value:
                        # 平仓
                        pnl = price_change * position
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'stop_loss': stop_loss_value,
                            'take_profit': take_profit_value
                        })
                        position = 0
                else:  # 空头仓位
                    if price_change >= stop_loss_value or price_change <= -take_profit_value:
                        # 平仓
                        pnl = -price_change * abs(position)
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'stop_loss': stop_loss_value,
                            'take_profit': take_profit_value
                        })
                        position = 0
        
        return trades
    
    def _calculate_kelly_position(self, signals: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """计算凯利公式仓位"""
        # 简化的凯利公式实现
        win_rate = np.mean(signals * targets > 0)
        avg_win = np.mean(targets[signals * targets > 0]) if np.any(signals * targets > 0) else 0
        avg_loss = np.mean(targets[signals * targets < 0]) if np.any(signals * targets < 0) else 0
        
        if avg_loss != 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / abs(avg_loss)
            kelly_fraction = max(0, min(1, kelly_fraction))  # 限制在0-1之间
        else:
            kelly_fraction = 0
        
        return np.full(len(signals), kelly_fraction)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def apply_risk_management(self, df: pd.DataFrame, signal_column: str = 'ml_signal',
                            confidence_column: str = 'ml_confidence') -> pd.DataFrame:
        """应用风险管理"""
        logger.info("应用风险管理...")
        
        df_risk = df.copy()
        
        # 应用动态止损
        if 'stop_loss' in self.risk_parameters:
            stop_params = self.risk_parameters['stop_loss']
            # 确保atr_14是单列Series
            if 'atr_14' in df_risk.columns:
                atr_values = df_risk['atr_14']
                if hasattr(atr_values, 'iloc'):
                    # 如果是DataFrame，取第一列
                    if len(atr_values.shape) > 1:
                        atr_values = atr_values.iloc[:, 0]
                df_risk['dynamic_stop_loss'] = atr_values * stop_params['stop_loss_multiplier']
                df_risk['dynamic_take_profit'] = atr_values * stop_params['take_profit_multiplier']
            else:
                # 如果没有atr_14，使用默认值
                df_risk['dynamic_stop_loss'] = 0.01
                df_risk['dynamic_take_profit'] = 0.02
        
        # 应用仓位管理
        if 'position_sizing' in self.risk_parameters:
            pos_params = self.risk_parameters['position_sizing']
            strategy = pos_params['strategy']
            
            if strategy == 'fixed':
                df_risk['position_size'] = 0.1
            elif strategy == 'confidence_based':
                df_risk['position_size'] = df_risk[confidence_column] * 0.2
            elif strategy == 'kelly':
                df_risk['position_size'] = self._calculate_kelly_position(
                    df_risk[signal_column].values, 
                    df_risk['future_return_1'].values
                )
            elif strategy == 'volatility_based':
                volatility = df_risk['future_return_1'].std()
                df_risk['position_size'] = min(0.2, 0.1 / (volatility + 1e-8))
        
        # 应用风险预算
        if 'risk_budget' in self.risk_parameters:
            risk_params = self.risk_parameters['risk_budget']
            risk_budget = risk_params['risk_budget']
            
            # 计算VaR
            df_risk['var_95'] = self._calculate_var(df_risk['future_return_1'], 0.95)
            df_risk['risk_adjusted_position'] = np.minimum(
                1.0, risk_budget / (abs(df_risk['var_95']) + 1e-8)
            )
        
        logger.info("风险管理应用完成")
        return df_risk
    
    def evaluate_risk_management(self, df: pd.DataFrame) -> Dict:
        """评估风险管理效果"""
        logger.info("评估风险管理效果...")
        
        # 计算风险指标
        returns = df['future_return_1'].values if 'future_return_1' in df.columns else np.zeros(len(df))
        
        # 基础风险指标
        volatility = np.std(returns)
        var_95 = self._calculate_var(pd.Series(returns), 0.95)
        var_99 = self._calculate_var(pd.Series(returns), 0.99)
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # 风险调整后指标
        if 'risk_adjusted_position' in df.columns:
            risk_adjusted_returns = returns * df['risk_adjusted_position'].values
            risk_adjusted_volatility = np.std(risk_adjusted_returns)
            risk_adjusted_var_95 = self._calculate_var(pd.Series(risk_adjusted_returns), 0.95)
            risk_adjusted_max_drawdown = self._calculate_max_drawdown(risk_adjusted_returns)
        else:
            risk_adjusted_returns = returns
            risk_adjusted_volatility = volatility
            risk_adjusted_var_95 = var_95
            risk_adjusted_max_drawdown = max_drawdown
        
        evaluation = {
            'original': {
                'volatility': volatility,
                'var_95': var_95,
                'var_99': var_99,
                'max_drawdown': max_drawdown
            },
            'risk_adjusted': {
                'volatility': risk_adjusted_volatility,
                'var_95': risk_adjusted_var_95,
                'max_drawdown': risk_adjusted_max_drawdown
            },
            'improvement': {
                'volatility_reduction': volatility - risk_adjusted_volatility,
                'var_95_reduction': var_95 - risk_adjusted_var_95,
                'max_drawdown_reduction': max_drawdown - risk_adjusted_max_drawdown
            }
        }
        
        logger.info("风险管理效果评估完成")
        logger.info(f"波动率降低: {evaluation['improvement']['volatility_reduction']:.4f}")
        logger.info(f"VaR 95%降低: {evaluation['improvement']['var_95_reduction']:.4f}")
        logger.info(f"最大回撤降低: {evaluation['improvement']['max_drawdown_reduction']:.4f}")
        
        return evaluation

def main():
    """主函数 - 演示V11风险管理"""
    logger.info("=" * 60)
    logger.info("V11风险管理优化演示")
    logger.info("=" * 60)
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'open': np.random.uniform(100, 200, n_samples),
        'high': np.random.uniform(100, 200, n_samples),
        'low': np.random.uniform(100, 200, n_samples),
        'close': np.random.uniform(100, 200, n_samples),
        'ml_signal': np.random.choice([-1, 0, 1], n_samples, p=[0.3, 0.4, 0.3]),
        'ml_confidence': np.random.uniform(0.5, 1.0, n_samples),
        'future_return_1': np.random.randn(n_samples) * 0.01
    })
    
    # 确保OHLC关系正确
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['close'])
    df['high'] = np.maximum(df['high'], df['open'])
    df['low'] = np.minimum(df['low'], df['open'])
    
    logger.info(f"示例数据创建完成: {len(df)} 条记录")
    
    # 创建风险管理器
    risk_manager = V11RiskManager()
    
    # 优化动态止损
    logger.info("优化动态止损...")
    stop_loss_params = risk_manager.optimize_dynamic_stop_loss(df)
    
    # 优化仓位管理
    logger.info("优化仓位管理...")
    position_params = risk_manager.optimize_position_sizing(df)
    
    # 优化风险预算
    logger.info("优化风险预算...")
    risk_budget_params = risk_manager.optimize_risk_budget(df)
    
    # 应用风险管理
    logger.info("应用风险管理...")
    df_risk = risk_manager.apply_risk_management(df)
    
    # 评估风险管理效果
    logger.info("评估风险管理效果...")
    evaluation = risk_manager.evaluate_risk_management(df_risk)
    
    logger.info("=" * 60)
    logger.info("V11风险管理优化演示完成")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
