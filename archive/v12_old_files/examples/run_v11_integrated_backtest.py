#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11集成回测系统
结合深度学习模型和高级特征工程的完整回测系统
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入V11模块
from v11_deep_learning import V11DeepLearning
from v11_advanced_features import V11AdvancedFeatureEngine

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11IntegratedBacktester:
    """V11集成回测系统"""
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 初始化组件
        self.feature_engine = V11AdvancedFeatureEngine()
        self.deep_learning = V11DeepLearning(device=self.device)
        
        # 回测状态
        self.reset()
        
        logger.info(f"V11集成回测系统初始化完成，设备: {self.device}")
    
    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = []
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_capital
        self.ml_predictions = []
        self.signal_history = []
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备数据"""
        logger.info("开始数据准备...")
        
        # 特征工程
        df_features = self.feature_engine.create_all_features(df)
        
        # 创建目标变量（未来价格变化）
        df_features['future_return_1'] = df_features['close'].shift(-1) / df_features['close'] - 1
        df_features['future_return_5'] = df_features['close'].shift(-5) / df_features['close'] - 1
        df_features['future_return_10'] = df_features['close'].shift(-10) / df_features['close'] - 1
        
        # 创建信号标签
        df_features['signal_label'] = np.where(
            df_features['future_return_1'] > 0.001, 1,
            np.where(df_features['future_return_1'] < -0.001, -1, 0)
        )
        
        logger.info(f"数据准备完成，特征数: {len(self.feature_engine.feature_columns)}")
        
        return df_features
    
    def train_models(self, df: pd.DataFrame, sequence_length: int = 60) -> Dict:
        """训练机器学习模型"""
        logger.info("开始训练机器学习模型...")
        
        # 准备特征数据
        feature_columns = self.feature_engine.feature_columns
        features = df[feature_columns].values
        targets = df['future_return_1'].values
        
        # 准备训练数据
        X_train, X_test, y_train, y_test = self.deep_learning.prepare_data(
            features, targets, sequence_length=sequence_length
        )
        
        # 训练各个模型
        models_to_train = ['lstm', 'transformer', 'cnn']
        training_results = {}
        
        for model_name in models_to_train:
            logger.info(f"训练 {model_name} 模型...")
            
            if model_name == 'lstm':
                model = self.deep_learning.create_lstm_model(input_size=len(feature_columns))
            elif model_name == 'transformer':
                model = self.deep_learning.create_transformer_model(input_size=len(feature_columns))
            elif model_name == 'cnn':
                model = self.deep_learning.create_cnn_model(input_size=len(feature_columns))
            
            # 训练模型
            training_history = self.deep_learning.train_model(
                model_name, X_train, y_train, X_test, y_test,
                epochs=100, batch_size=32, learning_rate=0.001
            )
            
            # 评估模型
            evaluation = self.deep_learning.evaluate_model(model_name, X_test, y_test)
            
            training_results[model_name] = {
                'training_history': training_history,
                'evaluation': evaluation
            }
        
        # 创建和训练集成模型
        logger.info("训练集成模型...")
        ensemble_model = self.deep_learning.create_ensemble_model(input_size=len(feature_columns))
        
        ensemble_history = self.deep_learning.train_model(
            'ensemble', X_train, y_train, X_test, y_test,
            epochs=100, batch_size=32, learning_rate=0.001
        )
        
        ensemble_evaluation = self.deep_learning.evaluate_model('ensemble', X_test, y_test)
        
        training_results['ensemble'] = {
            'training_history': ensemble_history,
            'evaluation': ensemble_evaluation
        }
        
        logger.info("机器学习模型训练完成")
        return training_results
    
    def generate_ml_signals(self, df: pd.DataFrame, model_name: str = 'ensemble') -> pd.DataFrame:
        """生成机器学习信号"""
        logger.info(f"生成 {model_name} 机器学习信号...")
        
        # 准备特征数据
        feature_columns = self.feature_engine.feature_columns
        features = df[feature_columns].values
        
        # 创建序列数据
        sequence_length = 60
        X_sequences = []
        valid_indices = []
        
        for i in range(sequence_length, len(features)):
            X_sequences.append(features[i-sequence_length:i])
            valid_indices.append(i)
        
        X_sequences = np.array(X_sequences)
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        
        # 生成预测
        predictions = self.deep_learning.predict(model_name, X_tensor)
        
        # 创建信号DataFrame
        signals_df = pd.DataFrame(index=df.index)
        signals_df['ml_prediction'] = 0.0
        signals_df['ml_signal'] = 0
        signals_df['ml_confidence'] = 0.0
        
        # 填充预测结果
        for i, idx in enumerate(valid_indices):
            if idx < len(signals_df):
                signals_df.iloc[idx, signals_df.columns.get_loc('ml_prediction')] = predictions[i]
                signals_df.iloc[idx, signals_df.columns.get_loc('ml_confidence')] = abs(predictions[i])
        
        # 生成信号
        signal_threshold = 0.001
        confidence_threshold = 0.5
        
        for i in range(len(signals_df)):
            prediction = signals_df.iloc[i]['ml_prediction']
            confidence = signals_df.iloc[i]['ml_confidence']
            
            if confidence > confidence_threshold:
                if prediction > signal_threshold:
                    signals_df.iloc[i, signals_df.columns.get_loc('ml_signal')] = 1
                elif prediction < -signal_threshold:
                    signals_df.iloc[i, signals_df.columns.get_loc('ml_signal')] = -1
        
        # 统计信号
        total_signals = (signals_df['ml_signal'] != 0).sum()
        long_signals = (signals_df['ml_signal'] == 1).sum()
        short_signals = (signals_df['ml_signal'] == -1).sum()
        
        logger.info(f"机器学习信号统计:")
        logger.info(f"  总信号数: {total_signals}")
        logger.info(f"  多头信号: {long_signals}")
        logger.info(f"  空头信号: {short_signals}")
        
        return signals_df
    
    def run_backtest(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> Dict:
        """运行回测"""
        logger.info("开始V11集成回测...")
        
        self.reset()
        
        for i, row in df.iterrows():
            current_price = row['close']
            signal = signals_df.iloc[i]['ml_signal'] if i < len(signals_df) else 0
            confidence = signals_df.iloc[i]['ml_confidence'] if i < len(signals_df) else 0
            
            # 更新权益曲线
            self.equity_curve.append(self.capital)
            
            # 计算回撤
            if self.capital > self.peak_equity:
                self.peak_equity = self.capital
            
            current_drawdown = (self.peak_equity - self.capital) / self.peak_equity
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # 处理信号
            if signal != 0 and self.position == 0 and confidence > 0.5:
                # 开仓
                self._open_position(signal, current_price, confidence)
            elif self.position != 0:
                # 检查平仓条件
                self._check_exit_conditions(current_price, signal, confidence)
        
        # 计算回测结果
        results = self._calculate_results()
        
        logger.info("V11集成回测完成")
        return results
    
    def _open_position(self, signal: int, price: float, confidence: float):
        """开仓"""
        # 根据置信度调整仓位大小
        position_size = min(1.0, confidence * 2)
        self.position = signal * position_size
        self.entry_price = price
        
        # 记录交易
        self.trades.append({
            'action': 'OPEN',
            'side': 'LONG' if signal > 0 else 'SHORT',
            'price': price,
            'size': position_size,
            'confidence': confidence,
            'timestamp': len(self.equity_curve)
        })
        
        # 记录信号
        self.signal_history.append({
            'signal': signal,
            'confidence': confidence,
            'price': price,
            'timestamp': len(self.equity_curve)
        })
    
    def _check_exit_conditions(self, current_price: float, signal: int, confidence: float):
        """检查平仓条件"""
        if self.position == 0:
            return
        
        # 计算当前盈亏
        if self.position > 0:  # 多头仓位
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # 空头仓位
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # 平仓条件
        should_close = False
        exit_reason = ""
        
        # 止损条件
        if pnl_pct < -0.02:  # 2%止损
            should_close = True
            exit_reason = "STOP_LOSS"
        
        # 止盈条件
        elif pnl_pct > 0.04:  # 4%止盈
            should_close = True
            exit_reason = "TAKE_PROFIT"
        
        # 信号反转
        elif signal != 0 and signal * self.position < 0:
            should_close = True
            exit_reason = "SIGNAL_REVERSAL"
        
        # 时间止损
        elif len(self.trades) > 0 and len(self.equity_curve) - self.trades[-1]['timestamp'] > 100:
            should_close = True
            exit_reason = "TIME_STOP"
        
        if should_close:
            self._close_position(current_price, exit_reason)
    
    def _close_position(self, price: float, reason: str):
        """平仓"""
        if self.position == 0:
            return
        
        # 计算盈亏
        if self.position > 0:  # 多头仓位
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:  # 空头仓位
            pnl_pct = (self.entry_price - price) / self.entry_price
        
        # 计算手续费
        commission_cost = abs(self.position) * self.commission
        
        # 更新资金
        pnl_amount = self.capital * pnl_pct * abs(self.position)
        self.capital += pnl_amount - commission_cost
        
        # 记录交易
        self.trades.append({
            'action': 'CLOSE',
            'side': 'LONG' if self.position > 0 else 'SHORT',
            'price': price,
            'size': abs(self.position),
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'commission': commission_cost,
            'reason': reason,
            'timestamp': len(self.equity_curve)
        })
        
        # 重置仓位
        self.position = 0.0
        self.entry_price = 0.0
    
    def _calculate_results(self) -> Dict:
        """计算回测结果"""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'final_capital': self.capital,
                'final_return': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'profit_factor': 0.0
            }
        
        # 计算交易统计
        closed_trades = [t for t in self.trades if t['action'] == 'CLOSE']
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'final_capital': self.capital,
                'final_return': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'profit_factor': 0.0
            }
        
        total_trades = len(closed_trades)
        total_pnl = sum(t['pnl_amount'] for t in closed_trades)
        winning_trades = [t for t in closed_trades if t['pnl_amount'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl_amount'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # 计算夏普比率
        returns = [t['pnl_pct'] for t in closed_trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # 计算盈利因子
        gross_profit = sum(t['pnl_amount'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl_amount'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # 计算最终收益
        final_return = (self.capital - self.initial_capital) / self.initial_capital
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'final_capital': self.capital,
            'final_return': final_return,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'avg_trade_pnl': total_pnl / total_trades if total_trades > 0 else 0,
            'max_win': max([t['pnl_amount'] for t in closed_trades]) if closed_trades else 0,
            'max_loss': min([t['pnl_amount'] for t in closed_trades]) if closed_trades else 0,
            'ml_signals': len(self.signal_history),
            'avg_confidence': np.mean([s['confidence'] for s in self.signal_history]) if self.signal_history else 0
        }

def create_v11_test_data(n_samples: int = 5000) -> pd.DataFrame:
    """创建V11测试数据"""
    np.random.seed(42)
    
    # 生成价格数据
    price_base = 100
    price_changes = np.random.randn(n_samples) * 0.01
    prices = price_base + np.cumsum(price_changes)
    
    # 生成OHLC数据
    highs = prices + np.random.uniform(0.01, 0.05, n_samples)
    lows = prices - np.random.uniform(0.01, 0.05, n_samples)
    opens = prices + np.random.uniform(-0.02, 0.02, n_samples)
    
    # 确保OHLC关系正确
    highs = np.maximum(highs, prices)
    lows = np.minimum(lows, prices)
    highs = np.maximum(highs, opens)
    lows = np.minimum(lows, opens)
    
    # 生成成交量数据
    volumes = np.random.randint(100, 1000, n_samples)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'open_time': pd.date_range('2024-01-01', periods=n_samples, freq='1s'),
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    return df

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("V11集成回测系统")
    logger.info("=" * 60)
    
    try:
        # 创建测试数据
        logger.info("创建测试数据...")
        df = create_v11_test_data(n_samples=2000)
        logger.info(f"测试数据创建完成: {len(df)} 条记录")
        
        # 创建V11集成回测系统
        backtester = V11IntegratedBacktester(initial_capital=10000.0)
        
        # 准备数据
        logger.info("准备数据...")
        df_features = backtester.prepare_data(df)
        logger.info(f"数据准备完成，特征数: {len(backtester.feature_engine.feature_columns)}")
        
        # 训练机器学习模型
        logger.info("训练机器学习模型...")
        training_results = backtester.train_models(df_features, sequence_length=60)
        
        # 生成机器学习信号
        logger.info("生成机器学习信号...")
        signals_df = backtester.generate_ml_signals(df_features, model_name='ensemble')
        
        # 运行回测
        logger.info("运行回测...")
        results = backtester.run_backtest(df_features, signals_df)
        
        # 输出结果
        logger.info("=" * 60)
        logger.info("V11集成回测结果")
        logger.info("=" * 60)
        logger.info(f"总交易数: {results['total_trades']}")
        logger.info(f"最终资金: ${results['final_capital']:,.2f}")
        logger.info(f"总收益: ${results['total_pnl']:,.2f}")
        logger.info(f"收益率: {results['final_return']:.2%}")
        logger.info(f"胜率: {results['win_rate']:.2%}")
        logger.info(f"最大回撤: {results['max_drawdown']:.2%}")
        logger.info(f"夏普比率: {results['sharpe_ratio']:.4f}")
        logger.info(f"盈利因子: {results['profit_factor']:.4f}")
        logger.info(f"平均每笔收益: ${results['avg_trade_pnl']:,.2f}")
        logger.info(f"最大单笔盈利: ${results['max_win']:,.2f}")
        logger.info(f"最大单笔亏损: ${results['max_loss']:,.2f}")
        logger.info(f"机器学习信号数: {results['ml_signals']}")
        logger.info(f"平均置信度: {results['avg_confidence']:.4f}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"v11_integrated_backtest_results_{timestamp}.json"
        
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"回测结果已保存到: {results_file}")
        
    except Exception as e:
        logger.error(f"回测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
