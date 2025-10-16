#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 Phase 2 集成回测系统
结合信号优化和风险管理的完整回测系统
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
from v11_signal_optimizer import V11SignalOptimizer
from v11_risk_manager import V11RiskManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11Phase2Backtester:
    """V11 Phase 2 集成回测系统"""
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 初始化组件
        self.feature_engine = V11AdvancedFeatureEngine()
        self.deep_learning = V11DeepLearning(device=self.device)
        self.signal_optimizer = V11SignalOptimizer()
        self.risk_manager = V11RiskManager(initial_capital=initial_capital)
        
        # 回测状态
        self.reset()
        
        logger.info(f"V11 Phase 2 集成回测系统初始化完成，设备: {self.device}")
    
    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = []
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_capital
        self.optimization_results = {}
    
    def run_complete_optimization(self, df: pd.DataFrame) -> Dict:
        """运行完整优化流程"""
        logger.info("开始V11 Phase 2 完整优化流程...")
        
        # 1. 特征工程
        logger.info("步骤1: 高级特征工程...")
        df_features = self.feature_engine.create_all_features(df)
        
        # 创建目标变量
        df_features['future_return_1'] = df_features['close'].shift(-1) / df_features['close'] - 1
        df_features['future_return_5'] = df_features['close'].shift(-5) / df_features['close'] - 1
        df_features['future_return_10'] = df_features['close'].shift(-10) / df_features['close'] - 1
        
        logger.info(f"特征工程完成，特征数: {len(self.feature_engine.feature_columns)}")
        
        # 2. 机器学习模型训练
        logger.info("步骤2: 机器学习模型训练...")
        training_results = self._train_ml_models(df_features)
        
        # 3. 信号生成
        logger.info("步骤3: 机器学习信号生成...")
        signals_df = self._generate_ml_signals(df_features)
        
        # 4. 信号优化
        logger.info("步骤4: 信号生成优化...")
        signals_optimized = self._optimize_signals(df_features, signals_df)
        
        # 5. 风险管理优化
        logger.info("步骤5: 风险管理优化...")
        risk_optimized = self._optimize_risk_management(df_features, signals_optimized)
        
        # 6. 集成回测
        logger.info("步骤6: 集成回测...")
        backtest_results = self._run_integrated_backtest(df_features, signals_optimized, risk_optimized)
        
        # 保存优化结果
        self.optimization_results = {
            'feature_engineering': {
                'total_features': len(self.feature_engine.feature_columns),
                'feature_columns': self.feature_engine.feature_columns
            },
            'ml_training': training_results,
            'signal_optimization': self.signal_optimizer.best_parameters,
            'risk_optimization': self.risk_manager.risk_parameters,
            'backtest_results': backtest_results
        }
        
        logger.info("V11 Phase 2 完整优化流程完成")
        return self.optimization_results
    
    def _train_ml_models(self, df: pd.DataFrame) -> Dict:
        """训练机器学习模型"""
        logger.info("训练机器学习模型...")
        
        # 准备特征数据
        feature_columns = self.feature_engine.feature_columns
        features = df[feature_columns].values
        targets = df['future_return_1'].values
        
        # 准备训练数据
        X_train, X_test, y_train, y_test = self.deep_learning.prepare_data(
            features, targets, sequence_length=60
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
                epochs=50, batch_size=32, learning_rate=0.001
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
            epochs=50, batch_size=32, learning_rate=0.001
        )
        
        ensemble_evaluation = self.deep_learning.evaluate_model('ensemble', X_test, y_test)
        
        training_results['ensemble'] = {
            'training_history': ensemble_history,
            'evaluation': ensemble_evaluation
        }
        
        return training_results
    
    def _generate_ml_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成机器学习信号"""
        logger.info("生成机器学习信号...")
        
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
        predictions = self.deep_learning.predict('ensemble', X_tensor)
        
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
        
        return signals_df
    
    def _optimize_signals(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
        """优化信号"""
        logger.info("优化信号生成...")
        
        # 合并数据
        df_combined = pd.concat([df, signals_df], axis=1)
        
        # 优化信号阈值
        threshold_params = self.signal_optimizer.optimize_signal_thresholds(
            df_combined, optimization_method='grid_search'
        )
        
        # 优化信号权重
        weight_params = self.signal_optimizer.optimize_signal_weights(df_combined)
        
        # 优化信号时机
        timing_params = self.signal_optimizer.optimize_signal_timing(df_combined)
        
        # 应用优化后的信号
        df_optimized = self.signal_optimizer.apply_optimized_signals(df_combined)
        
        return df_optimized
    
    def _optimize_risk_management(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
        """优化风险管理"""
        logger.info("优化风险管理...")
        
        # 合并数据
        df_combined = pd.concat([df, signals_df], axis=1)
        
        # 优化动态止损
        stop_loss_params = self.risk_manager.optimize_dynamic_stop_loss(df_combined)
        
        # 优化仓位管理
        position_params = self.risk_manager.optimize_position_sizing(df_combined)
        
        # 优化风险预算
        risk_budget_params = self.risk_manager.optimize_risk_budget(df_combined)
        
        # 应用风险管理
        df_risk = self.risk_manager.apply_risk_management(df_combined)
        
        return df_risk
    
    def _run_integrated_backtest(self, df: pd.DataFrame, signals_df: pd.DataFrame, 
                               risk_df: pd.DataFrame) -> Dict:
        """运行集成回测"""
        logger.info("运行集成回测...")
        
        self.reset()
        
        for i, row in df.iterrows():
            current_price = row['close']
            
            # 获取信号
            if i < len(signals_df):
                signal = signals_df.iloc[i].get('timed_signal', 
                       signals_df.iloc[i].get('weighted_signal', 
                       signals_df.iloc[i].get('ml_signal_optimized', 
                       signals_df.iloc[i].get('ml_signal', 0))))
                confidence = signals_df.iloc[i].get('ml_confidence', 0.5)
            else:
                signal = 0
                confidence = 0.5
            
            # 获取仓位大小
            if i < len(risk_df):
                position_size = risk_df.iloc[i].get('position_size', 0.1)
                risk_adjusted_position = risk_df.iloc[i].get('risk_adjusted_position', 1.0)
                final_position_size = min(position_size, risk_adjusted_position)
            else:
                final_position_size = 0.1
            
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
                self._open_position(signal, current_price, confidence, final_position_size)
            elif self.position != 0:
                # 检查平仓条件
                self._check_exit_conditions(current_price, signal, confidence)
        
        # 计算回测结果
        results = self._calculate_results()
        
        return results
    
    def _open_position(self, signal: int, price: float, confidence: float, position_size: float):
        """开仓"""
        # 根据置信度和仓位大小调整实际仓位
        actual_position_size = min(position_size, confidence * 2)
        self.position = signal * actual_position_size
        self.entry_price = price
        
        # 记录交易
        self.trades.append({
            'action': 'OPEN',
            'side': 'LONG' if signal > 0 else 'SHORT',
            'price': price,
            'size': actual_position_size,
            'confidence': confidence,
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
            'max_loss': min([t['pnl_amount'] for t in closed_trades]) if closed_trades else 0
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
    logger.info("V11 Phase 2 集成回测系统")
    logger.info("=" * 60)
    
    try:
        # 创建测试数据
        logger.info("创建测试数据...")
        df = create_v11_test_data(n_samples=2000)
        logger.info(f"测试数据创建完成: {len(df)} 条记录")
        
        # 创建V11 Phase 2回测系统
        backtester = V11Phase2Backtester(initial_capital=10000.0)
        
        # 运行完整优化流程
        logger.info("运行完整优化流程...")
        optimization_results = backtester.run_complete_optimization(df)
        
        # 输出结果
        logger.info("=" * 60)
        logger.info("V11 Phase 2 优化结果")
        logger.info("=" * 60)
        
        # 特征工程结果
        feature_results = optimization_results['feature_engineering']
        logger.info(f"特征工程: {feature_results['total_features']} 个特征")
        
        # 机器学习训练结果
        ml_results = optimization_results['ml_training']
        for model_name, results in ml_results.items():
            evaluation = results['evaluation']
            logger.info(f"{model_name} 模型:")
            logger.info(f"  方向准确率: {evaluation['direction_accuracy']:.2f}%")
            logger.info(f"  RMSE: {evaluation['rmse']:.6f}")
            logger.info(f"  MAE: {evaluation['mae']:.6f}")
        
        # 信号优化结果
        signal_results = optimization_results['signal_optimization']
        logger.info("信号优化结果:")
        for param_type, params in signal_results.items():
            logger.info(f"  {param_type}: {params}")
        
        # 风险管理结果
        risk_results = optimization_results['risk_optimization']
        logger.info("风险管理结果:")
        for param_type, params in risk_results.items():
            logger.info(f"  {param_type}: {params}")
        
        # 回测结果
        backtest_results = optimization_results['backtest_results']
        logger.info("回测结果:")
        logger.info(f"  总交易数: {backtest_results['total_trades']}")
        logger.info(f"  最终资金: ${backtest_results['final_capital']:,.2f}")
        logger.info(f"  总收益: ${backtest_results['total_pnl']:,.2f}")
        logger.info(f"  收益率: {backtest_results['final_return']:.2%}")
        logger.info(f"  胜率: {backtest_results['win_rate']:.2%}")
        logger.info(f"  最大回撤: {backtest_results['max_drawdown']:.2%}")
        logger.info(f"  夏普比率: {backtest_results['sharpe_ratio']:.4f}")
        logger.info(f"  盈利因子: {backtest_results['profit_factor']:.4f}")
        logger.info(f"  平均每笔收益: ${backtest_results['avg_trade_pnl']:,.2f}")
        logger.info(f"  最大单笔盈利: ${backtest_results['max_win']:,.2f}")
        logger.info(f"  最大单笔亏损: ${backtest_results['max_loss']:,.2f}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"v11_phase2_backtest_results_{timestamp}.json"
        
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(optimization_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"优化结果已保存到: {results_file}")
        
    except Exception as e:
        logger.error(f"回测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
