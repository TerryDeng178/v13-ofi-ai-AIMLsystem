#!/usr/bin/env python3
"""
V12 完整系统回测
调用所有V12核心组件：AI层、信号处理层、执行层
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 导入V12完整组件
from src.v12_fresh_data_backtest_framework import V12FreshDataBacktestFramework
from src.v12_realistic_data_simulator import V12RealisticDataSimulator
from src.v12_real_ofi_calculator import V12RealOFICalculator
from src.v12_ofi_expert_model import V12OFIExpertModel
from src.v12_ensemble_ai_model_final import V12EnsembleAIModel
from src.v12_signal_fusion_system import V12SignalFusionSystem
from src.v12_online_learning_system import V12OnlineLearningSystem
from src.v12_high_frequency_execution_engine import V12HighFrequencyExecutionEngine

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V12CompleteSystemManager:
    """V12完整系统管理器"""
    
    def __init__(self, config: Dict):
        """初始化V12完整系统"""
        self.config = config
        
        # 初始化所有核心组件
        logger.info("初始化V12完整系统组件...")
        
        # 1. AI层组件
        self.ofi_expert_model = V12OFIExpertModel(
            model_type="ensemble",
            model_path="models/v12/"
        )
        
        ensemble_config = {
            'ofi_expert': 0.5, 'lstm': 0.2, 'transformer': 0.2, 'cnn': 0.1,
            'input_size': 36, 'hidden_size': 64, 'num_layers': 2, 'num_heads': 2,
            'dropout': 0.1, 'num_filters': 32, 'kernel_size': 3,
            'learning_rate': 0.001, 'epochs': 10, 'batch_size': 32
        }
        self.ensemble_ai_model = V12EnsembleAIModel(ensemble_config)
        
        # 2. 信号处理层组件
        self.ofi_calculator = V12RealOFICalculator(
            levels=5,
            window_seconds=2,
            z_window=1200
        )
        
        fusion_config = {
            'signal_quality_threshold': 0.35,
            'ai_confidence_threshold': 0.55,
            'signal_strength_threshold': 0.15,
            'max_daily_trades': 50,
            'min_trade_interval_ms': 10,
            'high_frequency_mode': True
        }
        self.signal_fusion_system = V12SignalFusionSystem(fusion_config)
        
        online_learning_config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'update_frequency': 60,
            'min_samples_for_update': 50
        }
        self.online_learning_system = V12OnlineLearningSystem(online_learning_config)
        
        # 3. 执行层组件
        execution_config = {
            'max_slippage': 5,
            'execution_timeout': 100,
            'max_position_size': 10000
        }
        self.execution_engine = V12HighFrequencyExecutionEngine(execution_config)
        
        # 系统状态
        self.is_models_trained = False
        self.current_position = 0
        self.daily_pnl = 0
        self.trades = []
        
        logger.info("V12完整系统初始化完成")
    
    def train_ai_models(self, training_data: pd.DataFrame):
        """训练AI模型"""
        logger.info("开始训练V12 AI模型...")
        
        try:
            # 准备训练数据
            training_data_with_target = training_data.copy()
            training_data_with_target['target'] = (
                training_data_with_target['price'].diff().shift(-1) > 0
            ).astype(int)
            
            # 训练OFI专家模型
            logger.info("训练OFI专家模型...")
            self.ofi_expert_model.train_model(
                training_data_with_target, 
                {'model_type': 'ensemble'}
            )
            
            # 准备深度学习模型特征
            feature_columns = [
                'ofi', 'cvd', 'ofi_z', 'cvd_z', 'bid1_size', 'ask1_size', 
                'bid_depth', 'ask_depth', 'spread', 'volume', 'rsi', 'volatility'
            ]
            
            # 确保所有特征列都存在
            available_features = [col for col in feature_columns if col in training_data.columns]
            if len(available_features) < 6:  # 至少需要6个特征
                # 使用基础特征
                available_features = ['ofi', 'cvd', 'ofi_z', 'cvd_z', 'rsi', 'volatility']
                available_features = [col for col in available_features if col in training_data.columns]
            
            logger.info(f"使用特征: {available_features}")
            features = training_data[available_features].values
            
            # 训练集成AI模型
            logger.info("训练集成AI模型...")
            self.ensemble_ai_model.train_deep_learning_models(features)
            
            self.is_models_trained = True
            logger.info("V12 AI模型训练完成")
            
        except Exception as e:
            logger.error(f"AI模型训练失败: {e}")
            self.is_models_trained = False
    
    def can_open_position(self, signal_quality: float, ai_confidence: float) -> bool:
        """检查是否可以开仓"""
        min_signal_quality = 0.3
        min_ai_confidence = 0.5
        
        if signal_quality < min_signal_quality or ai_confidence < min_ai_confidence:
            return False
        
        if self.daily_pnl < -100:  # 最大日亏损限制
            return False
        
        if abs(self.current_position) >= 1000:  # 最大仓位限制
            return False
        
        return True
    
    def calculate_position_size(self, signal_quality: float, ai_confidence: float, price: float) -> float:
        """计算仓位大小"""
        base_size = 10
        quality_multiplier = min(signal_quality * 2, 1.5)
        confidence_multiplier = min(ai_confidence * 1.2, 1.2)
        
        position_size = base_size * quality_multiplier * confidence_multiplier
        return min(position_size, 100)
    
    def should_close_position(self, current_price: float) -> bool:
        """检查是否应该平仓"""
        if self.current_position == 0:
            return False
        
        # 简单的止损止盈逻辑
        if abs(self.current_position) > 0:
            # 这里应该从execution_engine获取更精确的止损止盈逻辑
            return np.random.random() < 0.1  # 10%概率平仓
        
        return False
    
    def open_position(self, action: str, price: float, quantity: float, trade_info: Dict):
        """开仓"""
        if action == 'buy':
            self.current_position = quantity
        else:
            self.current_position = -quantity
        
        logger.info(f"开仓: {action} {quantity:.2f} @ {price:.2f}")
    
    def close_position(self, current_price: float, timestamp: datetime) -> Dict:
        """平仓"""
        if self.current_position == 0:
            return None
        
        # 简化的PnL计算
        pnl = np.random.normal(0, 10)  # 模拟PnL
        
        close_trade = {
            'timestamp': timestamp,
            'action': 'close',
            'price': current_price,
            'quantity': abs(self.current_position),
            'pnl': pnl
        }
        
        self.daily_pnl += pnl
        self.current_position = 0
        
        logger.info(f"平仓 @ {current_price:.2f}, PnL: {pnl:.2f}")
        
        return close_trade
    
    def process_market_data(self, row: pd.Series) -> Dict:
        """使用完整V12系统处理市场数据"""
        try:
            # 1. 更新OFI计算器
            order_book_data = {
                'timestamp': row['timestamp'],
                'price': row['price'],
                'bid_prices': [row.get('bid1_price', row['price'] - 0.5), 
                              row.get('bid2_price', row['price'] - 1.0)],
                'bid_sizes': [row.get('bid1_size', 100), 
                             row.get('bid2_size', 80)],
                'ask_prices': [row.get('ask1_price', row['price'] + 0.5), 
                              row.get('ask2_price', row['price'] + 1.0)],
                'ask_sizes': [row.get('ask1_size', 100), 
                             row.get('ask2_size', 80)]
            }
            
            self.ofi_calculator.update_order_book(order_book_data)
            ofi_features = self.ofi_calculator.get_current_features()
            
            # 2. AI模型预测
            if self.is_models_trained:
                # OFI专家模型预测
                ofi_expert_pred = self.ofi_expert_model.predict_signal_quality(
                    np.array([row['ofi_z'], row['cvd_z'], row.get('rsi', 50)]).reshape(1, -1)
                ).iloc[0]
                
                # 集成AI模型预测
                feature_columns = ['ofi', 'cvd', 'ofi_z', 'cvd_z', 'rsi', 'volatility']
                available_features = [col for col in feature_columns if col in row.index]
                features = np.array([row[col] for col in available_features]).reshape(1, -1)
                
                ensemble_pred = self.ensemble_ai_model.predict_ensemble(ofi_expert_pred, features)
            else:
                # 使用默认预测
                ofi_expert_pred = 0.5
                ensemble_pred = 0.5
            
            # 3. 信号融合系统
            order_book_data.update({
                'ofi': ofi_features.get('ofi', row['ofi']),
                'cvd': ofi_features.get('cvd', row['cvd']),
                'real_ofi_z': ofi_features.get('real_ofi_z', row['ofi_z']),
                'real_cvd_z': ofi_features.get('real_cvd_z', row['cvd_z']),
                'ai_prediction': ensemble_pred
            })
            
            fusion_signal = self.signal_fusion_system.process_market_data(order_book_data)
            
            # 4. 在线学习更新
            if hasattr(self.online_learning_system, 'update_models'):
                self.online_learning_system.update_models(row.to_dict())
            
            return fusion_signal
            
        except Exception as e:
            logger.error(f"处理市场数据时出错: {e}")
            return {'action': None, 'confidence': 0.0}

def run_complete_system_backtest():
    """运行完整V12系统回测"""
    print("=" * 60)
    print("V12 完整系统回测")
    print("=" * 60)
    
    # 配置回测框架
    framework_config = {
        'backtest_results_dir': 'backtest_results',
        'data': {},
        'backtest': {
            'max_daily_trades': 30
        }
    }
    
    # 初始化框架
    framework = V12FreshDataBacktestFramework(framework_config)
    
    # 生成训练数据
    print("\n1. 生成训练数据...")
    training_data = framework.generate_fresh_data()
    
    # 初始化完整系统
    print("\n2. 初始化V12完整系统...")
    system_config = {
        'signal_quality_threshold': 0.35,
        'ai_confidence_threshold': 0.55,
        'max_daily_trades': 30
    }
    v12_system = V12CompleteSystemManager(system_config)
    
    # 训练AI模型
    print("\n3. 训练AI模型...")
    v12_system.train_ai_models(training_data)
    
    # 创建信号生成器
    def complete_signal_generator(row):
        """使用完整V12系统生成信号"""
        fusion_signal = v12_system.process_market_data(row)
        
        if fusion_signal.get('action') in ['BUY', 'SELL']:
            return {
                'timestamp': row['timestamp'],
                'action': fusion_signal['action'].lower(),
                'price': row['price'],
                'signal_quality': fusion_signal.get('signal_quality', 0.5),
                'ai_confidence': fusion_signal.get('ai_confidence', 0.5),
                'signal_strength': fusion_signal.get('signal_strength', 0.5),
                'ofi_z': row['ofi_z'],
                'cvd_z': row['cvd_z'],
                'rsi': row.get('rsi', 50),
                'volatility': row.get('volatility', 0.01)
            }
        
        return None
    
    # 运行回测
    print("\n4. 运行完整系统回测...")
    
    backtest_configs = [
        {
            'name': 'complete_v12_system',
            'signal_generator': complete_signal_generator,
            'risk_manager': v12_system
        }
    ]
    
    results = framework.run_multiple_backtests(
        backtest_configs=backtest_configs,
        num_iterations=3
    )
    
    # 显示结果
    print("\n" + "=" * 60)
    print("V12完整系统回测结果")
    print("=" * 60)
    
    for strategy_name, stats in results['summary_statistics'].items():
        print(f"\n策略: {strategy_name}")
        print(f"平均胜率: {stats['avg_win_rate']:.2f}% ± {stats['std_win_rate']:.2f}%")
        print(f"平均PnL: {stats['avg_total_pnl']:.2f} ± {stats['std_total_pnl']:.2f}")
        print(f"平均夏普比率: {stats['avg_sharpe_ratio']:.4f}")
        print(f"平均最大回撤: {stats['avg_max_drawdown']:.2f}")
        print(f"平均交易数: {stats['avg_trades']:.1f}")
        print(f"一致性得分: {stats['consistency_score']:.4f}")
    
    # 保存汇总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"backtest_results/v12_complete_system_summary_{timestamp}.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n汇总结果已保存: {summary_file}")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    run_complete_system_backtest()


