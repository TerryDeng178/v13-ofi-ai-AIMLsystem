#!/usr/bin/env python3
"""
V12 使用已训练AI模型的回测
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

# 导入V12组件
from src.v12_realistic_data_simulator import V12RealisticDataSimulator
from src.v12_strict_validation_framework import V12StrictValidationFramework
from src.v12_ofi_expert_model import V12OFIExpertModel
from src.v12_ensemble_ai_model_final import V12EnsembleAIModel
from src.v12_signal_fusion_system import V12SignalFusionSystem
from src.v12_online_learning_system import V12OnlineLearningSystem
from src.v12_high_frequency_execution_engine import V12HighFrequencyExecutionEngine
from src.v12_real_ofi_calculator import V12RealOFICalculator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_trained_ai_models():
    """创建并训练AI模型"""
    print("=" * 60)
    print("创建并训练AI模型")
    print("=" * 60)
    
    # 创建数据模拟器
    data_simulator = V12RealisticDataSimulator()
    data = data_simulator.generate_complete_dataset()
    
    print(f"生成训练数据: {data.shape}")
    
    # 创建OFI专家模型
    print("\n1. 创建并训练OFI专家模型...")
    ofi_model = V12OFIExpertModel(model_type="ensemble")
    
    # 训练OFI模型
    params = {'model_type': 'ensemble'}
    ofi_model.train_model(data, params)
    
    print(f"OFI模型训练状态: {ofi_model.is_trained}")
    
    # 创建集成AI模型
    print("\n2. 创建并训练集成AI模型...")
    ensemble_config = {
        'device': 'cuda',
        'input_size': 9,  # 根据实际特征数量调整
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'ensemble_weights': {
            'ofi_expert': 0.4,
            'lstm': 0.2,
            'transformer': 0.2,
            'cnn': 0.2
        }
    }
    
    ensemble_model = V12EnsembleAIModel(ensemble_config)
    
    # 准备特征数据
    feature_columns = ['ofi', 'cvd', 'real_ofi_z', 'real_cvd_z'] + [f'ofi_{i}' for i in range(1, 6)]
    available_features = [col for col in feature_columns if col in data.columns]
    features = data[available_features].values
    
    print(f"使用特征: {available_features}")
    print(f"特征维度: {features.shape}")
    
    # 训练集成模型
    ensemble_model.train_deep_learning_models(features)
    
    print(f"集成模型训练状态: {ensemble_model.is_trained}")
    
    return ofi_model, ensemble_model, data

def run_trained_ai_backtest():
    """运行使用已训练AI模型的回测"""
    print("=" * 60)
    print("V12 使用已训练AI模型的回测")
    print("=" * 60)
    
    # 创建并训练AI模型
    ofi_model, ensemble_model, training_data = create_trained_ai_models()
    
    # 创建新数据用于回测
    print("\n3. 生成回测数据...")
    data_simulator = V12RealisticDataSimulator()
    backtest_data = data_simulator.generate_complete_dataset()
    
    print(f"回测数据形状: {backtest_data.shape}")
    
    # 创建其他组件
    print("\n4. 初始化系统组件...")
    
    # 验证框架
    validation_framework = V12StrictValidationFramework()
    
    # OFI计算器
    ofi_calculator = V12RealOFICalculator(
        levels=5,
        window_seconds=2,
        z_window=1200
    )
    
    # 信号融合系统
    fusion_config = {
        'signal_quality_threshold': 0.35,  # 降低阈值
        'ai_confidence_threshold': 0.55,   # 降低阈值
        'signal_strength_threshold': 0.15, # 降低阈值
        'max_daily_trades': 50,            # 增加交易数
        'high_frequency_mode': True
    }
    
    signal_fusion = V12SignalFusionSystem(fusion_config)
    
    # 在线学习系统
    online_learning_config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'update_frequency': 60,
        'min_samples_for_update': 50
    }
    online_learning = V12OnlineLearningSystem(online_learning_config)
    
    # 高频执行引擎
    execution_config = {
        'max_slippage': 5,
        'execution_timeout': 100,
        'position_limit': 100000
    }
    
    execution_engine = V12HighFrequencyExecutionEngine(execution_config)
    
    # 运行回测
    print("\n5. 开始回测...")
    
    trades = []
    positions = []
    current_position = 0
    daily_trades = 0
    last_trade_day = None
    
    for i, row in backtest_data.iterrows():
        current_time = row['timestamp']
        current_day = current_time.date()
        
        # 重置日交易计数
        if last_trade_day != current_day:
            daily_trades = 0
            last_trade_day = current_day
        
        # 检查日交易限制
        if daily_trades >= 50:
            continue
        
        try:
            # 准备特征数据
            feature_columns = ['ofi', 'cvd', 'real_ofi_z', 'real_cvd_z'] + [f'ofi_{i}' for i in range(1, 6)]
            available_features = [col for col in feature_columns if col in backtest_data.columns]
            
            if len(available_features) < 4:
                continue
                
            current_features = backtest_data[available_features].iloc[i:i+1].values
            
            # 获取OFI专家模型预测
            ofi_prediction = ofi_model.predict_signal_quality(backtest_data.iloc[i:i+1])
            ofi_signal_quality = float(ofi_prediction.iloc[0]) if hasattr(ofi_prediction, 'iloc') else float(ofi_prediction)
            
            # 获取集成AI模型预测
            ensemble_prediction = ensemble_model.predict_ensemble(current_features)
            ai_confidence = float(ensemble_prediction)
            
            # 生成融合信号
            signal_data = {
                'ofi_signal_quality': ofi_signal_quality,
                'ai_confidence': ai_confidence,
                'signal_strength': abs(ofi_signal_quality - 0.5) * 2,  # 转换为0-1范围
                'timestamp': current_time,
                'price': row['price']
            }
            
            fused_signal = signal_fusion.generate_fused_signal(signal_data)
            
            if fused_signal and fused_signal['action'] != 'hold':
                # 验证信号
                if validation_framework.validate_signal(fused_signal, current_position):
                    # 执行交易
                    trade_result = execution_engine.execute_order({
                        'symbol': 'ETHUSDT',
                        'side': fused_signal['action'],
                        'quantity': 1.0,
                        'price': row['price'],
                        'timestamp': current_time
                    })
                    
                    if trade_result['status'] == 'filled':
                        trades.append({
                            'timestamp': current_time,
                            'action': fused_signal['action'],
                            'price': row['price'],
                            'quantity': 1.0,
                            'signal_quality': ofi_signal_quality,
                            'ai_confidence': ai_confidence
                        })
                        
                        current_position += 1 if fused_signal['action'] == 'buy' else -1
                        daily_trades += 1
                        
                        # 在线学习
                        online_learning.update_model({
                            'features': current_features[0],
                            'signal': fused_signal['action'],
                            'outcome': 'pending'
                        })
        
        except Exception as e:
            logger.error(f"处理第{i}行数据时出错: {e}")
            continue
    
    # 计算回测结果
    print("\n6. 计算回测结果...")
    
    if not trades:
        print("没有执行任何交易")
        return
    
    trades_df = pd.DataFrame(trades)
    
    # 计算基本指标
    total_trades = len(trades)
    buy_trades = len(trades_df[trades_df['action'] == 'buy'])
    sell_trades = len(trades_df[trades_df['action'] == 'sell'])
    
    # 计算PnL (简化版本)
    total_pnl = 0
    winning_trades = 0
    
    for i in range(1, len(trades_df)):
        prev_trade = trades_df.iloc[i-1]
        curr_trade = trades_df.iloc[i]
        
        if prev_trade['action'] == 'buy' and curr_trade['action'] == 'sell':
            pnl = curr_trade['price'] - prev_trade['price']
            total_pnl += pnl
            if pnl > 0:
                winning_trades += 1
    
    win_rate = (winning_trades / max(total_trades // 2, 1)) * 100
    
    # 计算平均信号质量
    avg_signal_quality = trades_df['signal_quality'].mean()
    avg_ai_confidence = trades_df['ai_confidence'].mean()
    
    # 生成报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'backtest_type': 'V12_Trained_AI_Backtest',
        'results': {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_signal_quality': avg_signal_quality,
            'avg_ai_confidence': avg_ai_confidence,
            'data_points_processed': len(backtest_data)
        },
        'model_status': {
            'ofi_model_trained': ofi_model.is_trained,
            'ensemble_model_trained': ensemble_model.is_trained
        },
        'parameters': fusion_config
    }
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"backtest_results/v12_trained_ai_backtest_{timestamp}.json"
    
    os.makedirs('backtest_results', exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n回测报告已保存: {report_file}")
    
    # 显示结果
    print("\n" + "=" * 60)
    print("V12 已训练AI模型回测结果")
    print("=" * 60)
    print(f"总交易数: {total_trades}")
    print(f"买入交易: {buy_trades}")
    print(f"卖出交易: {sell_trades}")
    print(f"总PnL: {total_pnl:.2f}")
    print(f"胜率: {win_rate:.2f}%")
    print(f"平均信号质量: {avg_signal_quality:.4f}")
    print(f"平均AI置信度: {avg_ai_confidence:.4f}")
    print(f"处理数据点: {len(backtest_data)}")
    print("=" * 60)
    
    return report

if __name__ == "__main__":
    run_trained_ai_backtest()
