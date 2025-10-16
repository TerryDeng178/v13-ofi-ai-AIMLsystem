#!/usr/bin/env python3
"""
V12 超简单回测
使用极低阈值进行测试
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ultra_simple_backtest():
    """运行超简单回测"""
    print("=" * 60)
    print("V12 超简单回测")
    print("=" * 60)
    
    # 生成回测数据
    print("\n1. 生成回测数据...")
    data_simulator = V12RealisticDataSimulator()
    backtest_data = data_simulator.generate_complete_dataset()
    
    print(f"回测数据形状: {backtest_data.shape}")
    print(f"OFI Z-score范围: {backtest_data['ofi_z'].min():.4f} - {backtest_data['ofi_z'].max():.4f}")
    print(f"CVD Z-score范围: {backtest_data['cvd_z'].min():.4f} - {backtest_data['cvd_z'].max():.4f}")
    
    # 超低阈值参数
    signal_quality_threshold = 0.1   # 极低阈值
    ai_confidence_threshold = 0.1    # 极低阈值
    signal_strength_threshold = 0.05 # 极低阈值
    max_daily_trades = 100           # 增加交易数
    
    print(f"信号质量阈值: {signal_quality_threshold}")
    print(f"AI置信度阈值: {ai_confidence_threshold}")
    print(f"信号强度阈值: {signal_strength_threshold}")
    print(f"最大日交易数: {max_daily_trades}")
    
    # 运行回测
    print("\n2. 开始回测...")
    
    trades = []
    signals_generated = 0
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
        if daily_trades >= max_daily_trades:
            continue
        
        try:
            # 超简单信号生成逻辑
            ofi_signal_quality = abs(row['ofi_z']) / 3.0  # 标准化到0-1
            ofi_signal_quality = min(ofi_signal_quality, 1.0)
            
            ai_confidence = abs(row['cvd_z']) / 3.0  # 标准化到0-1
            ai_confidence = min(ai_confidence, 1.0)
            
            signal_strength = abs(ofi_signal_quality - 0.5) * 2  # 转换为0-1范围
            
            # 检查阈值
            if (ofi_signal_quality >= signal_quality_threshold and 
                ai_confidence >= ai_confidence_threshold and 
                signal_strength >= signal_strength_threshold):
                
                signals_generated += 1
                
                # 确定交易方向
                if row['ofi_z'] > 0:
                    action = 'buy'
                else:
                    action = 'sell'
                
                # 执行交易
                trade_result = {
                    'timestamp': current_time,
                    'action': action,
                    'price': row['price'],
                    'quantity': 1.0,
                    'signal_quality': ofi_signal_quality,
                    'ai_confidence': ai_confidence,
                    'signal_strength': signal_strength,
                    'ofi_z': row['ofi_z'],
                    'cvd_z': row['cvd_z']
                }
                
                trades.append(trade_result)
                daily_trades += 1
        
        except Exception as e:
            logger.error(f"处理第{i}行数据时出错: {e}")
            continue
    
    # 计算回测结果
    print("\n3. 计算回测结果...")
    
    if not trades:
        print("没有执行任何交易")
        print("尝试进一步降低阈值...")
        
        # 尝试更低的阈值
        signal_quality_threshold = 0.01
        ai_confidence_threshold = 0.01
        signal_strength_threshold = 0.01
        
        print(f"新阈值: 信号质量={signal_quality_threshold}, AI置信度={ai_confidence_threshold}, 信号强度={signal_strength_threshold}")
        
        for i, row in backtest_data.iterrows():
            if len(trades) >= 10:  # 限制交易数量
                break
                
            try:
                ofi_signal_quality = abs(row['ofi_z']) / 3.0
                ofi_signal_quality = min(ofi_signal_quality, 1.0)
                
                ai_confidence = abs(row['cvd_z']) / 3.0
                ai_confidence = min(ai_confidence, 1.0)
                
                signal_strength = abs(ofi_signal_quality - 0.5) * 2
                
                if (ofi_signal_quality >= signal_quality_threshold and 
                    ai_confidence >= ai_confidence_threshold and 
                    signal_strength >= signal_strength_threshold):
                    
                    action = 'buy' if row['ofi_z'] > 0 else 'sell'
                    
                    trade_result = {
                        'timestamp': row['timestamp'],
                        'action': action,
                        'price': row['price'],
                        'quantity': 1.0,
                        'signal_quality': ofi_signal_quality,
                        'ai_confidence': ai_confidence,
                        'signal_strength': signal_strength,
                        'ofi_z': row['real_ofi_z'],
                        'cvd_z': row['real_cvd_z']
                    }
                    
                    trades.append(trade_result)
            
            except Exception as e:
                continue
    
    if not trades:
        print("即使使用极低阈值也没有交易，可能存在数据问题")
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
    
    # 计算平均指标
    avg_signal_quality = trades_df['signal_quality'].mean()
    avg_ai_confidence = trades_df['ai_confidence'].mean()
    avg_signal_strength = trades_df['signal_strength'].mean()
    
    # 生成报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'backtest_type': 'V12_Ultra_Simple_Backtest',
        'parameters': {
            'signal_quality_threshold': signal_quality_threshold,
            'ai_confidence_threshold': ai_confidence_threshold,
            'signal_strength_threshold': signal_strength_threshold,
            'max_daily_trades': max_daily_trades
        },
        'results': {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_signal_quality': avg_signal_quality,
            'avg_ai_confidence': avg_ai_confidence,
            'avg_signal_strength': avg_signal_strength,
            'signals_generated': signals_generated,
            'signal_generation_rate': signals_generated / len(backtest_data) * 100,
            'data_points_processed': len(backtest_data)
        }
    }
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"backtest_results/v12_ultra_simple_backtest_{timestamp}.json"
    
    os.makedirs('backtest_results', exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n回测报告已保存: {report_file}")
    
    # 显示结果
    print("\n" + "=" * 60)
    print("V12 超简单回测结果")
    print("=" * 60)
    print(f"总交易数: {total_trades}")
    print(f"买入交易: {buy_trades}")
    print(f"卖出交易: {sell_trades}")
    print(f"总PnL: {total_pnl:.2f}")
    print(f"胜率: {win_rate:.2f}%")
    print(f"平均信号质量: {avg_signal_quality:.4f}")
    print(f"平均AI置信度: {avg_ai_confidence:.4f}")
    print(f"平均信号强度: {avg_signal_strength:.4f}")
    print(f"生成信号数: {signals_generated}")
    print(f"信号生成率: {signals_generated/len(backtest_data)*100:.2f}%")
    print(f"处理数据点: {len(backtest_data)}")
    print("=" * 60)
    
    # 显示前几个交易
    if len(trades) > 0:
        print("\n前5个交易:")
        for i, trade in enumerate(trades[:5]):
            print(f"交易{i+1}: {trade['action']} @ {trade['price']:.2f}, "
                  f"OFI Z: {trade['ofi_z']:.4f}, CVD Z: {trade['cvd_z']:.4f}")
    
    return report

if __name__ == "__main__":
    run_ultra_simple_backtest()
