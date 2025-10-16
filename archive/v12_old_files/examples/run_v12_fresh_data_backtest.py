#!/usr/bin/env python3
"""
V12 新鲜数据回测示例
使用新鲜数据框架确保每次回测都使用全新数据
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
from src.v12_fresh_data_backtest_framework import V12FreshDataBacktestFramework
from src.v12_realistic_data_simulator import V12RealisticDataSimulator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V12SimpleRiskManager:
    """V12简单风险管理器"""
    
    def __init__(self, config: Dict):
        self.max_position_size = config.get('max_position_size', 100)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = config.get('take_profit_pct', 0.01)
        self.max_daily_loss = config.get('max_daily_loss', 100)
        
        self.current_position = 0
        self.entry_price = 0
        self.position_quantity = 0
        self.daily_pnl = 0
    
    def can_open_position(self, signal_quality: float, ai_confidence: float) -> bool:
        """检查是否可以开仓"""
        min_signal_quality = 0.3
        min_ai_confidence = 0.5
        
        if signal_quality < min_signal_quality or ai_confidence < min_ai_confidence:
            return False
        
        if self.daily_pnl < -self.max_daily_loss:
            return False
        
        if abs(self.current_position) >= self.max_position_size:
            return False
        
        return True
    
    def calculate_position_size(self, signal_quality: float, ai_confidence: float, price: float) -> float:
        """计算仓位大小"""
        base_size = 10
        quality_multiplier = min(signal_quality * 2, 1.5)
        confidence_multiplier = min(ai_confidence * 1.2, 1.2)
        
        position_size = base_size * quality_multiplier * confidence_multiplier
        return min(position_size, self.max_position_size)
    
    def should_close_position(self, current_price: float) -> bool:
        """检查是否应该平仓"""
        if self.current_position == 0:
            return False
        
        if self.current_position > 0:  # 多仓
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # 空仓
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        if pnl_pct < -self.stop_loss_pct or pnl_pct > self.take_profit_pct:
            return True
        
        return False
    
    def open_position(self, action: str, price: float, quantity: float, trade_info: Dict):
        """开仓"""
        if action == 'buy':
            self.current_position = quantity
        else:
            self.current_position = -quantity
        
        self.entry_price = price
        self.position_quantity = quantity
        
        logger.info(f"开仓: {action} {quantity:.2f} @ {price:.2f}")
    
    def close_position(self, current_price: float, timestamp: datetime) -> Dict:
        """平仓"""
        if self.current_position == 0:
            return None
        
        if self.current_position > 0:  # 平多仓
            pnl = (current_price - self.entry_price) * self.position_quantity
            close_action = 'close_buy'
        else:  # 平空仓
            pnl = (self.entry_price - current_price) * abs(self.position_quantity)
            close_action = 'close_sell'
        
        close_trade = {
            'timestamp': timestamp,
            'action': close_action,
            'price': current_price,
            'quantity': abs(self.position_quantity),
            'entry_price': self.entry_price,
            'pnl': pnl,
            'pnl_pct': pnl / (self.entry_price * abs(self.position_quantity))
        }
        
        self.daily_pnl += pnl
        self.current_position = 0
        self.entry_price = 0
        self.position_quantity = 0
        
        logger.info(f"平仓: {close_action} @ {current_price:.2f}, PnL: {pnl:.2f}")
        
        return close_trade

def create_signal_generator(thresholds: Dict):
    """创建信号生成器"""
    def signal_generator(row):
        try:
            # 基于OFI Z-score的信号质量
            ofi_signal_quality = abs(row['ofi_z']) / 3.0
            ofi_signal_quality = min(ofi_signal_quality, 1.0)
            
            # 基于CVD Z-score的AI置信度
            ai_confidence = abs(row['cvd_z']) / 3.0
            ai_confidence = min(ai_confidence, 1.0)
            
            # 计算信号强度
            signal_strength = abs(ofi_signal_quality - 0.5) * 2
            
            # 技术指标过滤
            rsi = row.get('rsi', 50)
            volatility = row.get('volatility', 0.01)
            
            rsi_filter = not (rsi > 85 or rsi < 15)
            volatility_filter = volatility > 0.003
            
            # 综合信号质量评分
            combined_signal_quality = (
                ofi_signal_quality * 0.4 +
                ai_confidence * 0.3 +
                signal_strength * 0.2 +
                (1.0 if rsi_filter else 0.0) * 0.1
            )
            
            # 检查所有条件
            if (combined_signal_quality >= thresholds['signal_quality'] and
                ai_confidence >= thresholds['ai_confidence'] and
                signal_strength >= thresholds['signal_strength'] and
                rsi_filter and volatility_filter):
                
                action = 'buy' if row['ofi_z'] > 0 else 'sell'
                
                return {
                    'timestamp': row['timestamp'],
                    'action': action,
                    'price': row['price'],
                    'signal_quality': combined_signal_quality,
                    'ai_confidence': ai_confidence,
                    'signal_strength': signal_strength,
                    'ofi_z': row['ofi_z'],
                    'cvd_z': row['cvd_z'],
                    'rsi': rsi,
                    'volatility': volatility
                }
            
            return None
            
        except Exception as e:
            return None
    
    return signal_generator

def run_fresh_data_backtest():
    """运行新鲜数据回测"""
    print("=" * 60)
    print("V12 新鲜数据回测")
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
    
    # 定义多个回测配置
    backtest_configs = [
        {
            'name': 'conservative',
            'signal_generator': create_signal_generator({
                'signal_quality': 0.4,
                'ai_confidence': 0.6,
                'signal_strength': 0.2
            }),
            'risk_manager': V12SimpleRiskManager({
                'max_position_size': 50,
                'stop_loss_pct': 0.015,
                'take_profit_pct': 0.008,
                'max_daily_loss': 50
            })
        },
        {
            'name': 'balanced',
            'signal_generator': create_signal_generator({
                'signal_quality': 0.3,
                'ai_confidence': 0.5,
                'signal_strength': 0.15
            }),
            'risk_manager': V12SimpleRiskManager({
                'max_position_size': 100,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.01,
                'max_daily_loss': 100
            })
        },
        {
            'name': 'aggressive',
            'signal_generator': create_signal_generator({
                'signal_quality': 0.25,
                'ai_confidence': 0.45,
                'signal_strength': 0.12
            }),
            'risk_manager': V12SimpleRiskManager({
                'max_position_size': 150,
                'stop_loss_pct': 0.025,
                'take_profit_pct': 0.012,
                'max_daily_loss': 150
            })
        }
    ]
    
    # 运行多次回测，每次使用新数据
    print("\n开始运行多次回测，每次使用全新数据...")
    
    results = framework.run_multiple_backtests(
        backtest_configs=backtest_configs,
        num_iterations=3
    )
    
    # 显示结果
    print("\n" + "=" * 60)
    print("新鲜数据回测结果汇总")
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
    summary_file = f"backtest_results/v12_fresh_data_summary_{timestamp}.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n汇总结果已保存: {summary_file}")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    run_fresh_data_backtest()
