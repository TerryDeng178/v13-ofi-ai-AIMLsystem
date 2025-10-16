#!/usr/bin/env python3
"""
V12 胜率优化回测
提高信号质量阈值，优化信号筛选逻辑，添加风险管理
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

class V12RiskManager:
    """V12风险管理器"""
    
    def __init__(self, config: Dict):
        self.max_position_size = config.get('max_position_size', 1000)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # 2%止损
        self.take_profit_pct = config.get('take_profit_pct', 0.01)  # 1%止盈
        self.max_daily_loss = config.get('max_daily_loss', 50)  # 最大日亏损
        self.current_position = 0
        self.position_value = 0
        self.daily_pnl = 0
        self.trades = []
    
    def can_open_position(self, signal_quality: float, ai_confidence: float) -> bool:
        """检查是否可以开仓"""
        # 基于信号质量和AI置信度的开仓条件
        min_signal_quality = 0.4
        min_ai_confidence = 0.6
        
        if signal_quality < min_signal_quality or ai_confidence < min_ai_confidence:
            return False
        
        # 检查日亏损限制
        if self.daily_pnl < -self.max_daily_loss:
            return False
        
        # 检查仓位限制
        if abs(self.current_position) >= self.max_position_size:
            return False
        
        return True
    
    def calculate_position_size(self, signal_quality: float, ai_confidence: float, price: float) -> float:
        """计算仓位大小"""
        # 基于信号质量和置信度计算仓位大小
        base_size = 100
        quality_multiplier = signal_quality * 2  # 0.4-0.8 -> 0.8-1.6
        confidence_multiplier = ai_confidence * 1.5  # 0.6-1.0 -> 0.9-1.5
        
        position_size = base_size * quality_multiplier * confidence_multiplier
        
        # 限制最大仓位
        return min(position_size, self.max_position_size)
    
    def should_close_position(self, current_price: float, entry_price: float, action: str) -> bool:
        """检查是否应该平仓"""
        if self.current_position == 0:
            return False
        
        # 计算当前PnL
        if action == 'buy' and self.current_position < 0:  # 平空仓
            pnl_pct = (entry_price - current_price) / entry_price
        elif action == 'sell' and self.current_position > 0:  # 平多仓
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            return False
        
        # 检查止损
        if pnl_pct < -self.stop_loss_pct:
            logger.info(f"触发止损: PnL={pnl_pct:.4f}")
            return True
        
        # 检查止盈
        if pnl_pct > self.take_profit_pct:
            logger.info(f"触发止盈: PnL={pnl_pct:.4f}")
            return True
        
        return False
    
    def update_position(self, action: str, size: float, price: float):
        """更新仓位"""
        if action == 'buy':
            self.current_position += size
        else:
            self.current_position -= size
        
        self.position_value = self.current_position * price

def generate_optimized_signal(row, signal_quality_threshold=0.35, ai_confidence_threshold=0.55, signal_strength_threshold=0.2):
    """
    生成优化后的信号
    """
    try:
        # 基于OFI Z-score的信号质量 (提高阈值)
        ofi_signal_quality = abs(row['ofi_z']) / 3.0
        ofi_signal_quality = min(ofi_signal_quality, 1.0)
        
        # 基于CVD Z-score的AI置信度 (提高阈值)
        ai_confidence = abs(row['cvd_z']) / 3.0
        ai_confidence = min(ai_confidence, 1.0)
        
        # 计算信号强度 (提高阈值)
        signal_strength = abs(ofi_signal_quality - 0.5) * 2
        
        # 添加技术指标过滤
        rsi = row.get('rsi', 50)
        volatility = row.get('volatility', 0.01)
        
        # RSI过滤：避免极端超买超卖
        rsi_filter = True
        if rsi > 80 or rsi < 20:
            rsi_filter = False
        
        # 波动率过滤：避免低波动率
        volatility_filter = volatility > 0.005
        
        # 综合信号质量评分
        combined_signal_quality = (
            ofi_signal_quality * 0.4 +
            ai_confidence * 0.3 +
            signal_strength * 0.2 +
            (1.0 if rsi_filter else 0.0) * 0.1
        )
        
        # 检查所有条件
        if (combined_signal_quality >= signal_quality_threshold and
            ai_confidence >= ai_confidence_threshold and
            signal_strength >= signal_strength_threshold and
            rsi_filter and
            volatility_filter):
            
            # 确定交易方向
            if row['ofi_z'] > 0:
                action = 'buy'
            else:
                action = 'sell'
            
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

def run_winrate_optimization_backtest():
    """运行胜率优化回测"""
    print("=" * 60)
    print("V12 胜率优化回测")
    print("=" * 60)
    
    # 生成回测数据
    print("\n1. 生成回测数据...")
    data_simulator = V12RealisticDataSimulator()
    backtest_data = data_simulator.generate_complete_dataset()
    
    print(f"回测数据形状: {backtest_data.shape}")
    print(f"OFI Z-score范围: {backtest_data['ofi_z'].min():.4f} - {backtest_data['ofi_z'].max():.4f}")
    print(f"CVD Z-score范围: {backtest_data['cvd_z'].min():.4f} - {backtest_data['cvd_z'].max():.4f}")
    
    # 优化后的参数
    signal_quality_threshold = 0.35  # 提高阈值
    ai_confidence_threshold = 0.55   # 提高阈值
    signal_strength_threshold = 0.2  # 提高阈值
    max_daily_trades = 50            # 降低交易数，提高质量
    
    print(f"信号质量阈值: {signal_quality_threshold}")
    print(f"AI置信度阈值: {ai_confidence_threshold}")
    print(f"信号强度阈值: {signal_strength_threshold}")
    print(f"最大日交易数: {max_daily_trades}")
    
    # 初始化风险管理器
    risk_config = {
        'max_position_size': 1000,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.01,
        'max_daily_loss': 50
    }
    risk_manager = V12RiskManager(risk_config)
    
    # 运行回测
    print("\n2. 开始回测...")
    
    trades = []
    signals_generated = 0
    signals_filtered = 0
    daily_trades = 0
    last_trade_day = None
    current_trade = None
    
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
            if current_trade and risk_manager.should_close_position(
                row['price'], current_trade['price'], current_trade['action']
            ):
                # 执行平仓
                close_trade = {
                    'timestamp': current_time,
                    'action': 'close_' + current_trade['action'],
                    'price': row['price'],
                    'quantity': abs(current_trade['quantity']),
                    'pnl': (row['price'] - current_trade['price']) * current_trade['quantity'],
                    'trade_type': 'close'
                }
                trades.append(close_trade)
                risk_manager.update_position(
                    'sell' if current_trade['action'] == 'buy' else 'buy',
                    abs(current_trade['quantity']),
                    row['price']
                )
                current_trade = None
            
            # 生成信号
            signal = generate_optimized_signal(
                row,
                signal_quality_threshold,
                ai_confidence_threshold,
                signal_strength_threshold
            )
            
            if signal:
                signals_generated += 1
                
                # 风险管理检查
                if risk_manager.can_open_position(
                    signal['signal_quality'],
                    signal['ai_confidence']
                ):
                    signals_filtered += 1
                    
                    # 计算仓位大小
                    position_size = risk_manager.calculate_position_size(
                        signal['signal_quality'],
                        signal['ai_confidence'],
                        signal['price']
                    )
                    
                    # 执行交易
                    trade_result = {
                        'timestamp': current_time,
                        'action': signal['action'],
                        'price': signal['price'],
                        'quantity': position_size,
                        'signal_quality': signal['signal_quality'],
                        'ai_confidence': signal['ai_confidence'],
                        'signal_strength': signal['signal_strength'],
                        'ofi_z': signal['ofi_z'],
                        'cvd_z': signal['cvd_z'],
                        'rsi': signal['rsi'],
                        'volatility': signal['volatility'],
                        'trade_type': 'open'
                    }
                    
                    trades.append(trade_result)
                    risk_manager.update_position(signal['action'], position_size, signal['price'])
                    current_trade = trade_result
                    daily_trades += 1
        
        except Exception as e:
            logger.error(f"处理第{i}行数据时出错: {e}")
            continue
    
    # 计算回测结果
    print("\n3. 计算回测结果...")
    
    if not trades:
        print("没有执行任何交易")
        return
    
    trades_df = pd.DataFrame(trades)
    
    # 计算基本指标
    total_trades = len(trades)
    open_trades = trades_df[trades_df['trade_type'] == 'open']
    close_trades = trades_df[trades_df['trade_type'] == 'close']
    
    # 计算PnL
    total_pnl = 0
    winning_trades = 0
    
    for i in range(len(trades_df)):
        if trades_df.iloc[i]['trade_type'] == 'close':
            pnl = trades_df.iloc[i]['pnl']
            total_pnl += pnl
            if pnl > 0:
                winning_trades += 1
    
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
    
    # 生成报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'backtest_type': 'V12_Winrate_Optimization_Backtest',
        'parameters': {
            'signal_quality_threshold': signal_quality_threshold,
            'ai_confidence_threshold': ai_confidence_threshold,
            'signal_strength_threshold': signal_strength_threshold,
            'max_daily_trades': max_daily_trades,
            'risk_config': risk_config
        },
        'results': {
            'total_trades': total_trades,
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
    }
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"backtest_results/v12_winrate_optimization_backtest_{timestamp}.json"
    
    os.makedirs('backtest_results', exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n回测报告已保存: {report_file}")
    
    # 显示结果
    print("\n" + "=" * 60)
    print("V12 胜率优化回测结果")
    print("=" * 60)
    print(f"总交易数: {total_trades}")
    print(f"开仓交易: {len(open_trades)}")
    print(f"平仓交易: {len(close_trades)}")
    print(f"总PnL: {total_pnl:.2f}")
    print(f"胜率: {win_rate:.2f}%")
    print(f"夏普比率: {sharpe_ratio:.4f}")
    print(f"最大回撤: {max_drawdown:.2f}")
    print(f"平均信号质量: {avg_signal_quality:.4f}")
    print(f"平均AI置信度: {avg_ai_confidence:.4f}")
    print(f"平均信号强度: {avg_signal_strength:.4f}")
    print(f"生成信号数: {signals_generated}")
    print(f"过滤信号数: {signals_filtered}")
    print(f"信号生成率: {signals_generated/len(backtest_data)*100:.2f}%")
    print(f"信号过滤率: {signals_filtered/max(signals_generated, 1)*100:.2f}%")
    print(f"处理数据点: {len(backtest_data)}")
    print("=" * 60)
    
    # 显示前几个交易
    if len(open_trades) > 0:
        print("\n前5个开仓交易:")
        for i, trade in enumerate(open_trades.head().to_dict('records')):
            print(f"交易{i+1}: {trade['action']} @ {trade['price']:.2f}, "
                  f"质量: {trade['signal_quality']:.4f}, "
                  f"置信度: {trade['ai_confidence']:.4f}")
    
    return report

if __name__ == "__main__":
    run_winrate_optimization_backtest()
