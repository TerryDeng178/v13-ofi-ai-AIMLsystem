#!/usr/bin/env python3
"""
V12 修复风险管理回测
修复PnL计算、仓位管理和平仓逻辑问题
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

class V12FixedRiskManager:
    """V12修复版风险管理器"""
    
    def __init__(self, config: Dict):
        self.max_position_size = config.get('max_position_size', 100)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # 2%止损
        self.take_profit_pct = config.get('take_profit_pct', 0.01)  # 1%止盈
        self.max_daily_loss = config.get('max_daily_loss', 100)  # 最大日亏损
        self.current_position = 0
        self.position_value = 0
        self.entry_price = 0
        self.position_quantity = 0
        self.daily_pnl = 0
        self.trades = []
        self.open_trades = []  # 跟踪开仓交易
    
    def can_open_position(self, signal_quality: float, ai_confidence: float) -> bool:
        """检查是否可以开仓"""
        # 基于信号质量和AI置信度的开仓条件
        min_signal_quality = 0.3
        min_ai_confidence = 0.5
        
        if signal_quality < min_signal_quality or ai_confidence < min_ai_confidence:
            return False
        
        # 检查日亏损限制
        if self.daily_pnl < -self.max_daily_loss:
            return False
        
        # 检查仓位限制 - 简化逻辑
        if abs(self.current_position) >= self.max_position_size:
            return False
        
        return True
    
    def calculate_position_size(self, signal_quality: float, ai_confidence: float, price: float) -> float:
        """计算仓位大小 - 简化版本"""
        # 固定小仓位，避免风险
        base_size = 10
        quality_multiplier = min(signal_quality * 2, 1.5)  # 限制最大倍数
        confidence_multiplier = min(ai_confidence * 1.2, 1.2)
        
        position_size = base_size * quality_multiplier * confidence_multiplier
        
        # 限制最大仓位
        return min(position_size, self.max_position_size)
    
    def open_position(self, action: str, price: float, quantity: float, trade_info: Dict):
        """开仓"""
        if action == 'buy':
            self.current_position = quantity
            self.position_value = quantity * price
        else:
            self.current_position = -quantity
            self.position_value = -quantity * price
        
        self.entry_price = price
        self.position_quantity = quantity
        
        # 记录开仓交易
        open_trade = {
            'timestamp': trade_info['timestamp'],
            'action': action,
            'price': price,
            'quantity': quantity,
            'entry_price': price,
            'trade_info': trade_info
        }
        self.open_trades.append(open_trade)
        
        logger.info(f"开仓: {action} {quantity} @ {price}")
    
    def should_close_position(self, current_price: float) -> bool:
        """检查是否应该平仓"""
        if self.current_position == 0:
            return False
        
        # 计算当前PnL百分比
        if self.current_position > 0:  # 多仓
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # 空仓
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # 检查止损
        if pnl_pct < -self.stop_loss_pct:
            logger.info(f"触发止损: PnL={pnl_pct:.4f}")
            return True
        
        # 检查止盈
        if pnl_pct > self.take_profit_pct:
            logger.info(f"触发止盈: PnL={pnl_pct:.4f}")
            return True
        
        return False
    
    def close_position(self, current_price: float, timestamp: datetime) -> Dict:
        """平仓"""
        if self.current_position == 0:
            return None
        
        # 计算PnL
        if self.current_position > 0:  # 平多仓
            pnl = (current_price - self.entry_price) * self.position_quantity
            close_action = 'close_buy'
        else:  # 平空仓
            pnl = (self.entry_price - current_price) * abs(self.position_quantity)
            close_action = 'close_sell'
        
        # 记录平仓交易
        close_trade = {
            'timestamp': timestamp,
            'action': close_action,
            'price': current_price,
            'quantity': abs(self.position_quantity),
            'entry_price': self.entry_price,
            'pnl': pnl,
            'pnl_pct': pnl / (self.entry_price * abs(self.position_quantity))
        }
        
        # 更新日PnL
        self.daily_pnl += pnl
        
        # 重置仓位
        self.current_position = 0
        self.position_value = 0
        self.entry_price = 0
        self.position_quantity = 0
        
        logger.info(f"平仓: {close_action} @ {current_price}, PnL: {pnl:.2f}")
        
        return close_trade

def generate_optimized_signal(row, signal_quality_threshold=0.3, ai_confidence_threshold=0.5, signal_strength_threshold=0.15):
    """
    生成优化后的信号 - 降低阈值
    """
    try:
        # 基于OFI Z-score的信号质量
        ofi_signal_quality = abs(row['ofi_z']) / 3.0
        ofi_signal_quality = min(ofi_signal_quality, 1.0)
        
        # 基于CVD Z-score的AI置信度
        ai_confidence = abs(row['cvd_z']) / 3.0
        ai_confidence = min(ai_confidence, 1.0)
        
        # 计算信号强度
        signal_strength = abs(ofi_signal_quality - 0.5) * 2
        
        # 添加技术指标过滤
        rsi = row.get('rsi', 50)
        volatility = row.get('volatility', 0.01)
        
        # RSI过滤：避免极端超买超卖
        rsi_filter = True
        if rsi > 85 or rsi < 15:  # 放宽RSI过滤
            rsi_filter = False
        
        # 波动率过滤：避免极低波动率
        volatility_filter = volatility > 0.003  # 降低波动率要求
        
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

def run_fixed_risk_backtest():
    """运行修复风险管理回测"""
    print("=" * 60)
    print("V12 修复风险管理回测")
    print("=" * 60)
    
    # 生成全新的回测数据（避免数据泄露）
    print("\n1. 生成全新回测数据...")
    # 使用当前时间戳作为随机种子，确保每次都是新数据
    import time
    random_seed = int(time.time() * 1000) % 1000000
    print(f"使用随机种子: {random_seed}")
    
    data_simulator = V12RealisticDataSimulator(seed=random_seed)
    backtest_data = data_simulator.generate_complete_dataset()
    
    print(f"回测数据形状: {backtest_data.shape}")
    print(f"OFI Z-score范围: {backtest_data['ofi_z'].min():.4f} - {backtest_data['ofi_z'].max():.4f}")
    print(f"CVD Z-score范围: {backtest_data['cvd_z'].min():.4f} - {backtest_data['cvd_z'].max():.4f}")
    
    # 修复后的参数
    signal_quality_threshold = 0.3   # 降低阈值
    ai_confidence_threshold = 0.5    # 降低阈值
    signal_strength_threshold = 0.15 # 降低阈值
    max_daily_trades = 30            # 减少交易数
    
    print(f"信号质量阈值: {signal_quality_threshold}")
    print(f"AI置信度阈值: {ai_confidence_threshold}")
    print(f"信号强度阈值: {signal_strength_threshold}")
    print(f"最大日交易数: {max_daily_trades}")
    
    # 初始化风险管理器
    risk_config = {
        'max_position_size': 100,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.01,
        'max_daily_loss': 100
    }
    risk_manager = V12FixedRiskManager(risk_config)
    
    # 运行回测
    print("\n2. 开始回测...")
    
    all_trades = []
    signals_generated = 0
    signals_filtered = 0
    daily_trades = 0
    last_trade_day = None
    
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
            if risk_manager.should_close_position(row['price']):
                close_trade = risk_manager.close_position(row['price'], current_time)
                if close_trade:
                    all_trades.append(close_trade)
            
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
                    
                    # 执行开仓
                    risk_manager.open_position(
                        signal['action'],
                        signal['price'],
                        position_size,
                        signal
                    )
                    
                    # 记录开仓交易
                    open_trade = {
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
                    all_trades.append(open_trade)
                    daily_trades += 1
        
        except Exception as e:
            logger.error(f"处理第{i}行数据时出错: {e}")
            continue
    
    # 强制平仓所有未平仓位
    if risk_manager.current_position != 0:
        final_price = backtest_data.iloc[-1]['price']
        close_trade = risk_manager.close_position(final_price, backtest_data.iloc[-1]['timestamp'])
        if close_trade:
            all_trades.append(close_trade)
    
    # 计算回测结果
    print("\n3. 计算回测结果...")
    
    if not all_trades:
        print("没有执行任何交易")
        return
    
    trades_df = pd.DataFrame(all_trades)
    
    # 分离开仓和平仓交易
    open_trades = trades_df[trades_df['trade_type'] == 'open']
    close_trades = trades_df[trades_df['trade_type'].isnull()]  # 平仓交易没有trade_type
    
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
    
    # 生成报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'backtest_type': 'V12_Fixed_Risk_Backtest',
        'parameters': {
            'signal_quality_threshold': signal_quality_threshold,
            'ai_confidence_threshold': ai_confidence_threshold,
            'signal_strength_threshold': signal_strength_threshold,
            'max_daily_trades': max_daily_trades,
            'risk_config': risk_config
        },
        'results': {
            'total_trades': len(all_trades),
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
    report_file = f"backtest_results/v12_fixed_risk_backtest_{timestamp}.json"
    
    os.makedirs('backtest_results', exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n回测报告已保存: {report_file}")
    
    # 显示结果
    print("\n" + "=" * 60)
    print("V12 修复风险管理回测结果")
    print("=" * 60)
    print(f"总交易数: {len(all_trades)}")
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
    
    if len(close_trades) > 0:
        print("\n平仓交易统计:")
        print(f"盈利交易: {winning_trades}")
        print(f"亏损交易: {len(close_trades) - winning_trades}")
        print(f"平均盈利: {close_trades[close_trades['pnl'] > 0]['pnl'].mean():.2f}")
        print(f"平均亏损: {close_trades[close_trades['pnl'] < 0]['pnl'].mean():.2f}")
    
    return report

if __name__ == "__main__":
    run_fixed_risk_backtest()
