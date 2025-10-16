#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安V10实时交易系统
集成币安实时数据与V10深度学习系统，实现智能交易
"""

import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
from typing import Dict, List, Optional
from binance_integration import BinanceTradingBot

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceV10RealtimeTrading:
    """币安V10实时交易系统"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.bot = BinanceTradingBot(api_key, secret_key)
        self.symbol = "ETHUSDT"
        self.is_running = False
        
        # 数据存储
        self.market_data = []
        self.orderbook_data = []
        self.trade_data = []
        self.ofi_data = []
        self.signals = []
        self.trades = []
        
        # V10参数
        self.ofi_window = 100  # OFI计算窗口
        self.signal_threshold = 2.0  # 信号阈值
        self.position_size = 0.01  # 仓位大小
        self.max_position = 0.1  # 最大仓位
        
        # 交易状态
        self.current_position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.unrealized_pnl = 0.0
        
        # 性能统计
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
    def start(self):
        """启动实时交易系统"""
        logger.info("启动币安V10实时交易系统...")
        
        try:
            # 启动币安机器人
            self.bot.start()
            self.is_running = True
            
            # 启动数据处理线程
            self.data_thread = threading.Thread(target=self._data_processing_loop)
            self.data_thread.daemon = True
            self.data_thread.start()
            
            # 启动信号生成线程
            self.signal_thread = threading.Thread(target=self._signal_generation_loop)
            self.signal_thread.daemon = True
            self.signal_thread.start()
            
            # 启动交易执行线程
            self.trading_thread = threading.Thread(target=self._trading_execution_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            # 启动监控线程
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            logger.info("实时交易系统启动成功")
            
        except Exception as e:
            logger.error(f"启动实时交易系统失败: {e}")
            self.stop()
    
    def stop(self):
        """停止实时交易系统"""
        logger.info("停止币安V10实时交易系统...")
        self.is_running = False
        self.bot.stop()
        
        # 保存数据
        self._save_trading_data()
        
        # 显示最终统计
        self._show_final_stats()
    
    def _data_processing_loop(self):
        """数据处理循环"""
        while self.is_running:
            try:
                # 获取最新数据
                market_data = self.bot.get_market_data()
                orderbook_data = self.bot.get_orderbook_data()
                trade_data = self.bot.get_trade_data()
                
                # 处理市场数据
                if market_data:
                    for data in market_data[-5:]:  # 处理最近5条数据
                        self._process_market_data(data)
                
                # 处理订单簿数据
                if orderbook_data:
                    for data in orderbook_data[-5:]:  # 处理最近5条数据
                        self._process_orderbook_data(data)
                
                # 处理交易数据
                if trade_data:
                    for data in trade_data[-10:]:  # 处理最近10条数据
                        self._process_trade_data(data)
                
                time.sleep(0.1)  # 100ms间隔
                
            except Exception as e:
                logger.error(f"数据处理错误: {e}")
                time.sleep(1)
    
    def _process_market_data(self, data: Dict):
        """处理市场数据"""
        processed_data = {
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'price': data['price'],
            'bid': data['bid'],
            'ask': data['ask'],
            'volume': data['volume'],
            'change': data['change']
        }
        
        self.market_data.append(processed_data)
        
        # 保持最近1000条数据
        if len(self.market_data) > 1000:
            self.market_data = self.market_data[-1000:]
    
    def _process_orderbook_data(self, data: Dict):
        """处理订单簿数据"""
        try:
            # 计算OFI
            ofi = self._calculate_ofi(data)
            
            ofi_data = {
                'timestamp': data['timestamp'],
                'symbol': data['symbol'],
                'ofi': ofi,
                'bid_volume': sum([float(bid[1]) for bid in data['bids']]),
                'ask_volume': sum([float(ask[1]) for ask in data['asks']])
            }
            
            self.ofi_data.append(ofi_data)
            
            # 保持最近1000条数据
            if len(self.ofi_data) > 1000:
                self.ofi_data = self.ofi_data[-1000:]
                
        except Exception as e:
            logger.error(f"处理订单簿数据错误: {e}")
    
    def _process_trade_data(self, data: Dict):
        """处理交易数据"""
        trade_data = {
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'price': data['price'],
            'quantity': data['quantity'],
            'side': data['side']
        }
        
        self.trade_data.append(trade_data)
        
        # 保持最近1000条数据
        if len(self.trade_data) > 1000:
            self.trade_data = self.trade_data[-1000:]
    
    def _calculate_ofi(self, orderbook_data: Dict) -> float:
        """计算订单流不平衡"""
        try:
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']
            
            if not bids or not asks:
                return 0.0
            
            # 计算买卖压力
            bid_pressure = sum([float(bid[1]) for bid in bids])
            ask_pressure = sum([float(ask[1]) for ask in asks])
            
            # 计算OFI
            if bid_pressure + ask_pressure > 0:
                ofi = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure)
                return ofi
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"OFI计算错误: {e}")
            return 0.0
    
    def _signal_generation_loop(self):
        """信号生成循环"""
        while self.is_running:
            try:
                if len(self.ofi_data) < self.ofi_window:
                    time.sleep(1)
                    continue
                
                # 计算OFI Z-score
                ofi_values = [data['ofi'] for data in self.ofi_data[-self.ofi_window:]]
                if len(ofi_values) > 1:
                    ofi_mean = np.mean(ofi_values)
                    ofi_std = np.std(ofi_values)
                    
                    if ofi_std > 0:
                        ofi_z = (ofi_values[-1] - ofi_mean) / ofi_std
                        
                        # 生成信号
                        if abs(ofi_z) > self.signal_threshold:
                            signal = {
                                'timestamp': datetime.now(),
                                'symbol': self.symbol,
                                'side': 'BUY' if ofi_z > 0 else 'SELL',
                                'strength': abs(ofi_z),
                                'ofi_z': ofi_z,
                                'price': self.market_data[-1]['price'] if self.market_data else 0
                            }
                            
                            self.signals.append(signal)
                            logger.info(f"生成信号: {signal['side']} 强度: {signal['strength']:.2f} 价格: {signal['price']}")
                
                time.sleep(1)  # 1秒间隔
                
            except Exception as e:
                logger.error(f"信号生成错误: {e}")
                time.sleep(1)
    
    def _trading_execution_loop(self):
        """交易执行循环"""
        while self.is_running:
            try:
                if not self.signals:
                    time.sleep(1)
                    continue
                
                # 获取最新信号
                latest_signal = self.signals[-1]
                
                # 检查是否应该执行交易
                if self._should_execute_trade(latest_signal):
                    # 执行交易
                    result = self._execute_trade(latest_signal)
                    if result:
                        self.trades.append({
                            'timestamp': datetime.now(),
                            'signal': latest_signal,
                            'result': result
                        })
                        logger.info(f"执行交易: {latest_signal['side']} 结果: {result}")
                
                time.sleep(1)  # 1秒间隔
                
            except Exception as e:
                logger.error(f"交易执行错误: {e}")
                time.sleep(1)
    
    def _should_execute_trade(self, signal: Dict) -> bool:
        """判断是否应该执行交易"""
        # 检查是否有未成交订单
        open_orders = self.bot.get_open_orders()
        if open_orders:
            return False
        
        # 检查信号强度
        if signal['strength'] < self.signal_threshold:
            return False
        
        # 检查仓位限制
        if abs(self.current_position) >= self.max_position:
            return False
        
        # 检查信号时间（避免重复信号）
        if self.signals and len(self.signals) > 1:
            last_signal = self.signals[-2]
            time_diff = (signal['timestamp'] - last_signal['timestamp']).total_seconds()
            if time_diff < 10:  # 10秒内不重复交易
                return False
        
        return True
    
    def _execute_trade(self, signal: Dict) -> Optional[Dict]:
        """执行交易"""
        try:
            side = signal['side']
            quantity = self.position_size
            
            # 下市价单
            result = self.bot.place_market_order(side, quantity)
            
            if result and result.get('orderId'):
                # 更新仓位
                if side == 'BUY':
                    self.current_position += quantity
                else:
                    self.current_position -= quantity
                
                self.entry_price = signal['price']
                self.entry_time = signal['timestamp']
                
                # 更新统计
                self.total_trades += 1
                
                return result
            else:
                logger.error(f"交易失败: {result}")
                return None
                
        except Exception as e:
            logger.error(f"交易执行错误: {e}")
            return None
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 更新未实现盈亏
                if self.current_position != 0 and self.market_data:
                    current_price = self.market_data[-1]['price']
                    self.unrealized_pnl = (current_price - self.entry_price) * self.current_position
                
                # 每30秒显示一次状态
                if self.total_trades % 30 == 0:
                    self._show_status()
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"监控错误: {e}")
                time.sleep(1)
    
    def _show_status(self):
        """显示状态"""
        logger.info(f"交易状态:")
        logger.info(f"  当前仓位: {self.current_position:.4f}")
        logger.info(f"  未实现盈亏: {self.unrealized_pnl:.4f}")
        logger.info(f"  总交易数: {self.total_trades}")
        logger.info(f"  信号数量: {len(self.signals)}")
        logger.info(f"  OFI数据: {len(self.ofi_data)}")
    
    def _show_final_stats(self):
        """显示最终统计"""
        logger.info("="*60)
        logger.info("最终交易统计")
        logger.info("="*60)
        logger.info(f"总交易数: {self.total_trades}")
        logger.info(f"信号数量: {len(self.signals)}")
        logger.info(f"OFI数据: {len(self.ofi_data)}")
        logger.info(f"市场数据: {len(self.market_data)}")
        logger.info(f"订单簿数据: {len(self.orderbook_data)}")
        logger.info(f"交易数据: {len(self.trade_data)}")
        
        if self.ofi_data:
            ofi_values = [data['ofi'] for data in self.ofi_data]
            logger.info(f"OFI统计:")
            logger.info(f"  最小值: {min(ofi_values):.6f}")
            logger.info(f"  最大值: {max(ofi_values):.6f}")
            logger.info(f"  平均值: {np.mean(ofi_values):.6f}")
            logger.info(f"  标准差: {np.std(ofi_values):.6f}")
    
    def _save_trading_data(self):
        """保存交易数据"""
        try:
            # 保存OFI数据
            if self.ofi_data:
                ofi_df = pd.DataFrame(self.ofi_data)
                ofi_filename = f"binance_ofi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                ofi_df.to_csv(ofi_filename, index=False)
                logger.info(f"OFI数据已保存到: {ofi_filename}")
            
            # 保存信号数据
            if self.signals:
                signals_df = pd.DataFrame(self.signals)
                signals_filename = f"binance_signals_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                signals_df.to_csv(signals_filename, index=False)
                logger.info(f"信号数据已保存到: {signals_filename}")
            
            # 保存交易数据
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_filename = f"binance_trades_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                trades_df.to_csv(trades_filename, index=False)
                logger.info(f"交易数据已保存到: {trades_filename}")
                
        except Exception as e:
            logger.error(f"保存数据错误: {e}")

def main():
    """主函数"""
    # 币安测试网API配置
    API_KEY = "tKs2nQW3t3rcHnUCGOPWLIYsDotF6ot1GRZzo5n3QOsNRqbPTxO06TC8QglxtUYy"
    SECRET_KEY = "I5SqluL8yqvkcSSb0hBBMr80cHmNndxrl6DAufw9t6myjlRBQDvPxUmabHNJpkZJ"
    
    # 创建实时交易系统
    trading_system = BinanceV10RealtimeTrading(API_KEY, SECRET_KEY)
    
    try:
        # 启动系统
        trading_system.start()
        
        # 运行一段时间
        logger.info("系统运行中...")
        time.sleep(300)  # 运行5分钟
        
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        # 停止系统
        trading_system.stop()

if __name__ == "__main__":
    main()
