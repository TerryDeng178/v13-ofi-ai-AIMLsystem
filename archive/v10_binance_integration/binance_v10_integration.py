#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安V10集成系统
将币安实时数据与V10深度学习系统集成
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import json
import threading
from binance_integration import BinanceTradingBot

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceV10Integration:
    """币安V10集成系统"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.bot = BinanceTradingBot(api_key, secret_key)
        self.symbol = "ETHUSDT"
        self.data_buffer = []
        self.ofi_data = []
        self.signals = []
        self.trades = []
        self.is_running = False
        
        # V10参数
        self.ofi_window = 100  # OFI计算窗口
        self.signal_threshold = 2.0  # 信号阈值
        self.position_size = 0.01  # 仓位大小
        
    def start(self):
        """启动集成系统"""
        logger.info("启动币安V10集成系统...")
        
        # 启动币安机器人
        self.bot.start()
        self.is_running = True
        
        # 启动数据处理线程
        self.data_thread = threading.Thread(target=self._process_data_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # 启动信号生成线程
        self.signal_thread = threading.Thread(target=self._generate_signals_loop)
        self.signal_thread.daemon = True
        self.signal_thread.start()
        
        # 启动交易执行线程
        self.trading_thread = threading.Thread(target=self._execute_trades_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
    def stop(self):
        """停止集成系统"""
        logger.info("停止币安V10集成系统...")
        self.is_running = False
        self.bot.stop()
        
    def _process_data_loop(self):
        """数据处理循环"""
        while self.is_running:
            try:
                # 获取最新市场数据
                market_data = self.bot.get_market_data()
                orderbook_data = self.bot.get_orderbook_data()
                trade_data = self.bot.get_trade_data()
                
                if market_data:
                    latest_market = market_data[-1]
                    self._process_market_data(latest_market)
                
                if orderbook_data:
                    latest_orderbook = orderbook_data[-1]
                    self._process_orderbook_data(latest_orderbook)
                
                if trade_data:
                    for trade in trade_data[-10:]:  # 处理最近10笔交易
                        self._process_trade_data(trade)
                
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
            'change': data['change'],
            'high': data['high'],
            'low': data['low']
        }
        
        self.data_buffer.append(processed_data)
        
        # 保持最近1000条数据
        if len(self.data_buffer) > 1000:
            self.data_buffer = self.data_buffer[-1000:]
    
    def _process_orderbook_data(self, data: Dict):
        """处理订单簿数据"""
        if len(self.data_buffer) < 2:
            return
            
        # 计算订单流不平衡
        ofi = self._calculate_ofi(data)
        
        ofi_data = {
            'timestamp': data['timestamp'],
            'symbol': data['symbol'],
            'ofi': ofi,
            'bid_volume': sum([bid[1] for bid in data['bids']]),
            'ask_volume': sum([ask[1] for ask in data['asks']])
        }
        
        self.ofi_data.append(ofi_data)
        
        # 保持最近1000条数据
        if len(self.ofi_data) > 1000:
            self.ofi_data = self.ofi_data[-1000:]
    
    def _process_trade_data(self, data: Dict):
        """处理交易数据"""
        # 这里可以添加交易数据分析逻辑
        pass
    
    def _calculate_ofi(self, orderbook_data: Dict) -> float:
        """计算订单流不平衡"""
        try:
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']
            
            if not bids or not asks:
                return 0.0
            
            # 计算买卖压力
            bid_pressure = sum([bid[1] for bid in bids])
            ask_pressure = sum([ask[1] for ask in asks])
            
            # 计算OFI
            if bid_pressure + ask_pressure > 0:
                ofi = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure)
                return ofi
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"OFI计算错误: {e}")
            return 0.0
    
    def _generate_signals_loop(self):
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
                                'price': self.data_buffer[-1]['price'] if self.data_buffer else 0
                            }
                            
                            self.signals.append(signal)
                            logger.info(f"生成信号: {signal['side']} 强度: {signal['strength']:.2f}")
                
                time.sleep(1)  # 1秒间隔
                
            except Exception as e:
                logger.error(f"信号生成错误: {e}")
                time.sleep(1)
    
    def _execute_trades_loop(self):
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
        # 简单的交易逻辑
        # 可以添加更复杂的风险管理逻辑
        
        # 检查是否有未成交订单
        open_orders = self.bot.get_open_orders()
        if open_orders:
            return False
        
        # 检查信号强度
        if signal['strength'] < self.signal_threshold:
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
                return result
            else:
                logger.error(f"交易失败: {result}")
                return None
                
        except Exception as e:
            logger.error(f"交易执行错误: {e}")
            return None
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if not self.trades:
            return {
                'total_trades': 0,
                'successful_trades': 0,
                'success_rate': 0.0,
                'total_signals': len(self.signals)
            }
        
        successful_trades = sum(1 for trade in self.trades if trade['result'])
        
        return {
            'total_trades': len(self.trades),
            'successful_trades': successful_trades,
            'success_rate': successful_trades / len(self.trades) if self.trades else 0.0,
            'total_signals': len(self.signals),
            'signal_to_trade_ratio': len(self.trades) / len(self.signals) if self.signals else 0.0
        }
    
    def get_latest_data(self) -> Dict:
        """获取最新数据"""
        return {
            'market_data': self.data_buffer[-10:] if self.data_buffer else [],
            'ofi_data': self.ofi_data[-10:] if self.ofi_data else [],
            'signals': self.signals[-10:] if self.signals else [],
            'trades': self.trades[-10:] if self.trades else []
        }
    
    def save_data_to_csv(self, filename: str = None):
        """保存数据到CSV"""
        if not filename:
            filename = f"binance_v10_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # 合并所有数据
        all_data = []
        
        for i, market_data in enumerate(self.data_buffer):
            ofi_data = self.ofi_data[i] if i < len(self.ofi_data) else {}
            
            row = {
                'timestamp': market_data['timestamp'],
                'price': market_data['price'],
                'bid': market_data['bid'],
                'ask': market_data['ask'],
                'volume': market_data['volume'],
                'ofi': ofi_data.get('ofi', 0),
                'bid_volume': ofi_data.get('bid_volume', 0),
                'ask_volume': ofi_data.get('ask_volume', 0)
            }
            all_data.append(row)
        
        # 保存到CSV
        df = pd.DataFrame(all_data)
        df.to_csv(filename, index=False)
        logger.info(f"数据已保存到: {filename}")

def main():
    """主函数"""
    # 币安测试网API配置
    API_KEY = "tKs2nQW3t3rcHnUCGOPWLIYsDotF6ot1GRZzo5n3QOsNRqbPTxO06TC8QglxtUYy"
    SECRET_KEY = "I5SqluL8yqvkcSSb0hBBMr80cHmNndxrl6DAufw9t6myjlRBQDvPxUmabHNJpkZJ"
    
    # 创建币安V10集成系统
    integration = BinanceV10Integration(API_KEY, SECRET_KEY)
    
    try:
        # 启动系统
        integration.start()
        
        # 运行一段时间
        logger.info("系统运行中...")
        time.sleep(60)  # 运行60秒
        
        # 获取性能统计
        stats = integration.get_performance_stats()
        logger.info(f"性能统计: {stats}")
        
        # 保存数据
        integration.save_data_to_csv()
        
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        # 停止系统
        integration.stop()

if __name__ == "__main__":
    main()
