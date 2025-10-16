#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的币安V10集成测试
测试币安数据与V10系统的集成
"""

import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from binance_integration import BinanceTradingBot

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleBinanceV10Test:
    """简化的币安V10测试"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.bot = BinanceTradingBot(api_key, secret_key)
        self.symbol = "ETHUSDT"
        self.data_buffer = []
        self.ofi_data = []
        
    def start_test(self):
        """启动测试"""
        logger.info("启动简化币安V10测试...")
        
        try:
            # 启动币安机器人
            self.bot.start()
            logger.info("币安机器人启动成功")
            
            # 等待数据
            time.sleep(5)
            
            # 获取数据
            market_data = self.bot.get_market_data()
            orderbook_data = self.bot.get_orderbook_data()
            trade_data = self.bot.get_trade_data()
            
            logger.info(f"市场数据: {len(market_data)}条")
            logger.info(f"订单簿数据: {len(orderbook_data)}条")
            logger.info(f"交易数据: {len(trade_data)}条")
            
            # 处理数据
            if orderbook_data:
                self._process_orderbook_data(orderbook_data)
            
            # 计算OFI
            if len(self.ofi_data) > 0:
                self._calculate_ofi_stats()
            
            # 显示最新数据
            if market_data:
                latest = market_data[-1]
                logger.info(f"最新价格: {latest['price']}")
                logger.info(f"买卖价差: {latest['ask'] - latest['bid']}")
            
            # 保存数据
            self._save_data()
            
        except Exception as e:
            logger.error(f"测试过程中发生错误: {e}")
        finally:
            # 停止机器人
            self.bot.stop()
            logger.info("测试完成")
    
    def _process_orderbook_data(self, orderbook_data: list):
        """处理订单簿数据"""
        for data in orderbook_data[-10:]:  # 处理最近10条数据
            try:
                # 计算OFI
                ofi = self._calculate_ofi(data)
                
                ofi_record = {
                    'timestamp': data['timestamp'],
                    'symbol': data['symbol'],
                    'ofi': ofi,
                    'bid_volume': sum([bid[1] for bid in data['bids']]),
                    'ask_volume': sum([ask[1] for ask in data['asks']])
                }
                
                self.ofi_data.append(ofi_record)
                logger.info(f"OFI: {ofi:.6f}, 买单量: {ofi_record['bid_volume']:.2f}, 卖单量: {ofi_record['ask_volume']:.2f}")
                
            except Exception as e:
                logger.error(f"处理订单簿数据错误: {e}")
    
    def _calculate_ofi(self, orderbook_data: dict) -> float:
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
    
    def _calculate_ofi_stats(self):
        """计算OFI统计"""
        if len(self.ofi_data) < 2:
            return
        
        ofi_values = [data['ofi'] for data in self.ofi_data]
        
        logger.info(f"OFI统计:")
        logger.info(f"  数据点: {len(ofi_values)}")
        logger.info(f"  最小值: {min(ofi_values):.6f}")
        logger.info(f"  最大值: {max(ofi_values):.6f}")
        logger.info(f"  平均值: {np.mean(ofi_values):.6f}")
        logger.info(f"  标准差: {np.std(ofi_values):.6f}")
        
        # 计算Z-score
        if np.std(ofi_values) > 0:
            z_scores = [(ofi - np.mean(ofi_values)) / np.std(ofi_values) for ofi in ofi_values]
            logger.info(f"Z-score统计:")
            logger.info(f"  最小值: {min(z_scores):.6f}")
            logger.info(f"  最大值: {max(z_scores):.6f}")
            logger.info(f"  平均值: {np.mean(z_scores):.6f}")
            
            # 统计超过阈值的信号
            threshold = 2.0
            strong_signals = [z for z in z_scores if abs(z) > threshold]
            logger.info(f"强信号数量 (|Z| > {threshold}): {len(strong_signals)}")
    
    def _save_data(self):
        """保存数据"""
        if not self.ofi_data:
            logger.info("没有数据可保存")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.ofi_data)
        
        # 保存到CSV
        filename = f"binance_ofi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"数据已保存到: {filename}")
        
        # 显示数据摘要
        logger.info(f"数据摘要:")
        logger.info(f"  总记录数: {len(df)}")
        logger.info(f"  时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        logger.info(f"  OFI范围: {df['ofi'].min():.6f} 到 {df['ofi'].max():.6f}")

def main():
    """主函数"""
    # 币安测试网API配置
    API_KEY = "tKs2nQW3t3rcHnUCGOPWLIYsDotF6ot1GRZzo5n3QOsNRqbPTxO06TC8QglxtUYy"
    SECRET_KEY = "I5SqluL8yqvkcSSb0hBBMr80cHmNndxrl6DAufw9t6myjlRBQDvPxUmabHNJpkZJ"
    
    # 创建测试实例
    test = SimpleBinanceV10Test(API_KEY, SECRET_KEY)
    
    # 运行测试
    test.start_test()

if __name__ == "__main__":
    main()
