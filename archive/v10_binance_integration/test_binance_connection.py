#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安连接测试脚本
测试币安测试网API连接和基本功能
"""

import time
import logging
from binance_integration import BinanceTradingBot

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_binance_connection():
    """测试币安连接"""
    # 币安测试网API配置
    API_KEY = "tKs2nQW3t3rcHnUCGOPWLIYsDotF6ot1GRZzo5n3QOsNRqbPTxO06TC8QglxtUYy"
    SECRET_KEY = "I5SqluL8yqvkcSSb0hBBMr80cHmNndxrl6DAufw9t6myjlRBQDvPxUmabHNJpkZJ"
    
    logger.info("开始测试币安连接...")
    
    try:
        # 创建交易机器人
        bot = BinanceTradingBot(API_KEY, SECRET_KEY)
        
        # 测试1: 获取账户信息
        logger.info("测试1: 获取账户信息...")
        try:
            account_info = bot.api.get_account_info()
            logger.info(f"账户信息获取成功: {account_info}")
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
        
        # 测试2: 获取持仓信息
        logger.info("测试2: 获取持仓信息...")
        try:
            positions = bot.api.get_position_info("ETHUSDT")
            logger.info(f"持仓信息: {positions}")
        except Exception as e:
            logger.error(f"获取持仓信息失败: {e}")
        
        # 测试3: 获取订单簿
        logger.info("测试3: 获取订单簿...")
        try:
            orderbook = bot.api.get_orderbook("ETHUSDT", 10)
            logger.info(f"订单簿: {orderbook}")
        except Exception as e:
            logger.error(f"获取订单簿失败: {e}")
        
        # 测试4: 获取K线数据
        logger.info("测试4: 获取K线数据...")
        try:
            klines = bot.api.get_klines("ETHUSDT", "1m", 10)
            logger.info(f"K线数据: {len(klines)}条")
            if klines:
                latest = klines[-1]
                logger.info(f"最新K线: 开盘={latest[1]}, 收盘={latest[4]}, 最高={latest[2]}, 最低={latest[3]}")
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
        
        # 测试5: 启动WebSocket连接
        logger.info("测试5: 启动WebSocket连接...")
        try:
            bot.start()
            logger.info("WebSocket连接启动成功")
            
            # 等待数据
            time.sleep(10)
            
            # 获取数据
            market_data = bot.get_market_data()
            orderbook_data = bot.get_orderbook_data()
            trade_data = bot.get_trade_data()
            
            logger.info(f"市场数据: {len(market_data)}条")
            logger.info(f"订单簿数据: {len(orderbook_data)}条")
            logger.info(f"交易数据: {len(trade_data)}条")
            
            # 显示最新数据
            if market_data:
                latest = market_data[-1]
                logger.info(f"最新价格: {latest['price']}")
                logger.info(f"买卖价差: {latest['ask'] - latest['bid']}")
            
        except Exception as e:
            logger.error(f"WebSocket连接失败: {e}")
        finally:
            # 停止机器人
            bot.stop()
        
        logger.info("币安连接测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")

if __name__ == "__main__":
    test_binance_connection()
