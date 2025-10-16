#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安V10实时交易测试
测试币安V10实时交易系统
"""

import time
import logging
from binance_v10_realtime_trading import BinanceV10RealtimeTrading

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_binance_v10_realtime():
    """测试币安V10实时交易系统"""
    # 币安测试网API配置
    API_KEY = "tKs2nQW3t3rcHnUCGOPWLIYsDotF6ot1GRZzo5n3QOsNRqbPTxO06TC8QglxtUYy"
    SECRET_KEY = "I5SqluL8yqvkcSSb0hBBMr80cHmNndxrl6DAufw9t6myjlRBQDvPxUmabHNJpkZJ"
    
    logger.info("开始测试币安V10实时交易系统...")
    
    try:
        # 创建实时交易系统
        trading_system = BinanceV10RealtimeTrading(API_KEY, SECRET_KEY)
        
        # 启动系统
        trading_system.start()
        logger.info("实时交易系统启动成功")
        
        # 运行测试
        logger.info("系统运行中...")
        time.sleep(60)  # 运行60秒
        
        # 显示状态
        trading_system._show_status()
        
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
    finally:
        # 停止系统
        trading_system.stop()
        logger.info("测试完成")

if __name__ == "__main__":
    test_binance_v10_realtime()
