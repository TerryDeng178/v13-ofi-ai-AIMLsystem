#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安V10高级交易系统测试
测试完整的订单管理、风险控制、仓位管理功能
"""

import time
import logging
from binance_v10_advanced_trading import BinanceV10AdvancedTrading

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_advanced_trading_system():
    """测试高级交易系统"""
    # 币安测试网API配置
    API_KEY = "tKs2nQW3t3rcHnUCGOPWLIYsDotF6ot1GRZzo5n3QOsNRqbPTxO06TC8QglxtUYy"
    SECRET_KEY = "I5SqluL8yqvkcSSb0hBBMr80cHmNndxrl6DAufw9t6myjlRBQDvPxUmabHNJpkZJ"
    SYMBOL = "ETHUSDT"
    
    logger.info("开始测试币安V10高级交易系统...")
    
    trading_system = None
    try:
        # 创建高级交易系统
        trading_system = BinanceV10AdvancedTrading(API_KEY, SECRET_KEY, SYMBOL)
        
        # 启动系统
        trading_system.start()
        logger.info("高级交易系统启动成功")
        
        # 运行测试
        logger.info("系统运行中，监控交易状态...")
        
        for i in range(30):  # 运行30秒
            # 获取交易状态
            status = trading_system.get_trading_status()
            
            logger.info(f"=== 交易状态 {i+1}/30 ===")
            logger.info(f"当前价格: {status['current_price']:.2f}")
            logger.info(f"OFI Z-score: {status['ofi_zscore']:.3f}")
            
            if status['position']:
                pos = status['position']
                logger.info(f"仓位: {pos['side']} {pos['size']:.4f}")
                logger.info(f"入场价格: {pos['entry_price']:.2f}")
                logger.info(f"未实现盈亏: {pos['unrealized_pnl']:.2f}")
            else:
                logger.info("当前无仓位")
            
            logger.info(f"日盈亏: {status['daily_pnl']:.2f}")
            logger.info(f"订单数量: {status['order_count']}")
            logger.info(f"活跃订单: {status['active_orders']}")
            
            time.sleep(1)
        
        # 最终状态
        final_status = trading_system.get_trading_status()
        logger.info("=== 最终交易状态 ===")
        logger.info(f"总订单数: {final_status['order_count']}")
        logger.info(f"活跃订单: {final_status['active_orders']}")
        logger.info(f"日盈亏: {final_status['daily_pnl']:.2f}")
        
        if final_status['position']:
            pos = final_status['position']
            logger.info(f"最终仓位: {pos['side']} {pos['size']:.4f}")
            logger.info(f"未实现盈亏: {pos['unrealized_pnl']:.2f}")
        
    except KeyboardInterrupt:
        logger.info("用户中断测试")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
    finally:
        # 停止系统
        if trading_system:
            trading_system.stop()
        logger.info("高级交易系统测试完成")

if __name__ == "__main__":
    test_advanced_trading_system()
