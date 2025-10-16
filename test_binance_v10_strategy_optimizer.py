#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安V10策略优化器测试
测试信号过滤、参数调优、回测验证功能
"""

import time
import logging
from binance_v10_strategy_optimizer import BinanceV10StrategyOptimizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_strategy_optimizer():
    """测试策略优化器"""
    # 币安测试网API配置
    API_KEY = "tKs2nQW3t3rcHnUCGOPWLIYsDotF6ot1GRZzo5n3QOsNRqbPTxO06TC8QglxtUYy"
    SECRET_KEY = "I5SqluL8yqvkcSSb0hBBMr80cHmNndxrl6DAufw9t6myjlRBQDvPxUmabHNJpkZJ"
    SYMBOL = "ETHUSDT"
    
    logger.info("开始测试币安V10策略优化器...")
    
    optimizer = None
    try:
        # 创建策略优化器
        optimizer = BinanceV10StrategyOptimizer(API_KEY, SECRET_KEY, SYMBOL)
        
        # 启动优化器
        optimizer.start()
        logger.info("策略优化器启动成功")
        
        # 运行测试
        logger.info("策略优化器运行中，监控优化状态...")
        
        for i in range(60):  # 运行60秒
            # 获取优化状态
            status = optimizer.get_optimization_status()
            
            logger.info(f"=== 优化状态 {i+1}/60 ===")
            logger.info(f"当前价格: {status['trading_status']['current_price']:.2f}")
            logger.info(f"OFI Z-score: {status['trading_status']['ofi_zscore']:.3f}")
            logger.info(f"信号历史数量: {status['signal_history_count']}")
            logger.info(f"优化运行次数: {status['optimization_runs']}")
            
            # 显示当前参数
            params = status['current_params']
            logger.info(f"当前参数:")
            logger.info(f"  OFI阈值: {params['ofi_threshold']:.2f}")
            logger.info(f"  动量阈值: {params['momentum_threshold']:.4f}")
            logger.info(f"  止损百分比: {params['stop_loss_pct']:.2f}")
            logger.info(f"  止盈百分比: {params['take_profit_pct']:.2f}")
            
            # 显示交易状态
            trading_status = status['trading_status']
            if trading_status['position']:
                pos = trading_status['position']
                logger.info(f"当前仓位: {pos['side']} {pos['size']:.4f}")
                logger.info(f"未实现盈亏: {pos['unrealized_pnl']:.2f}")
            else:
                logger.info("当前无仓位")
            
            logger.info(f"订单数量: {trading_status['order_count']}")
            logger.info(f"活跃订单: {trading_status['active_orders']}")
            
            time.sleep(1)
        
        # 最终状态
        final_status = optimizer.get_optimization_status()
        logger.info("=== 最终优化状态 ===")
        logger.info(f"信号历史数量: {final_status['signal_history_count']}")
        logger.info(f"优化运行次数: {final_status['optimization_runs']}")
        
        # 导出优化结果
        logger.info("导出优化结果...")
        optimizer.export_optimization_results()
        
    except KeyboardInterrupt:
        logger.info("用户中断测试")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
    finally:
        # 停止优化器
        if optimizer:
            optimizer.stop()
        logger.info("策略优化器测试完成")

if __name__ == "__main__":
    test_strategy_optimizer()
