#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试补丁A和补丁B的验证脚本
验证交易流和订单簿流的超时重连机制
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# 添加项目路径
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

# 设置测试环境变量
os.environ['STREAM_IDLE_SEC'] = '10'  # 10秒超时，便于测试
os.environ['TRADE_TIMEOUT'] = '15'    # 15秒交易流超时（10+5s缓冲）
os.environ['ORDERBOOK_TIMEOUT'] = '20'  # 20秒订单簿流超时（10+10s缓冲）
os.environ['HEALTH_CHECK_INTERVAL'] = '5'  # 5秒健康检查
os.environ['EXTREME_TRAFFIC_THRESHOLD'] = '1000'  # 测试用低阈值
os.environ['EXTREME_ROTATE_SEC'] = '15'  # 极端流量轮转间隔
os.environ['SAVE_CONCURRENCY'] = '3'  # 测试并发保存
os.environ['RUN_HOURS'] = '0.1'  # 6分钟测试
os.environ['SYMBOLS'] = 'BTCUSDT'  # 只测试一个symbol

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_stream_patches():
    """测试补丁功能"""
    try:
        # 导入修复后的采集器
        from run_success_harvest import SuccessOFICVDHarvester
        
        logger.info("开始测试补丁A和补丁B...")
        logger.info(f"测试配置:")
        logger.info(f"  STREAM_IDLE_SEC: {os.getenv('STREAM_IDLE_SEC')}")
        logger.info(f"  TRADE_TIMEOUT: {os.getenv('TRADE_TIMEOUT')}")
        logger.info(f"  ORDERBOOK_TIMEOUT: {os.getenv('ORDERBOOK_TIMEOUT')}")
        logger.info(f"  HEALTH_CHECK_INTERVAL: {os.getenv('HEALTH_CHECK_INTERVAL')}")
        logger.info(f"  EXTREME_TRAFFIC_THRESHOLD: {os.getenv('EXTREME_TRAFFIC_THRESHOLD')}")
        logger.info(f"  EXTREME_ROTATE_SEC: {os.getenv('EXTREME_ROTATE_SEC')}")
        logger.info(f"  SAVE_CONCURRENCY: {os.getenv('SAVE_CONCURRENCY')}")
        
        # 创建测试采集器
        harvester = SuccessOFICVDHarvester(
            symbols=['BTCUSDT'],
            run_hours=0.1,  # 6分钟测试
            output_dir=Path(__file__).parent / "test_data"
        )
        
        # 运行采集器
        await harvester.run()
        
        logger.info("测试完成！")
        
        # 输出最终统计
        logger.info("最终统计:")
        for symbol in harvester.symbols:
            logger.info(f"{symbol}: 交易{harvester.stats['total_trades'][symbol]}, "
                      f"OFI{harvester.stats['total_ofi'][symbol]}, "
                      f"CVD{harvester.stats['total_cvd'][symbol]}, "
                      f"事件{harvester.stats['total_events'][symbol]}, "
                      f"订单簿{harvester.stats['total_orderbook'][symbol]}")
        
        logger.info(f"重连次数: {harvester.reconnect_count}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_stream_patches())
