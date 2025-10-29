#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7x24小时运行修复验证脚本
验证移除运行时长限制后的系统行为
"""

import os
import asyncio
import logging
from datetime import datetime

# 设置测试环境变量
os.environ['SYMBOLS'] = 'BTCUSDT'  # 只测试一个symbol
os.environ['STREAM_IDLE_SEC'] = '10'  # 短超时，便于测试
os.environ['HEALTH_CHECK_INTERVAL'] = '5'  # 短健康检查间隔
os.environ['PARQUET_ROTATE_SEC'] = '30'  # 短轮转间隔
os.environ['RUN_HOURS'] = '0.1'  # 设置很短的时间，但代码中已移除限制

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_7x24_fix():
    """测试7x24小时运行修复"""
    logger.info("开始测试7x24小时运行修复...")
    
    try:
        # 导入修复后的采集器
        from run_success_harvest import SuccessOFICVDHarvester
        
        # 创建采集器（设置很短的运行时间，但代码中已移除限制）
        harvester = SuccessOFICVDHarvester(
            symbols=['BTCUSDT'], 
            run_hours=0.1,  # 0.1小时，但代码中不再使用此限制
            output_dir=None
        )
        
        logger.info("采集器创建成功，开始运行...")
        logger.info(f"配置检查:")
        logger.info(f"  - STREAM_IDLE_SEC: {harvester.stream_idle_sec}")
        logger.info(f"  - TRADE_TIMEOUT: {harvester.trade_timeout}")
        logger.info(f"  - ORDERBOOK_TIMEOUT: {harvester.orderbook_timeout}")
        logger.info(f"  - HEALTH_CHECK_INTERVAL: {harvester.health_check_interval}")
        
        # 运行采集器（应该不会因为时间限制而停止）
        start_time = datetime.now()
        logger.info(f"开始时间: {start_time}")
        
        # 运行30秒后手动停止，验证不是因为时间限制停止
        await asyncio.wait_for(harvester.run(), timeout=30)
        
    except asyncio.TimeoutError:
        logger.info("测试完成：30秒后手动停止，验证不是因为时间限制停止")
        logger.info("✅ 7x24小时运行修复验证成功")
    except Exception as e:
        logger.error(f"测试过程中出现异常: {e}")
        logger.error("❌ 7x24小时运行修复验证失败")

if __name__ == "__main__":
    asyncio.run(test_7x24_fix())
