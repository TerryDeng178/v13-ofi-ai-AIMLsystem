#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安数据下载脚本
基于优化后的下载器下载完整数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
import time

from src.binance_data_downloader_optimized import OptimizedBinanceDataDownloader

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_binance_data():
    """下载币安数据"""
    logger.info("=" * 80)
    logger.info("币安数据下载")
    logger.info("=" * 80)
    
    # 创建下载器
    downloader = OptimizedBinanceDataDownloader()
    
    # 测试连接
    logger.info("测试连接...")
    if not downloader.test_connection():
        logger.error("连接测试失败，请检查网络")
        return False
    
    # 下载完整数据
    logger.info("开始下载ETH/USDT永续合约6个月数据...")
    start_time = time.time()
    
    full_data = downloader.download_full_data("ETHUSDT", "1m", 6)
    
    end_time = time.time()
    duration = end_time - start_time
    
    if full_data.empty:
        logger.error("完整数据下载失败")
        return False
    
    # 验证数据质量
    quality_report = downloader.validate_data_quality(full_data)
    
    logger.info("=" * 80)
    logger.info("下载完成！")
    logger.info("=" * 80)
    logger.info(f"下载耗时: {duration:.2f} 秒")
    logger.info(f"记录数: {quality_report['total_records']:,}")
    logger.info(f"时间范围: {quality_report['time_range']['start']} 到 {quality_report['time_range']['end']}")
    logger.info(f"价格范围: {quality_report['price_range']['min_price']:.2f} - {quality_report['price_range']['max_price']:.2f}")
    logger.info(f"价格波动率: {quality_report['price_range']['price_volatility']:.4f}")
    logger.info(f"平均成交量: {quality_report['volume_stats']['avg_volume']:,.0f}")
    logger.info(f"缺失值: {quality_report['missing_values']}")
    logger.info(f"重复值: {quality_report['duplicates']}")
    
    logger.info("✅ 币安数据下载成功！")
    return True

if __name__ == "__main__":
    download_binance_data()
