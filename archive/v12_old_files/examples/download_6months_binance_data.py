#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安6个月数据下载运行脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.binance_6months_downloader import Binance6MonthsDownloader
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("币安6个月数据下载系统")
    logger.info("=" * 80)
    
    # 创建下载器
    downloader = Binance6MonthsDownloader()
    
    # 下载ETHUSDT 6个月数据
    logger.info("🚀 开始下载ETHUSDT 6个月数据...")
    success = downloader.download_6months_data("ETHUSDT", "1m")
    
    if success:
        logger.info("✅ ETHUSDT数据下载成功！")
        
        # 可选：下载其他主要交易对
        logger.info("📊 是否下载其他交易对数据？")
        logger.info("建议下载: BTCUSDT, BNBUSDT")
        
        # 这里可以添加其他交易对的下载逻辑
        # downloader.download_multiple_symbols(["BTCUSDT", "BNBUSDT"], "1m")
        
    else:
        logger.error("❌ ETHUSDT数据下载失败")
    
    logger.info("=" * 80)
    logger.info("数据下载任务完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()