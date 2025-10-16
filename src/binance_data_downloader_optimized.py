#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的币安数据下载器
解决下载速度慢的问题，先进行小量测试
"""

import pandas as pd
import numpy as np
import requests
import time
import logging
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedBinanceDataDownloader:
    """优化的币安数据下载器"""
    
    def __init__(self, base_url: str = "https://fapi.binance.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        })
        
        # 数据存储路径
        self.data_dir = "data/binance"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 下载配置
        self.request_delay = 0.05  # 50ms延迟，避免请求过快
        self.max_retries = 3
        self.timeout = 30
        
        logger.info("优化币安数据下载器初始化完成")
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            url = f"{self.base_url}/fapi/v1/ping"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            logger.info("✅ 币安连接测试成功")
            return True
        except Exception as e:
            logger.error(f"❌ 币安连接测试失败: {e}")
            return False
    
    def get_server_time(self) -> int:
        """获取服务器时间"""
        try:
            url = f"{self.base_url}/fapi/v1/time"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data['serverTime']
        except Exception as e:
            logger.error(f"获取服务器时间失败: {e}")
            return int(time.time() * 1000)
    
    def download_small_test(self, symbol: str = "ETHUSDT", interval: str = "1m", hours: int = 24) -> pd.DataFrame:
        """
        下载小量测试数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            hours: 下载小时数
        
        Returns:
            测试数据DataFrame
        """
        logger.info(f"开始下载 {symbol} {interval} 测试数据 ({hours}小时)")
        
        # 计算时间范围
        end_time = self.get_server_time()
        start_time = end_time - (hours * 60 * 60 * 1000)  # 转换为毫秒
        
        logger.info(f"时间范围: {datetime.fromtimestamp(start_time/1000)} 到 {datetime.fromtimestamp(end_time/1000)}")
        
        # 下载数据
        klines = self._download_klines_batch(symbol, interval, start_time, end_time)
        
        if not klines:
            logger.error("测试数据下载失败")
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = self._convert_to_dataframe(klines)
        
        logger.info(f"测试数据下载完成: {len(df)} 条记录")
        logger.info(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
        logger.info(f"平均成交量: {df['volume'].mean():,.0f}")
        
        return df
    
    def _download_klines_batch(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[List]:
        """批量下载K线数据"""
        url = f"{self.base_url}/fapi/v1/klines"
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_time,
                'limit': 1500
            }
            
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                
                if not data:
                    logger.warning("未获取到数据，停止下载")
                    break
                
                all_data.extend(data)
                
                # 更新起始时间
                last_timestamp = int(data[-1][0])
                current_start = last_timestamp + 1
                
                logger.info(f"已下载: {len(all_data)} 条记录, 当前时间: {datetime.fromtimestamp(current_start/1000)}")
                
                # 控制请求频率
                time.sleep(self.request_delay)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"请求失败: {e}")
                time.sleep(1)
                continue
            except Exception as e:
                logger.error(f"处理失败: {e}")
                break
        
        return all_data
    
    def _convert_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """将K线数据转换为DataFrame"""
        if not klines:
            return pd.DataFrame()
        
        # 定义列名
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
            'taker_buy_quote_volume', 'ignore'
        ]
        
        # 创建DataFrame
        df = pd.DataFrame(klines, columns=columns)
        
        # 数据类型转换
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # 价格和成交量转换为数值类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'taker_buy_base_volume', 'taker_buy_quote_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 重命名列以匹配V11系统
        df = df.rename(columns={
            'open_time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        # 选择需要的列
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # 移除缺失值
        df = df.dropna()
        
        # 按时间排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def download_full_data(self, symbol: str = "ETHUSDT", interval: str = "1m", months: int = 6) -> pd.DataFrame:
        """
        下载完整历史数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            months: 下载月数
        
        Returns:
            完整数据DataFrame
        """
        logger.info(f"开始下载 {symbol} {interval} 完整数据 ({months}个月)")
        
        # 计算时间范围
        end_time = self.get_server_time()
        start_time = end_time - (months * 30 * 24 * 60 * 60 * 1000)  # 转换为毫秒
        
        logger.info(f"时间范围: {datetime.fromtimestamp(start_time/1000)} 到 {datetime.fromtimestamp(end_time/1000)}")
        
        # 下载数据
        klines = self._download_klines_batch(symbol, interval, start_time, end_time)
        
        if not klines:
            logger.error("完整数据下载失败")
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = self._convert_to_dataframe(klines)
        
        # 保存数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.data_dir}/{symbol}_{interval}_{months}months_{timestamp}.csv"
        
        df.to_csv(output_file, index=False)
        
        logger.info(f"完整数据下载完成: {len(df)} 条记录")
        logger.info(f"数据已保存到: {output_file}")
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """验证数据质量"""
        if df.empty:
            return {"error": "数据为空"}
        
        report = {
            "total_records": len(df),
            "time_range": {
                "start": df['timestamp'].min(),
                "end": df['timestamp'].max()
            },
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "price_range": {
                "min_price": df['close'].min(),
                "max_price": df['close'].max(),
                "price_volatility": df['close'].pct_change().std()
            },
            "volume_stats": {
                "avg_volume": df['volume'].mean(),
                "max_volume": df['volume'].max(),
                "volume_volatility": df['volume'].std()
            }
        }
        
        return report


def test_small_download():
    """测试小量下载"""
    logger.info("=" * 80)
    logger.info("币安数据下载器小量测试")
    logger.info("=" * 80)
    
    # 创建下载器
    downloader = OptimizedBinanceDataDownloader()
    
    # 测试连接
    if not downloader.test_connection():
        logger.error("连接测试失败，请检查网络")
        return False
    
    # 下载24小时测试数据
    test_data = downloader.download_small_test("ETHUSDT", "1m", 24)
    
    if test_data.empty:
        logger.error("测试数据下载失败")
        return False
    
    # 验证数据质量
    quality_report = downloader.validate_data_quality(test_data)
    
    logger.info("测试数据质量报告:")
    logger.info(f"  记录数: {quality_report['total_records']:,}")
    logger.info(f"  时间范围: {quality_report['time_range']['start']} 到 {quality_report['time_range']['end']}")
    logger.info(f"  价格范围: {quality_report['price_range']['min_price']:.2f} - {quality_report['price_range']['max_price']:.2f}")
    logger.info(f"  价格波动率: {quality_report['price_range']['price_volatility']:.4f}")
    logger.info(f"  平均成交量: {quality_report['volume_stats']['avg_volume']:,.0f}")
    logger.info(f"  缺失值: {quality_report['missing_values']}")
    logger.info(f"  重复值: {quality_report['duplicates']}")
    
    logger.info("✅ 小量测试成功！")
    return True


def download_full_data():
    """下载完整数据"""
    logger.info("=" * 80)
    logger.info("币安完整数据下载")
    logger.info("=" * 80)
    
    # 创建下载器
    downloader = OptimizedBinanceDataDownloader()
    
    # 下载完整数据
    full_data = downloader.download_full_data("ETHUSDT", "1m", 6)
    
    if full_data.empty:
        logger.error("完整数据下载失败")
        return False
    
    # 验证数据质量
    quality_report = downloader.validate_data_quality(full_data)
    
    logger.info("完整数据质量报告:")
    logger.info(f"  记录数: {quality_report['total_records']:,}")
    logger.info(f"  时间范围: {quality_report['time_range']['start']} 到 {quality_report['time_range']['end']}")
    logger.info(f"  价格范围: {quality_report['price_range']['min_price']:.2f} - {quality_report['price_range']['max_price']:.2f}")
    logger.info(f"  价格波动率: {quality_report['price_range']['price_volatility']:.4f}")
    logger.info(f"  平均成交量: {quality_report['volume_stats']['avg_volume']:,.0f}")
    
    logger.info("✅ 完整数据下载成功！")
    return True


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("优化币安数据下载器")
    logger.info("=" * 80)
    
    # 先进行小量测试
    logger.info("步骤1: 小量测试")
    if test_small_download():
        logger.info("小量测试成功，继续下载完整数据...")
        
        # 用户确认
        response = input("小量测试成功！是否继续下载6个月完整数据？(y/n): ")
        if response.lower() == 'y':
            logger.info("步骤2: 下载完整数据")
            download_full_data()
        else:
            logger.info("用户取消下载")
    else:
        logger.error("小量测试失败，请检查网络连接")


if __name__ == "__main__":
    main()
