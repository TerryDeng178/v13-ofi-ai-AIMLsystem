#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安官方API下载器
严格按照币安官方API文档规范
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

class BinanceOfficialDownloader:
    """币安官方API下载器"""
    
    def __init__(self):
        # 币安期货API官方端点
        self.base_url = "https://fapi.binance.com"
        self.session = requests.Session()
        
        # 官方推荐的请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
        })
        
        # 数据存储路径
        self.data_dir = "data/binance"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 币安官方API限制（2024年最新）
        self.api_limits = {
            'requests_per_minute': 6000,     # 每分钟6000次请求
            'requests_per_second': 100,      # 每秒100次请求
            'weight_per_minute': 6000,       # 每分钟6000权重
            'kline_weight': 1,               # K线数据权重为1
        }
        
        # 保守的下载配置
        self.config = {
            'request_delay': 0.1,            # 100ms延迟，确保不超过10req/s
            'batch_delay': 10,               # 批次间10秒延迟
            'batch_size': 1000,              # 每批1000条记录
            'max_retries': 5,                # 最大重试次数
            'timeout': 30,                   # 请求超时
            'backoff_factor': 2,             # 退避因子
        }
        
        logger.info("币安官方下载器初始化完成")
        logger.info(f"API限制: {self.api_limits['requests_per_second']} req/s, {self.api_limits['requests_per_minute']} req/min")
        logger.info(f"配置: 请求延迟{self.config['request_delay']}s, 批次延迟{self.config['batch_delay']}s")
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            logger.info("测试币安官方API连接...")
            
            # 测试ping端点
            ping_url = f"{self.base_url}/fapi/v1/ping"
            response = self.session.get(ping_url, timeout=self.config['timeout'])
            
            if response.status_code == 200:
                logger.info("✅ 币安API连接成功")
                return True
            else:
                logger.error(f"❌ 币安API连接失败: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 连接测试异常: {e}")
            return False
    
    def get_server_time(self) -> int:
        """获取服务器时间"""
        try:
            url = f"{self.base_url}/fapi/v1/time"
            response = self.session.get(url, timeout=self.config['timeout'])
            response.raise_for_status()
            data = response.json()
            return data['serverTime']
        except Exception as e:
            logger.error(f"获取服务器时间失败: {e}")
            return int(time.time() * 1000)
    
    def download_historical_data(self, symbol: str = "ETHUSDT", interval: str = "1m", 
                                days: int = 180) -> pd.DataFrame:
        """
        下载历史数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            days: 下载天数
        
        Returns:
            数据DataFrame
        """
        logger.info(f"开始下载 {symbol} {interval} 历史数据 ({days}天)")
        
        # 计算时间范围
        end_time = self.get_server_time()
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        logger.info(f"时间范围: {datetime.fromtimestamp(start_time/1000)} 到 {datetime.fromtimestamp(end_time/1000)}")
        
        # 分批下载
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            # 计算当前批次的结束时间
            current_end = min(current_start + (self.config['batch_size'] * 60 * 1000), end_time)
            
            logger.info(f"下载批次: {datetime.fromtimestamp(current_start/1000)} 到 {datetime.fromtimestamp(current_end/1000)}")
            
            # 获取数据
            klines = self._get_klines_with_retry(symbol, interval, current_start, current_end)
            
            if not klines:
                logger.warning("未获取到数据，等待重试...")
                time.sleep(5)
                continue
            
            all_data.extend(klines)
            
            # 更新起始时间
            if klines:
                last_timestamp = int(klines[-1][0])
                current_start = last_timestamp + 1
            else:
                break
            
            # 显示进度
            progress = (current_start - start_time) / (end_time - start_time) * 100
            logger.info(f"下载进度: {progress:.1f}%, 已获取: {len(all_data)} 条记录")
            
            # 批次间延迟
            time.sleep(self.config['batch_delay'])
        
        # 转换为DataFrame
        df = self._convert_to_dataframe(all_data)
        
        logger.info(f"数据下载完成: {len(df)} 条记录")
        return df
    
    def _get_klines_with_retry(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[List]:
        """带重试机制的K线数据获取"""
        url = f"{self.base_url}/fapi/v1/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': self.config['batch_size']
        }
        
        for attempt in range(self.config['max_retries']):
            try:
                # 请求延迟
                time.sleep(self.config['request_delay'])
                
                response = self.session.get(url, params=params, timeout=self.config['timeout'])
                
                # 检查响应状态
                if response.status_code == 429:  # 请求过于频繁
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"触发API限制，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code == 418:  # I'm a teapot
                    wait_time = 120 * (attempt + 1)
                    logger.warning(f"触发418错误，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                return data
                
            except requests.exceptions.RequestException as e:
                logger.error(f"请求失败 (尝试 {attempt + 1}/{self.config['max_retries']}): {e}")
                if attempt < self.config['max_retries'] - 1:
                    wait_time = self.config['backoff_factor'] ** attempt
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                continue
            except Exception as e:
                logger.error(f"处理失败: {e}")
                break
        
        return []
    
    def _convert_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """将K线数据转换为DataFrame"""
        if not klines:
            return pd.DataFrame()
        
        # 定义列名（按照币安官方文档）
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


def download_binance_official():
    """使用官方API下载数据"""
    logger.info("=" * 80)
    logger.info("币安官方API数据下载")
    logger.info("=" * 80)
    
    # 创建下载器
    downloader = BinanceOfficialDownloader()
    
    # 测试连接
    if not downloader.test_connection():
        logger.error("连接测试失败，请检查网络")
        return False
    
    # 先下载少量数据测试
    logger.info("步骤1: 下载3天测试数据")
    test_data = downloader.download_historical_data("ETHUSDT", "1m", 3)
    
    if test_data.empty:
        logger.error("测试数据下载失败")
        return False
    
    # 验证测试数据
    quality_report = downloader.validate_data_quality(test_data)
    
    logger.info("测试数据质量:")
    logger.info(f"  记录数: {quality_report['total_records']:,}")
    logger.info(f"  时间范围: {quality_report['time_range']['start']} 到 {quality_report['time_range']['end']}")
    logger.info(f"  价格范围: {quality_report['price_range']['min_price']:.2f} - {quality_report['price_range']['max_price']:.2f}")
    
    logger.info("✅ 测试数据下载成功！")
    
    # 保存测试数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_file = f"data/binance/ETHUSDT_1m_3days_test_{timestamp}.csv"
    test_data.to_csv(test_file, index=False)
    logger.info(f"测试数据已保存到: {test_file}")
    
    return True


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("币安官方API下载器")
    logger.info("=" * 80)
    
    download_binance_official()


if __name__ == "__main__":
    main()
