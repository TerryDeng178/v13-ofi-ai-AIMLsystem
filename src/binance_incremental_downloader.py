#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安增量分次下载器
每次只下载少量数据，避免API限制
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

class BinanceIncrementalDownloader:
    """币安增量分次下载器"""
    
    def __init__(self):
        # 币安期货API端点
        self.base_url = "https://fapi.binance.com"
        self.session = requests.Session()
        
        # 请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        
        # 数据存储路径
        self.data_dir = "data/binance"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 增量下载配置 - 非常保守
        self.config = {
            'batch_size': 500,              # 每批只下载500条记录
            'request_delay': 1.0,           # 每次请求延迟1秒
            'batch_delay': 5,               # 批次间延迟5秒
            'max_retries': 3,               # 最大重试次数
            'timeout': 30,                  # 请求超时
        }
        
        logger.info("币安增量下载器初始化完成")
        logger.info(f"配置: 每批{self.config['batch_size']}条, 请求延迟{self.config['request_delay']}s")
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            logger.info("测试币安连接...")
            url = f"{self.base_url}/fapi/v1/ping"
            response = self.session.get(url, timeout=self.config['timeout'])
            
            if response.status_code == 200:
                logger.info("✅ 币安连接成功")
                return True
            else:
                logger.error(f"❌ 币安连接失败: {response.status_code}")
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
    
    def download_small_batch(self, symbol: str = "ETHUSDT", interval: str = "1m", 
                           hours: int = 24) -> pd.DataFrame:
        """
        下载小批次数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            hours: 下载小时数
        
        Returns:
            数据DataFrame
        """
        logger.info(f"开始下载 {symbol} {interval} 数据 ({hours}小时)")
        
        # 计算时间范围
        end_time = self.get_server_time()
        start_time = end_time - (hours * 60 * 60 * 1000)
        
        logger.info(f"时间范围: {datetime.fromtimestamp(start_time/1000)} 到 {datetime.fromtimestamp(end_time/1000)}")
        
        # 分小批次下载
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            # 计算当前批次的结束时间
            current_end = min(current_start + (self.config['batch_size'] * 60 * 1000), end_time)
            
            logger.info(f"下载小批次: {datetime.fromtimestamp(current_start/1000)} 到 {datetime.fromtimestamp(current_end/1000)}")
            
            # 获取数据
            klines = self._get_klines_safe(symbol, interval, current_start, current_end)
            
            if not klines:
                logger.warning("未获取到数据，等待重试...")
                time.sleep(2)
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
    
    def _get_klines_safe(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[List]:
        """安全获取K线数据"""
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
                    wait_time = 60
                    logger.warning(f"触发API限制，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code == 418:  # I'm a teapot
                    wait_time = 120
                    logger.warning(f"触发418错误，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                return data
                
            except requests.exceptions.RequestException as e:
                logger.error(f"请求失败 (尝试 {attempt + 1}/{self.config['max_retries']}): {e}")
                if attempt < self.config['max_retries'] - 1:
                    time.sleep(5)
                continue
            except Exception as e:
                logger.error(f"处理失败: {e}")
                break
        
        return []
    
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


def download_incremental():
    """增量下载测试"""
    logger.info("=" * 80)
    logger.info("币安增量数据下载测试")
    logger.info("=" * 80)
    
    # 创建下载器
    downloader = BinanceIncrementalDownloader()
    
    # 测试连接
    if not downloader.test_connection():
        logger.error("连接测试失败，请检查网络")
        return False
    
    # 下载24小时数据测试
    logger.info("步骤1: 下载24小时测试数据")
    test_data = downloader.download_small_batch("ETHUSDT", "1m", 24)
    
    if test_data.empty:
        logger.error("测试数据下载失败")
        return False
    
    # 验证数据质量
    quality_report = downloader.validate_data_quality(test_data)
    
    logger.info("=" * 50)
    logger.info("测试数据质量报告")
    logger.info("=" * 50)
    logger.info(f"记录数: {quality_report['total_records']:,}")
    logger.info(f"时间范围: {quality_report['time_range']['start']} 到 {quality_report['time_range']['end']}")
    logger.info(f"价格范围: {quality_report['price_range']['min_price']:.2f} - {quality_report['price_range']['max_price']:.2f}")
    logger.info(f"价格波动率: {quality_report['price_range']['price_volatility']:.4f}")
    logger.info(f"平均成交量: {quality_report['volume_stats']['avg_volume']:,.0f}")
    logger.info(f"缺失值: {quality_report['missing_values']}")
    logger.info(f"重复值: {quality_report['duplicates']}")
    
    # 保存测试数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_file = f"data/binance/ETHUSDT_1m_24h_test_{timestamp}.csv"
    test_data.to_csv(test_file, index=False)
    
    logger.info("=" * 50)
    logger.info(f"✅ 测试数据下载成功！")
    logger.info(f"数据已保存到: {test_file}")
    logger.info("=" * 50)
    
    return True


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("币安增量数据下载器")
    logger.info("=" * 80)
    
    download_incremental()


if __name__ == "__main__":
    main()
