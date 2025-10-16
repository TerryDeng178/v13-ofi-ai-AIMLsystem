#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
符合币安API限制的数据下载器
遵守币安API频率限制，避免触发限制
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

class BinanceLimitedDataDownloader:
    """符合币安API限制的数据下载器"""
    
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
        
        # 币安API限制配置
        self.api_limits = {
            'requests_per_minute': 6000,  # 每分钟6000次请求
            'requests_per_second': 100,   # 每秒100次请求 (保守估计)
            'weight_per_minute': 6000,    # 每分钟6000权重
            'kline_weight': 1,            # K线数据权重为1
        }
        
        # 请求控制
        self.request_delay = 0.012  # 12ms延迟，确保不超过100req/s
        self.batch_size = 1000      # 减少批次大小，降低权重消耗
        self.max_retries = 3
        self.timeout = 30
        
        # 请求计数器
        self.request_count = 0
        self.last_reset_time = time.time()
        
        logger.info("币安限制下载器初始化完成")
        logger.info(f"API限制: {self.api_limits['requests_per_second']} req/s, {self.api_limits['requests_per_minute']} req/min")
    
    def _check_rate_limit(self):
        """检查并控制请求频率"""
        current_time = time.time()
        
        # 每分钟重置计数器
        if current_time - self.last_reset_time >= 60:
            self.request_count = 0
            self.last_reset_time = current_time
        
        # 检查是否超过限制
        if self.request_count >= self.api_limits['requests_per_minute'] * 0.8:  # 80%安全边际
            wait_time = 60 - (current_time - self.last_reset_time)
            if wait_time > 0:
                logger.warning(f"接近API限制，等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
                self.request_count = 0
                self.last_reset_time = time.time()
        
        # 控制请求频率
        time.sleep(self.request_delay)
        self.request_count += 1
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            self._check_rate_limit()
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
            self._check_rate_limit()
            url = f"{self.base_url}/fapi/v1/time"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data['serverTime']
        except Exception as e:
            logger.error(f"获取服务器时间失败: {e}")
            return int(time.time() * 1000)
    
    def download_limited_data(self, symbol: str = "ETHUSDT", interval: str = "1m", 
                             days: int = 7) -> pd.DataFrame:
        """
        下载限制数量的数据（避免API限制）
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            days: 下载天数
        
        Returns:
            数据DataFrame
        """
        logger.info(f"开始下载 {symbol} {interval} 数据 ({days}天)")
        
        # 计算时间范围
        end_time = self.get_server_time()
        start_time = end_time - (days * 24 * 60 * 60 * 1000)  # 转换为毫秒
        
        logger.info(f"时间范围: {datetime.fromtimestamp(start_time/1000)} 到 {datetime.fromtimestamp(end_time/1000)}")
        
        # 分批下载数据
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            # 计算当前批次的结束时间
            current_end = min(current_start + (self.batch_size * 60 * 1000), end_time)
            
            logger.info(f"下载批次: {datetime.fromtimestamp(current_start/1000)} 到 {datetime.fromtimestamp(current_end/1000)}")
            
            # 获取数据
            klines = self._get_klines_safe(symbol, interval, current_start, current_end)
            
            if not klines:
                logger.warning("未获取到数据，等待重试...")
                time.sleep(1)
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
        
        # 转换为DataFrame
        df = self._convert_to_dataframe(all_data)
        
        logger.info(f"数据下载完成: {len(df)} 条记录")
        return df
    
    def _get_klines_safe(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[List]:
        """安全获取K线数据（遵守API限制）"""
        url = f"{self.base_url}/fapi/v1/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': min(self.batch_size, 1500)
        }
        
        for attempt in range(self.max_retries):
            try:
                # 检查速率限制
                self._check_rate_limit()
                
                response = self.session.get(url, params=params, timeout=self.timeout)
                
                # 检查响应状态
                if response.status_code == 429:  # 请求过于频繁
                    logger.warning("触发API限制，等待重试...")
                    time.sleep(60)  # 等待1分钟
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                return data
                
            except requests.exceptions.RequestException as e:
                logger.error(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
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
    
    def download_monthly_data(self, symbol: str = "ETHUSDT", interval: str = "1m", 
                             months: int = 6) -> pd.DataFrame:
        """
        按月分批下载数据，避免API限制
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            months: 下载月数
        
        Returns:
            完整数据DataFrame
        """
        logger.info(f"开始按月下载 {symbol} {interval} 数据 ({months}个月)")
        
        all_data = []
        
        for month in range(months):
            logger.info(f"下载第 {month + 1}/{months} 月数据...")
            
            # 计算当月时间范围
            end_time = self.get_server_time()
            month_start = end_time - ((months - month) * 30 * 24 * 60 * 60 * 1000)
            month_end = end_time - ((months - month - 1) * 30 * 24 * 60 * 60 * 1000)
            
            logger.info(f"当月范围: {datetime.fromtimestamp(month_start/1000)} 到 {datetime.fromtimestamp(month_end/1000)}")
            
            # 下载当月数据
            month_data = self.download_limited_data(symbol, interval, 30)
            
            if not month_data.empty:
                all_data.append(month_data)
                logger.info(f"第 {month + 1} 月数据下载完成: {len(month_data)} 条记录")
            else:
                logger.warning(f"第 {month + 1} 月数据下载失败")
            
            # 月间休息，避免API限制
            if month < months - 1:
                logger.info("月间休息，避免API限制...")
                time.sleep(30)  # 休息30秒
        
        # 合并所有数据
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # 保存数据
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.data_dir}/{symbol}_{interval}_{months}months_limited_{timestamp}.csv"
            combined_df.to_csv(output_file, index=False)
            
            logger.info(f"完整数据下载完成: {len(combined_df)} 条记录")
            logger.info(f"数据已保存到: {output_file}")
            
            return combined_df
        else:
            logger.error("没有成功下载任何数据")
            return pd.DataFrame()
    
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


def download_with_limits():
    """遵守限制的数据下载"""
    logger.info("=" * 80)
    logger.info("币安限制数据下载")
    logger.info("=" * 80)
    
    # 创建下载器
    downloader = BinanceLimitedDataDownloader()
    
    # 测试连接
    logger.info("测试连接...")
    if not downloader.test_connection():
        logger.error("连接测试失败，请检查网络")
        return False
    
    # 先下载1周数据测试
    logger.info("步骤1: 下载1周测试数据")
    test_data = downloader.download_limited_data("ETHUSDT", "1m", 7)
    
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
    
    # 下载完整数据
    logger.info("步骤2: 下载6个月完整数据")
    full_data = downloader.download_monthly_data("ETHUSDT", "1m", 6)
    
    if full_data.empty:
        logger.error("完整数据下载失败")
        return False
    
    # 验证完整数据
    quality_report = downloader.validate_data_quality(full_data)
    
    logger.info("=" * 80)
    logger.info("完整数据下载成功！")
    logger.info("=" * 80)
    logger.info(f"记录数: {quality_report['total_records']:,}")
    logger.info(f"时间范围: {quality_report['time_range']['start']} 到 {quality_report['time_range']['end']}")
    logger.info(f"价格范围: {quality_report['price_range']['min_price']:.2f} - {quality_report['price_range']['max_price']:.2f}")
    logger.info(f"价格波动率: {quality_report['price_range']['price_volatility']:.4f}")
    logger.info(f"平均成交量: {quality_report['volume_stats']['avg_volume']:,.0f}")
    
    return True


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("币安限制数据下载器")
    logger.info("=" * 80)
    
    download_with_limits()


if __name__ == "__main__":
    main()
