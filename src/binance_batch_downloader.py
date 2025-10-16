#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安分批数据下载器
智能分批下载，严格遵守API限制
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

class BinanceBatchDownloader:
    """币安分批数据下载器"""
    
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
            'requests_per_minute': 1200,  # 保守估计，每分钟1200次
            'requests_per_second': 20,    # 保守估计，每秒20次
            'weight_per_minute': 6000,    # 每分钟6000权重
            'kline_weight': 1,            # K线数据权重为1
        }
        
        # 分批下载配置
        self.batch_config = {
            'batch_days': 7,              # 每批下载7天
            'request_delay': 0.05,        # 50ms延迟
            'batch_delay': 5,             # 批次间延迟5秒
            'max_retries': 3,             # 最大重试次数
            'timeout': 30,                # 请求超时
        }
        
        logger.info("币安分批下载器初始化完成")
        logger.info(f"批次配置: 每批{self.batch_config['batch_days']}天, 延迟{self.batch_config['request_delay']}s")
    
    def download_data_in_batches(self, symbol: str = "ETHUSDT", interval: str = "1m", 
                                total_days: int = 180) -> pd.DataFrame:
        """
        分批下载数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            total_days: 总下载天数
        
        Returns:
            完整数据DataFrame
        """
        logger.info(f"开始分批下载 {symbol} {interval} 数据")
        logger.info(f"总天数: {total_days}天, 每批: {self.batch_config['batch_days']}天")
        
        # 计算总批次数
        total_batches = (total_days + self.batch_config['batch_days'] - 1) // self.batch_config['batch_days']
        logger.info(f"总批次数: {total_batches}")
        
        # 计算时间范围
        end_time = int(time.time() * 1000)
        start_time = end_time - (total_days * 24 * 60 * 60 * 1000)
        
        logger.info(f"时间范围: {datetime.fromtimestamp(start_time/1000)} 到 {datetime.fromtimestamp(end_time/1000)}")
        
        all_batch_data = []
        
        # 分批下载
        for batch_idx in range(total_batches):
            logger.info(f"=" * 60)
            logger.info(f"下载批次 {batch_idx + 1}/{total_batches}")
            logger.info(f"=" * 60)
            
            # 计算当前批次的时间范围
            batch_start = start_time + (batch_idx * self.batch_config['batch_days'] * 24 * 60 * 60 * 1000)
            batch_end = min(batch_start + (self.batch_config['batch_days'] * 24 * 60 * 60 * 1000), end_time)
            
            logger.info(f"批次时间: {datetime.fromtimestamp(batch_start/1000)} 到 {datetime.fromtimestamp(batch_end/1000)}")
            
            # 下载当前批次
            batch_data = self._download_single_batch(symbol, interval, batch_start, batch_end, batch_idx + 1)
            
            if not batch_data.empty:
                all_batch_data.append(batch_data)
                logger.info(f"批次 {batch_idx + 1} 完成: {len(batch_data)} 条记录")
            else:
                logger.error(f"批次 {batch_idx + 1} 下载失败")
                continue
            
            # 批次间延迟
            if batch_idx < total_batches - 1:
                logger.info(f"批次间休息 {self.batch_config['batch_delay']} 秒...")
                time.sleep(self.batch_config['batch_delay'])
        
        # 合并所有批次数据
        if all_batch_data:
            logger.info("合并所有批次数据...")
            combined_df = pd.concat(all_batch_data, ignore_index=True)
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # 去重
            original_len = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            logger.info(f"数据合并完成: {len(combined_df)} 条记录 (去重前: {original_len})")
            
            # 保存数据
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.data_dir}/{symbol}_{interval}_{total_days}days_batch_{timestamp}.csv"
            combined_df.to_csv(output_file, index=False)
            
            logger.info(f"数据已保存到: {output_file}")
            
            return combined_df
        else:
            logger.error("没有成功下载任何批次数据")
            return pd.DataFrame()
    
    def _download_single_batch(self, symbol: str, interval: str, start_time: int, end_time: int, batch_num: int) -> pd.DataFrame:
        """
        下载单个批次数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            start_time: 开始时间戳
            end_time: 结束时间戳
            batch_num: 批次号
        
        Returns:
            批次数据DataFrame
        """
        logger.info(f"开始下载批次 {batch_num}...")
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            # 获取数据
            klines = self._get_klines_with_retry(symbol, interval, current_start, end_time)
            
            if not klines:
                logger.warning(f"批次 {batch_num} 未获取到数据，停止")
                break
            
            all_data.extend(klines)
            
            # 更新起始时间
            if klines:
                last_timestamp = int(klines[-1][0])
                current_start = last_timestamp + 1
            else:
                break
            
            # 请求间延迟
            time.sleep(self.batch_config['request_delay'])
            
            # 显示进度
            progress = (current_start - start_time) / (end_time - start_time) * 100
            logger.info(f"批次 {batch_num} 进度: {progress:.1f}%, 已获取: {len(all_data)} 条")
        
        # 转换为DataFrame
        if all_data:
            df = self._convert_to_dataframe(all_data)
            logger.info(f"批次 {batch_num} 完成: {len(df)} 条记录")
            return df
        else:
            logger.error(f"批次 {batch_num} 无数据")
            return pd.DataFrame()
    
    def _get_klines_with_retry(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[List]:
        """
        带重试机制的K线数据获取
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            start_time: 开始时间戳
            end_time: 结束时间戳
        
        Returns:
            K线数据列表
        """
        url = f"{self.base_url}/fapi/v1/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1500  # 最大限制
        }
        
        for attempt in range(self.batch_config['max_retries']):
            try:
                response = self.session.get(url, params=params, timeout=self.batch_config['timeout'])
                
                # 检查响应状态
                if response.status_code == 429:  # 请求过于频繁
                    wait_time = 60 * (attempt + 1)  # 递增等待时间
                    logger.warning(f"触发API限制，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                return data
                
            except requests.exceptions.RequestException as e:
                logger.error(f"请求失败 (尝试 {attempt + 1}/{self.batch_config['max_retries']}): {e}")
                if attempt < self.batch_config['max_retries'] - 1:
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


def download_binance_data_batch():
    """分批下载币安数据"""
    logger.info("=" * 80)
    logger.info("币安分批数据下载")
    logger.info("=" * 80)
    
    # 创建下载器
    downloader = BinanceBatchDownloader()
    
    # 测试连接
    logger.info("测试连接...")
    if not downloader.test_connection():
        logger.error("连接测试失败，请检查网络")
        return False
    
    # 分批下载6个月数据 (约180天)
    logger.info("开始分批下载ETH/USDT永续合约6个月数据...")
    start_time = time.time()
    
    full_data = downloader.download_data_in_batches("ETHUSDT", "1m", 180)
    
    end_time = time.time()
    duration = end_time - start_time
    
    if full_data.empty:
        logger.error("分批数据下载失败")
        return False
    
    # 验证数据质量
    quality_report = downloader.validate_data_quality(full_data)
    
    logger.info("=" * 80)
    logger.info("分批下载完成！")
    logger.info("=" * 80)
    logger.info(f"下载耗时: {duration/60:.1f} 分钟")
    logger.info(f"记录数: {quality_report['total_records']:,}")
    logger.info(f"时间范围: {quality_report['time_range']['start']} 到 {quality_report['time_range']['end']}")
    logger.info(f"价格范围: {quality_report['price_range']['min_price']:.2f} - {quality_report['price_range']['max_price']:.2f}")
    logger.info(f"价格波动率: {quality_report['price_range']['price_volatility']:.4f}")
    logger.info(f"平均成交量: {quality_report['volume_stats']['avg_volume']:,.0f}")
    logger.info(f"缺失值: {quality_report['missing_values']}")
    logger.info(f"重复值: {quality_report['duplicates']}")
    
    logger.info("✅ 分批数据下载成功！")
    return True


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("币安分批数据下载器")
    logger.info("=" * 80)
    
    download_binance_data_batch()


if __name__ == "__main__":
    main()
