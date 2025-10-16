#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保守的币安数据下载器
使用极保守的设置避免API限制
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

class ConservativeBinanceDownloader:
    """保守的币安数据下载器"""
    
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # 数据存储路径
        self.data_dir = "data/binance"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 极保守的配置
        self.config = {
            'request_delay': 2.0,      # 2秒延迟
            'batch_delay': 30,         # 批次间30秒延迟
            'batch_days': 3,           # 每批只下载3天
            'max_retries': 5,          # 增加重试次数
            'timeout': 60,             # 增加超时时间
        }
        
        logger.info("保守下载器初始化完成")
        logger.info(f"配置: 每批{self.config['batch_days']}天, 请求延迟{self.config['request_delay']}s, 批次延迟{self.config['batch_delay']}s")
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            logger.info("测试币安连接...")
            
            # 使用不同的端点测试
            test_urls = [
                "https://fapi.binance.com/fapi/v1/ping",
                "https://fapi.binance.com/fapi/v1/time",
                "https://fapi.binance.com/fapi/v1/exchangeInfo"
            ]
            
            for url in test_urls:
                try:
                    logger.info(f"测试: {url}")
                    time.sleep(self.config['request_delay'])
                    
                    response = self.session.get(url, timeout=self.config['timeout'])
                    
                    if response.status_code == 200:
                        logger.info(f"✅ {url} 连接成功")
                        return True
                    else:
                        logger.warning(f"⚠️ {url} 状态码: {response.status_code}")
                        
                except Exception as e:
                    logger.warning(f"❌ {url} 失败: {e}")
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False
    
    def download_conservative_batch(self, symbol: str = "ETHUSDT", interval: str = "1m", 
                                   days: int = 180) -> pd.DataFrame:
        """
        保守分批下载
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            days: 总下载天数
        
        Returns:
            完整数据DataFrame
        """
        logger.info(f"开始保守下载 {symbol} {interval} 数据 ({days}天)")
        
        # 计算总批次数
        total_batches = (days + self.config['batch_days'] - 1) // self.config['batch_days']
        logger.info(f"总批次数: {total_batches}")
        
        # 计算时间范围
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        logger.info(f"时间范围: {datetime.fromtimestamp(start_time/1000)} 到 {datetime.fromtimestamp(end_time/1000)}")
        
        all_batch_data = []
        
        # 分批下载
        for batch_idx in range(total_batches):
            logger.info(f"=" * 50)
            logger.info(f"批次 {batch_idx + 1}/{total_batches}")
            logger.info(f"=" * 50)
            
            # 计算当前批次的时间范围
            batch_start = start_time + (batch_idx * self.config['batch_days'] * 24 * 60 * 60 * 1000)
            batch_end = min(batch_start + (self.config['batch_days'] * 24 * 60 * 60 * 1000), end_time)
            
            logger.info(f"批次时间: {datetime.fromtimestamp(batch_start/1000)} 到 {datetime.fromtimestamp(batch_end/1000)}")
            
            # 下载当前批次
            batch_data = self._download_single_batch_conservative(symbol, interval, batch_start, batch_end, batch_idx + 1)
            
            if not batch_data.empty:
                all_batch_data.append(batch_data)
                logger.info(f"批次 {batch_idx + 1} 完成: {len(batch_data)} 条记录")
            else:
                logger.error(f"批次 {batch_idx + 1} 下载失败")
                continue
            
            # 批次间长时间延迟
            if batch_idx < total_batches - 1:
                logger.info(f"批次间休息 {self.config['batch_delay']} 秒...")
                time.sleep(self.config['batch_delay'])
        
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
            output_file = f"{self.data_dir}/{symbol}_{interval}_{days}days_conservative_{timestamp}.csv"
            combined_df.to_csv(output_file, index=False)
            
            logger.info(f"数据已保存到: {output_file}")
            
            return combined_df
        else:
            logger.error("没有成功下载任何批次数据")
            return pd.DataFrame()
    
    def _download_single_batch_conservative(self, symbol: str, interval: str, start_time: int, end_time: int, batch_num: int) -> pd.DataFrame:
        """保守下载单个批次"""
        logger.info(f"开始下载批次 {batch_num}...")
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            # 获取数据
            klines = self._get_klines_conservative(symbol, interval, current_start, end_time)
            
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
            
            # 长时间延迟
            time.sleep(self.config['request_delay'])
            
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
    
    def _get_klines_conservative(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[List]:
        """保守获取K线数据"""
        url = f"{self.base_url}/fapi/v1/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000  # 减少每次请求的数据量
        }
        
        for attempt in range(self.config['max_retries']):
            try:
                logger.info(f"请求数据: {datetime.fromtimestamp(start_time/1000)} 到 {datetime.fromtimestamp(end_time/1000)}")
                
                response = self.session.get(url, params=params, timeout=self.config['timeout'])
                
                # 检查响应状态
                if response.status_code == 429:  # 请求过于频繁
                    wait_time = 120 * (attempt + 1)  # 等待2分钟以上
                    logger.warning(f"触发API限制，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code == 418:  # I'm a teapot
                    wait_time = 180 * (attempt + 1)  # 等待3分钟以上
                    logger.warning(f"触发418错误，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"获取到 {len(data)} 条数据")
                return data
                
            except requests.exceptions.RequestException as e:
                logger.error(f"请求失败 (尝试 {attempt + 1}/{self.config['max_retries']}): {e}")
                if attempt < self.config['max_retries'] - 1:
                    wait_time = 60 * (2 ** attempt)  # 指数退避，最少1分钟
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


def download_conservative():
    """保守下载"""
    logger.info("=" * 80)
    logger.info("保守币安数据下载")
    logger.info("=" * 80)
    
    # 创建下载器
    downloader = ConservativeBinanceDownloader()
    
    # 测试连接
    if not downloader.test_connection():
        logger.error("连接测试失败")
        return False
    
    # 先下载少量数据测试
    logger.info("步骤1: 下载7天测试数据")
    test_data = downloader.download_conservative_batch("ETHUSDT", "1m", 7)
    
    if test_data.empty:
        logger.error("测试数据下载失败")
        return False
    
    logger.info(f"测试数据下载成功: {len(test_data)} 条记录")
    logger.info(f"时间范围: {test_data['timestamp'].min()} 到 {test_data['timestamp'].max()}")
    logger.info(f"价格范围: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
    
    # 如果测试成功，询问是否下载完整数据
    logger.info("✅ 测试数据下载成功！")
    logger.info("现在可以下载完整的6个月数据，但这将需要很长时间（约2-3小时）")
    
    return True


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("保守币安数据下载器")
    logger.info("=" * 80)
    
    download_conservative()


if __name__ == "__main__":
    main()
