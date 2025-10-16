#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安6个月数据下载器
专门用于下载6个月完整历史数据，优化API限制处理
"""

import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
from typing import Dict, List, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Binance6MonthsDownloader:
    """币安6个月数据下载器"""
    
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # 保守的API限制配置
        self.api_config = {
            'request_delay': 0.2,        # 每个请求间隔0.2秒
            'batch_delay': 10,           # 批次间延迟10秒
            'max_retries': 5,            # 最大重试次数
            'timeout': 30,               # 请求超时
            'batch_size': 7,             # 每批下载7天数据
            'max_requests_per_minute': 30  # 每分钟最大请求数
        }
        
        # 创建数据目录
        self.data_dir = Path("data/binance")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("币安6个月数据下载器初始化完成")
        logger.info(f"API配置: 请求延迟{self.api_config['request_delay']}s, 批次延迟{self.api_config['batch_delay']}s")
    
    def download_6months_data(self, symbol: str = "ETHUSDT", interval: str = "1m") -> bool:
        """下载6个月数据"""
        logger.info("=" * 80)
        logger.info(f"开始下载 {symbol} {interval} 6个月数据")
        logger.info("=" * 80)
        
        # 计算时间范围
        end_time = datetime.now()
        start_time = end_time - timedelta(days=180)  # 6个月
        
        logger.info(f"时间范围: {start_time} ~ {end_time}")
        
        all_data = []
        current_start = start_time
        batch_count = 0
        total_records = 0
        
        while current_start < end_time:
            try:
                # 计算当前批次的结束时间
                current_end = min(current_start + timedelta(days=self.api_config['batch_size']), end_time)
                
                logger.info(f"📊 批次 {batch_count + 1}: {current_start.strftime('%Y-%m-%d')} ~ {current_end.strftime('%Y-%m-%d')}")
                
                # 下载当前批次数据
                batch_data = self._download_batch(symbol, interval, current_start, current_end)
                
                if batch_data is not None and not batch_data.empty:
                    all_data.append(batch_data)
                    total_records += len(batch_data)
                    logger.info(f"✅ 批次 {batch_count + 1} 成功: {len(batch_data)} 条记录")
                else:
                    logger.warning(f"⚠️ 批次 {batch_count + 1} 数据为空")
                
                # 更新下一个批次的开始时间
                current_start = current_end
                batch_count += 1
                
                # 批次间延迟
                if current_start < end_time:
                    logger.info(f"⏳ 等待 {self.api_config['batch_delay']} 秒...")
                    time.sleep(self.api_config['batch_delay'])
                
            except Exception as e:
                logger.error(f"❌ 批次 {batch_count + 1} 下载失败: {e}")
                if "429" in str(e) or "Too Many Requests" in str(e):
                    logger.info(f"🔄 API限速，等待 {self.api_config['batch_delay'] * 2} 秒后重试...")
                    time.sleep(self.api_config['batch_delay'] * 2)
                else:
                    logger.error(f"❌ 非限速错误，跳过当前批次: {e}")
                    current_start += timedelta(days=self.api_config['batch_size'])
                    batch_count += 1
                    continue
        
        # 合并所有数据
        if all_data:
            logger.info("🔄 合并数据...")
            final_df = pd.concat(all_data, ignore_index=True)
            
            # 去重和排序
            final_df = final_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            
            # 保存数据
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{interval}_6months_{timestamp}.csv"
            filepath = self.data_dir / filename
            
            final_df.to_csv(filepath, index=False)
            
            logger.info("=" * 80)
            logger.info("🎉 6个月数据下载完成！")
            logger.info("=" * 80)
            logger.info(f"📊 总记录数: {len(final_df)}")
            logger.info(f"📁 保存位置: {filepath}")
            logger.info(f"📅 时间范围: {final_df['timestamp'].min()} ~ {final_df['timestamp'].max()}")
            logger.info(f"💰 价格范围: {final_df['close'].min():.2f} ~ {final_df['close'].max():.2f}")
            
            # 数据质量分析
            self._analyze_data_quality(final_df)
            
            return True
        else:
            logger.error("❌ 没有下载到任何数据")
            return False
    
    def _download_batch(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """下载单个批次数据"""
        retry_count = 0
        
        while retry_count < self.api_config['max_retries']:
            try:
                # 计算时间戳
                start_timestamp = int(start_time.timestamp() * 1000)
                end_timestamp = int(end_time.timestamp() * 1000)
                
                # 构建请求URL
                url = f"{self.base_url}/fapi/v1/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': start_timestamp,
                    'endTime': end_timestamp,
                    'limit': 1000
                }
                
                # 发送请求
                response = self.session.get(url, params=params, timeout=self.api_config['timeout'])
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if not data:
                        logger.warning("API返回空数据")
                        return pd.DataFrame()
                    
                    # 转换为DataFrame
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # 数据类型转换
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                    
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                                      'quote_asset_volume', 'taker_buy_base_asset_volume', 
                                      'taker_buy_quote_asset_volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df['number_of_trades'] = pd.to_numeric(df['number_of_trades'], errors='coerce')
                    
                    # 请求间延迟
                    time.sleep(self.api_config['request_delay'])
                    
                    return df
                    
                elif response.status_code == 429:
                    logger.warning(f"API限速 (429)，等待重试...")
                    time.sleep(self.api_config['batch_delay'])
                    retry_count += 1
                    continue
                    
                elif response.status_code == 418:
                    logger.error(f"API被阻止 (418)，等待更长时间...")
                    time.sleep(self.api_config['batch_delay'] * 3)
                    retry_count += 1
                    continue
                    
                else:
                    logger.error(f"API请求失败: {response.status_code} - {response.text}")
                    retry_count += 1
                    time.sleep(self.api_config['batch_delay'])
                    continue
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"网络请求异常: {e}")
                retry_count += 1
                time.sleep(self.api_config['batch_delay'])
                continue
                
            except Exception as e:
                logger.error(f"未知错误: {e}")
                retry_count += 1
                time.sleep(self.api_config['batch_delay'])
                continue
        
        logger.error(f"批次下载失败，已达到最大重试次数 {self.api_config['max_retries']}")
        return None
    
    def _analyze_data_quality(self, df: pd.DataFrame):
        """分析数据质量"""
        logger.info("📊 数据质量分析:")
        
        # 基础统计
        logger.info(f"   总记录数: {len(df)}")
        logger.info(f"   时间跨度: {(df['timestamp'].max() - df['timestamp'].min()).days} 天")
        logger.info(f"   缺失值: {df.isnull().sum().sum()}")
        logger.info(f"   重复值: {df.duplicated().sum()}")
        
        # 价格统计
        logger.info(f"   开盘价范围: {df['open'].min():.2f} ~ {df['open'].max():.2f}")
        logger.info(f"   收盘价范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
        logger.info(f"   成交量范围: {df['volume'].min():.2f} ~ {df['volume'].max():.2f}")
        
        # 时间间隔检查
        time_diffs = df['timestamp'].diff().dropna()
        expected_interval = pd.Timedelta(minutes=1)  # 1分钟K线
        irregular_intervals = time_diffs[time_diffs != expected_interval]
        
        if len(irregular_intervals) > 0:
            logger.warning(f"   异常时间间隔: {len(irregular_intervals)} 个")
        else:
            logger.info("   ✅ 时间间隔正常")
        
        # 价格逻辑检查
        price_errors = df[(df['high'] < df['low']) | 
                         (df['high'] < df['open']) | 
                         (df['high'] < df['close']) |
                         (df['low'] > df['open']) | 
                         (df['low'] > df['close'])]
        
        if len(price_errors) > 0:
            logger.warning(f"   价格逻辑错误: {len(price_errors)} 条")
        else:
            logger.info("   ✅ 价格逻辑正常")
    
    def download_multiple_symbols(self, symbols: List[str], interval: str = "1m"):
        """下载多个交易对的数据"""
        logger.info(f"开始下载多个交易对的6个月数据: {symbols}")
        
        results = {}
        for symbol in symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"下载 {symbol} 数据")
            logger.info(f"{'='*50}")
            
            success = self.download_6months_data(symbol, interval)
            results[symbol] = success
            
            if success:
                logger.info(f"✅ {symbol} 下载成功")
            else:
                logger.error(f"❌ {symbol} 下载失败")
            
            # 交易对间延迟
            if symbol != symbols[-1]:  # 不是最后一个交易对
                logger.info("⏳ 交易对间延迟...")
                time.sleep(self.api_config['batch_delay'])
        
        # 输出总结
        logger.info("\n" + "="*80)
        logger.info("多交易对下载总结")
        logger.info("="*80)
        
        for symbol, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            logger.info(f"{symbol}: {status}")
        
        return results


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("币安6个月数据下载器")
    logger.info("=" * 80)
    
    # 创建下载器
    downloader = Binance6MonthsDownloader()
    
    # 下载ETHUSDT数据
    success = downloader.download_6months_data("ETHUSDT", "1m")
    
    if success:
        logger.info("🎉 6个月数据下载完成！")
        
        # 可选：下载其他交易对
        other_symbols = ["BTCUSDT", "BNBUSDT"]
        logger.info(f"是否下载其他交易对数据: {other_symbols}? (y/n)")
        # 这里可以添加用户输入逻辑
        
    else:
        logger.error("❌ 6个月数据下载失败")


if __name__ == "__main__":
    main()
