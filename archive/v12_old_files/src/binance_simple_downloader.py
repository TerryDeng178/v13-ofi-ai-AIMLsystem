#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安简单数据下载器
基于官方文档的简化版本
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceSimpleDownloader:
    """币安简单数据下载器"""
    
    def __init__(self, base_url="https://fapi.binance.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # 创建数据目录
        os.makedirs("data/binance", exist_ok=True)
        
        logger.info("币安简单下载器初始化完成")
    
    def fetch_klines(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            symbol: 交易对符号 (如: "ETHUSDT")
            interval: K线周期 (如: "1m", "5m", "1h")
            limit: 获取根数 (最大1000)
            
        Returns:
            包含K线数据的DataFrame
        """
        url = f"{self.base_url}/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)  # 币安API限制
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # 转换为DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 数据类型转换
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                              'quote_asset_volume', 'taker_buy_base_asset_volume', 
                              'taker_buy_quote_asset_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['number_of_trades'] = pd.to_numeric(df['number_of_trades'], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return pd.DataFrame()
    
    def download_klines_6months(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        下载6个月K线数据
        
        Args:
            symbol: 交易对符号
            interval: K线周期
        """
        logger.info(f"🚀 开始下载 {symbol} {interval} 6个月K线数据...")
        
        all_data = []
        end_time = datetime.now()
        start_time = end_time - timedelta(days=180)
        
        current_time = start_time
        batch_count = 0
        
        while current_time < end_time:
            try:
                # 获取数据
                df = self.fetch_klines(symbol, interval, 1000)
                
                if df.empty:
                    logger.info("⚠️ 没有更多数据，下载完成")
                    break
                
                all_data.append(df)
                batch_count += 1
                current_time = df['close_time'].max()
                
                logger.info(f"📊 批次 {batch_count}: {len(df)} 条数据, 最新时间: {current_time}")
                
                # API限制保护
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ 下载失败: {e}")
                time.sleep(1)
                continue
        
        if all_data:
            # 合并数据
            final_df = pd.concat(all_data, ignore_index=True)
            final_df = final_df.drop_duplicates(subset=['open_time']).sort_values('open_time')
            
            # 重命名列以匹配V11系统
            final_df = final_df.rename(columns={
                'open_time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # 选择需要的列
            final_df = final_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # 保存数据
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_file = f"data/binance/{symbol}_{interval}_6months_{timestamp}.csv"
            final_df.to_csv(cache_file, index=False)
            
            logger.info(f"✅ K线数据下载完成: {len(final_df)} 条记录")
            logger.info(f"📁 保存位置: {cache_file}")
            logger.info(f"📅 时间范围: {final_df['timestamp'].min()} ~ {final_df['timestamp'].max()}")
            
            return final_df
        else:
            logger.error("❌ 没有下载到任何数据")
            return pd.DataFrame()
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            url = f"{self.base_url}/fapi/v1/ping"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            logger.info("✅ 币安API连接成功")
            return True
        except Exception as e:
            logger.error(f"❌ 币安API连接失败: {e}")
            return False


def download_ethusdt_6months():
    """下载ETHUSDT 6个月数据"""
    logger.info("=" * 80)
    logger.info("币安ETHUSDT 6个月数据下载")
    logger.info("=" * 80)
    
    # 创建下载器
    downloader = BinanceSimpleDownloader()
    
    # 测试连接
    if not downloader.test_connection():
        logger.error("连接测试失败，请检查网络")
        return False
    
    # 下载ETHUSDT 1分钟数据
    logger.info("开始下载ETHUSDT 1分钟K线数据...")
    df = downloader.download_klines_6months("ETHUSDT", "1m")
    
    if not df.empty:
        logger.info("=" * 60)
        logger.info("数据质量报告")
        logger.info("=" * 60)
        logger.info(f"记录数: {len(df):,}")
        logger.info(f"时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        logger.info(f"价格范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
        logger.info(f"平均成交量: {df['volume'].mean():,.0f}")
        logger.info(f"缺失值: {df.isnull().sum().sum()}")
        logger.info(f"重复值: {df.duplicated().sum()}")
        
        logger.info("=" * 80)
        logger.info("🎉 ETHUSDT 6个月数据下载完成！")
        logger.info("现在可以开始V11训练了。")
        logger.info("=" * 80)
        
        return True
    else:
        logger.error("❌ 数据下载失败")
        return False


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("币安简单数据下载器")
    logger.info("=" * 80)
    
    download_ethusdt_6months()


if __name__ == "__main__":
    main()
