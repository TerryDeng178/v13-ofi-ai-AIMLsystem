#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安数据下载器
下载ETH/USDT永续合约历史数据用于V11训练
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

class BinanceDataDownloader:
    """币安数据下载器"""
    
    def __init__(self, base_url: str = "https://fapi.binance.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # 数据存储路径
        self.data_dir = "data/binance"
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("币安数据下载器初始化完成")
    
    def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, 
                   limit: int = 1500) -> List[List]:
        """
        获取K线数据
        
        Args:
            symbol: 交易对符号 (如: ETHUSDT)
            interval: 时间间隔 (如: 1m, 5m, 15m, 1h, 4h, 1d)
            start_time: 开始时间戳 (毫秒)
            end_time: 结束时间戳 (毫秒)
            limit: 每次请求的数据条数 (最大1500)
        
        Returns:
            K线数据列表
        """
        url = f"{self.base_url}/fapi/v1/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {e}")
            return []
        except Exception as e:
            logger.error(f"数据处理失败: {e}")
            return []
    
    def download_historical_data(self, symbol: str = "ETHUSDT", interval: str = "1m", 
                                months: int = 6, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        下载历史数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            months: 下载月数
            output_file: 输出文件名
        
        Returns:
            包含历史数据的DataFrame
        """
        logger.info(f"开始下载 {symbol} {interval} 数据，时间跨度: {months} 个月")
        
        # 计算时间范围
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=months * 30)).timestamp() * 1000)
        
        logger.info(f"时间范围: {datetime.fromtimestamp(start_time/1000)} 到 {datetime.fromtimestamp(end_time/1000)}")
        
        all_data = []
        current_start = start_time
        
        # 分批下载数据
        while current_start < end_time:
            logger.info(f"下载进度: {datetime.fromtimestamp(current_start/1000)}")
            
            # 获取数据
            klines = self.get_klines(symbol, interval, current_start, end_time, limit=1500)
            
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
            
            # 避免请求过于频繁
            time.sleep(0.1)
        
        # 转换为DataFrame
        df = self._convert_to_dataframe(all_data)
        
        # 保存数据
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.data_dir}/{symbol}_{interval}_{months}months_{timestamp}.csv"
        
        df.to_csv(output_file, index=False)
        logger.info(f"数据已保存到: {output_file}")
        logger.info(f"总数据量: {len(df)} 条记录")
        
        return df
    
    def _convert_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """
        将K线数据转换为DataFrame
        
        Args:
            klines: K线数据列表
        
        Returns:
            格式化的DataFrame
        """
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
    
    def get_orderbook_data(self, symbol: str = "ETHUSDT", limit: int = 1000) -> Dict:
        """
        获取订单簿数据 (用于OFI计算)
        
        Args:
            symbol: 交易对符号
            limit: 订单簿深度
        
        Returns:
            订单簿数据
        """
        url = f"{self.base_url}/fapi/v1/depth"
        
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data
            
        except Exception as e:
            logger.error(f"获取订单簿数据失败: {e}")
            return {}
    
    def download_multiple_intervals(self, symbol: str = "ETHUSDT", months: int = 6) -> Dict[str, pd.DataFrame]:
        """
        下载多个时间间隔的数据
        
        Args:
            symbol: 交易对符号
            months: 下载月数
        
        Returns:
            不同时间间隔的数据字典
        """
        intervals = ['1m', '5m', '15m', '1h', '4h', '1d']
        data_dict = {}
        
        logger.info(f"开始下载 {symbol} 多时间间隔数据")
        
        for interval in intervals:
            logger.info(f"下载 {interval} 数据...")
            
            try:
                df = self.download_historical_data(symbol, interval, months)
                data_dict[interval] = df
                
                logger.info(f"{interval} 数据下载完成: {len(df)} 条记录")
                
                # 避免请求过于频繁
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"下载 {interval} 数据失败: {e}")
                continue
        
        return data_dict
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        验证数据质量
        
        Args:
            df: 数据DataFrame
        
        Returns:
            数据质量报告
        """
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
        
        # 检查数据连续性
        time_diff = df['timestamp'].diff().dropna()
        expected_interval = pd.Timedelta(minutes=1)  # 假设1分钟数据
        gaps = time_diff[time_diff > expected_interval * 2]
        
        report["data_continuity"] = {
            "expected_interval": str(expected_interval),
            "time_gaps": len(gaps),
            "largest_gap": str(gaps.max()) if len(gaps) > 0 else "None"
        }
        
        return report
    
    def save_data_summary(self, data_dict: Dict[str, pd.DataFrame], symbol: str = "ETHUSDT"):
        """
        保存数据摘要报告
        
        Args:
            data_dict: 数据字典
            symbol: 交易对符号
        """
        summary = {
            "symbol": symbol,
            "download_time": datetime.now().isoformat(),
            "intervals": {}
        }
        
        for interval, df in data_dict.items():
            if not df.empty:
                quality_report = self.validate_data_quality(df)
                summary["intervals"][interval] = quality_report
        
        # 保存摘要报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"{self.data_dir}/{symbol}_data_summary_{timestamp}.json"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"数据摘要报告已保存到: {summary_file}")
        
        return summary


def main():
    """主函数 - 下载币安数据"""
    logger.info("=" * 80)
    logger.info("币安数据下载器")
    logger.info("=" * 80)
    
    # 创建下载器
    downloader = BinanceDataDownloader()
    
    # 下载ETH/USDT永续合约数据
    symbol = "ETHUSDT"
    months = 6
    
    logger.info(f"开始下载 {symbol} 永续合约 {months} 个月历史数据")
    
    try:
        # 下载多时间间隔数据
        data_dict = downloader.download_multiple_intervals(symbol, months)
        
        # 保存数据摘要
        summary = downloader.save_data_summary(data_dict, symbol)
        
        # 输出摘要信息
        logger.info("=" * 80)
        logger.info("数据下载完成摘要")
        logger.info("=" * 80)
        
        for interval, info in summary["intervals"].items():
            logger.info(f"{interval} 数据:")
            logger.info(f"  记录数: {info['total_records']:,}")
            logger.info(f"  时间范围: {info['time_range']['start']} 到 {info['time_range']['end']}")
            logger.info(f"  价格范围: {info['price_range']['min_price']:.2f} - {info['price_range']['max_price']:.2f}")
            logger.info(f"  价格波动率: {info['price_range']['price_volatility']:.4f}")
            logger.info(f"  平均成交量: {info['volume_stats']['avg_volume']:,.0f}")
            logger.info(f"  数据连续性: {info['data_continuity']['time_gaps']} 个时间缺口")
            logger.info("")
        
        logger.info("币安数据下载完成！")
        
        return data_dict
        
    except Exception as e:
        logger.error(f"下载过程中出现错误: {e}")
        return {}


if __name__ == "__main__":
    main()
