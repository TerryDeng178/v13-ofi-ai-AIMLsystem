#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安官方API下载器
严格按照币安官方文档规范实现
参考: https://binance-docs.github.io/apidocs/futures/cn
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

class BinanceOfficialAPIDownloader:
    """币安官方API下载器"""
    
    def __init__(self):
        # 币安期货API官方端点
        self.base_url = "https://fapi.binance.com"
        self.session = requests.Session()
        
        # 官方推荐的请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        })
        
        # 数据存储路径
        self.data_dir = "data/binance"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 根据官方文档的API限制配置
        self.api_limits = {
            'requests_per_minute': 1200,     # 现货API每分钟1200次
            'requests_per_second': 20,       # 保守估计每秒20次
            'weight_per_minute': 6000,       # 每分钟6000权重
            'kline_weight': 1,               # K线数据权重为1
        }
        
        # 保守的下载配置
        self.config = {
            'request_delay': 0.5,            # 500ms延迟，确保不超过2req/s
            'batch_delay': 10,               # 批次间10秒延迟
            'batch_size': 1000,              # 每批1000条记录（官方最大限制）
            'max_retries': 3,                # 最大重试次数
            'timeout': 30,                   # 请求超时
        }
        
        logger.info("币安官方API下载器初始化完成")
        logger.info(f"API限制: {self.api_limits['requests_per_second']} req/s, {self.api_limits['requests_per_minute']} req/min")
        logger.info(f"配置: 请求延迟{self.config['request_delay']}s, 批次延迟{self.config['batch_delay']}s")
    
    def test_connection(self) -> bool:
        """测试连接 - 使用官方ping端点"""
        try:
            logger.info("测试币安官方API连接...")
            
            # 使用官方ping端点
            ping_url = f"{self.base_url}/fapi/v1/ping"
            response = self.session.get(ping_url, timeout=self.config['timeout'])
            
            if response.status_code == 200:
                logger.info("✅ 币安API连接成功")
                return True
            else:
                logger.error(f"❌ 币安API连接失败: HTTP {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 连接测试异常: {e}")
            return False
    
    def get_server_time(self) -> int:
        """获取服务器时间 - 使用官方time端点"""
        try:
            url = f"{self.base_url}/fapi/v1/time"
            response = self.session.get(url, timeout=self.config['timeout'])
            response.raise_for_status()
            data = response.json()
            return data['serverTime']
        except Exception as e:
            logger.error(f"获取服务器时间失败: {e}")
            return int(time.time() * 1000)
    
    def get_exchange_info(self) -> Dict:
        """获取交易对信息"""
        try:
            url = f"{self.base_url}/fapi/v1/exchangeInfo"
            response = self.session.get(url, timeout=self.config['timeout'])
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取交易对信息失败: {e}")
            return {}
    
    def download_klines_official(self, symbol: str = "ETHUSDT", interval: str = "1m", 
                               start_time: int = None, end_time: int = None, 
                               limit: int = 1000) -> pd.DataFrame:
        """
        下载K线数据 - 严格按照官方文档规范
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔 (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）
            limit: 返回数据条数（最大1000）
        
        Returns:
            数据DataFrame
        """
        logger.info(f"开始下载 {symbol} {interval} K线数据")
        
        # 构建请求参数
        url = f"{self.base_url}/fapi/v1/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000)  # 官方限制最大1000
        }
        
        # 添加时间参数
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        logger.info(f"请求参数: {params}")
        
        # 发送请求
        for attempt in range(self.config['max_retries']):
            try:
                # 请求延迟
                time.sleep(self.config['request_delay'])
                
                response = self.session.get(url, params=params, timeout=self.config['timeout'])
                
                # 检查响应状态
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ 成功获取 {len(data)} 条K线数据")
                    return self._convert_klines_to_dataframe(data)
                
                elif response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"⚠️ 触发API限制 (429)，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 418:
                    wait_time = 120 * (attempt + 1)
                    logger.warning(f"⚠️ 触发418错误，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"❌ HTTP错误: {response.status_code}")
                    logger.error(f"响应内容: {response.text}")
                    return pd.DataFrame()
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ 请求异常 (尝试 {attempt + 1}/{self.config['max_retries']}): {e}")
                if attempt < self.config['max_retries'] - 1:
                    time.sleep(5)
                continue
            except Exception as e:
                logger.error(f"❌ 处理异常: {e}")
                break
        
        logger.error("❌ 所有重试均失败")
        return pd.DataFrame()
    
    def _convert_klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """
        将K线数据转换为DataFrame
        按照官方文档的返回格式
        """
        if not klines:
            return pd.DataFrame()
        
        # 官方文档中的K线数据格式
        columns = [
            'open_time',        # 开盘时间
            'open',             # 开盘价
            'high',             # 最高价
            'low',              # 最低价
            'close',            # 收盘价
            'volume',           # 成交量
            'close_time',       # 收盘时间
            'quote_volume',     # 成交额
            'trades',           # 成交笔数
            'taker_buy_base_volume',    # 主动买入成交量
            'taker_buy_quote_volume',   # 主动买入成交额
            'ignore'            # 忽略字段
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
    
    def download_historical_data_batch(self, symbol: str = "ETHUSDT", interval: str = "1m", 
                                     days: int = 30) -> pd.DataFrame:
        """
        分批下载历史数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            days: 下载天数
        
        Returns:
            完整数据DataFrame
        """
        logger.info(f"开始分批下载 {symbol} {interval} 历史数据 ({days}天)")
        
        # 计算时间范围
        end_time = self.get_server_time()
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        logger.info(f"时间范围: {datetime.fromtimestamp(start_time/1000)} 到 {datetime.fromtimestamp(end_time/1000)}")
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            # 计算当前批次的结束时间
            current_end = min(current_start + (self.config['batch_size'] * 60 * 1000), end_time)
            
            logger.info(f"下载批次: {datetime.fromtimestamp(current_start/1000)} 到 {datetime.fromtimestamp(current_end/1000)}")
            
            # 下载当前批次
            batch_data = self.download_klines_official(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=current_end,
                limit=self.config['batch_size']
            )
            
            if not batch_data.empty:
                all_data.append(batch_data)
                logger.info(f"批次完成: {len(batch_data)} 条记录")
            else:
                logger.warning("批次无数据，停止下载")
                break
            
            # 更新起始时间
            if not batch_data.empty:
                last_timestamp = int(batch_data['timestamp'].iloc[-1].timestamp() * 1000)
                current_start = last_timestamp + 1
            else:
                break
            
            # 批次间延迟
            time.sleep(self.config['batch_delay'])
            
            # 显示进度
            progress = (current_start - start_time) / (end_time - start_time) * 100
            logger.info(f"下载进度: {progress:.1f}%, 已获取: {sum(len(df) for df in all_data)} 条记录")
        
        # 合并所有数据
        if all_data:
            logger.info("合并所有批次数据...")
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # 去重
            original_len = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            logger.info(f"数据合并完成: {len(combined_df)} 条记录 (去重前: {original_len})")
            
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


def test_official_api():
    """测试官方API"""
    logger.info("=" * 80)
    logger.info("币安官方API测试")
    logger.info("=" * 80)
    
    # 创建下载器
    downloader = BinanceOfficialAPIDownloader()
    
    # 测试连接
    if not downloader.test_connection():
        logger.error("连接测试失败，请检查网络")
        return False
    
    # 获取交易对信息
    logger.info("获取交易对信息...")
    exchange_info = downloader.get_exchange_info()
    if exchange_info:
        logger.info("✅ 交易对信息获取成功")
        symbols = [s['symbol'] for s in exchange_info.get('symbols', [])]
        logger.info(f"可用交易对数量: {len(symbols)}")
        if 'ETHUSDT' in symbols:
            logger.info("✅ ETHUSDT 交易对可用")
        else:
            logger.warning("⚠️ ETHUSDT 交易对不可用")
    else:
        logger.warning("⚠️ 交易对信息获取失败")
    
    # 下载少量数据测试
    logger.info("下载少量数据测试...")
    test_data = downloader.download_klines_official("ETHUSDT", "1m", limit=100)
    
    if not test_data.empty:
        logger.info(f"✅ 测试数据下载成功: {len(test_data)} 条记录")
        
        # 验证数据质量
        quality_report = downloader.validate_data_quality(test_data)
        
        logger.info("=" * 50)
        logger.info("数据质量报告")
        logger.info("=" * 50)
        logger.info(f"记录数: {quality_report['total_records']:,}")
        logger.info(f"时间范围: {quality_report['time_range']['start']} 到 {quality_report['time_range']['end']}")
        logger.info(f"价格范围: {quality_report['price_range']['min_price']:.2f} - {quality_report['price_range']['max_price']:.2f}")
        logger.info(f"价格波动率: {quality_report['price_range']['price_volatility']:.4f}")
        logger.info(f"平均成交量: {quality_report['volume_stats']['avg_volume']:,.0f}")
        
        # 保存测试数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_file = f"data/binance/ETHUSDT_1m_official_test_{timestamp}.csv"
        test_data.to_csv(test_file, index=False)
        logger.info(f"测试数据已保存到: {test_file}")
        
        return True
    else:
        logger.error("❌ 测试数据下载失败")
        return False


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("币安官方API下载器")
    logger.info("=" * 80)
    
    test_official_api()


if __name__ == "__main__":
    main()
