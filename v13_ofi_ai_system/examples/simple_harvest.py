#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版OFI+CVD数据采集脚本
用于测试和演示
"""

import asyncio
import websockets
import json
import time
import logging
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleHarvester:
    """简化的数据采集器"""
    
    def __init__(self, symbols=None, run_hours=2, output_dir="data/ofi_cvd"):
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.run_hours = run_hours
        self.output_dir = Path(output_dir)
        self.start_time = time.time()
        self.end_time = self.start_time + (run_hours * 3600)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据缓冲区
        self.data_buffers = {symbol: [] for symbol in self.symbols}
        
        logger.info(f"初始化采集器: {self.symbols}, 运行{run_hours}小时")
    
    async def connect_websocket(self, symbol):
        """连接WebSocket"""
        stream_name = f"{symbol.lower()}@trade"
        url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        
        logger.info(f"连接WebSocket: {symbol}")
        
        try:
            async with websockets.connect(url, ping_interval=20) as websocket:
                logger.info(f"WebSocket连接成功: {symbol}")
                
                async for message in websocket:
                    if time.time() > self.end_time:
                        logger.info(f"达到运行时间限制，停止采集: {symbol}")
                        break
                    
                    try:
                        data = json.loads(message)
                        
                        # 处理交易数据
                        trade_data = {
                            'timestamp': datetime.now().isoformat(),
                            'event_ts_ms': data.get('E', 0),
                            'symbol': symbol,
                            'price': float(data.get('p', 0)),
                            'qty': float(data.get('q', 0)),
                            'trade_id': data.get('a', 0),
                            'is_buyer_maker': data.get('m', False)
                        }
                        
                        # 添加到缓冲区
                        self.data_buffers[symbol].append(trade_data)
                        
                        # 每100条数据保存一次
                        if len(self.data_buffers[symbol]) >= 100:
                            await self.save_data(symbol)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析错误: {e}")
                    except Exception as e:
                        logger.error(f"处理消息错误: {e}")
                        
        except Exception as e:
            logger.error(f"WebSocket连接错误: {e}")
            raise
    
    async def save_data(self, symbol):
        """保存数据到Parquet文件"""
        if not self.data_buffers[symbol]:
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.data_buffers[symbol])
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{timestamp}.parquet"
        filepath = self.output_dir / filename
        
        # 保存为Parquet
        df.to_parquet(filepath, compression='snappy', index=False)
        
        logger.info(f"保存数据: {symbol}, {len(df)}条记录, {filepath}")
        
        # 清空缓冲区
        self.data_buffers[symbol] = []
    
    async def run(self):
        """运行采集器"""
        logger.info("开始数据采集...")
        
        # 创建任务
        tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(self.connect_websocket(symbol))
            tasks.append(task)
        
        try:
            # 等待所有任务完成或超时
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.run_hours * 3600
            )
        except asyncio.TimeoutError:
            logger.info("达到运行时间限制")
        except Exception as e:
            logger.error(f"运行错误: {e}")
        finally:
            # 保存剩余数据
            for symbol in self.symbols:
                if self.data_buffers[symbol]:
                    await self.save_data(symbol)
            
            logger.info("数据采集完成")

async def main():
    """主函数"""
    # 从环境变量获取配置
    symbols = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT').split(',')
    run_hours = int(os.getenv('RUN_HOURS', '2'))
    output_dir = os.getenv('OUTPUT_DIR', 'data/ofi_cvd')
    
    # 创建采集器
    harvester = SimpleHarvester(symbols, run_hours, output_dir)
    
    # 运行采集
    await harvester.run()

if __name__ == "__main__":
    asyncio.run(main())

