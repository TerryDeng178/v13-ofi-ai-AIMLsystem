#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整版OFI+CVD数据采集脚本 (Task 1.3.1 v2)
实现5类数据集：prices, ofi, cvd, fusion, events
"""

import asyncio
import websockets
import json
import time
import logging
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from typing import Dict, List, Any, Optional
import math

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteOFICVDHarvester:
    """完整版OFI+CVD数据采集器"""
    
    def __init__(self, symbols=None, run_hours=2, output_dir="data/ofi_cvd"):
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.run_hours = run_hours
        self.output_dir = Path(output_dir)
        self.start_time = time.time()
        self.end_time = self.start_time + (run_hours * 3600)
        
        # 创建输出目录结构
        self._create_directory_structure()
        
        # 数据缓冲区
        self.data_buffers = {
            'prices': {symbol: [] for symbol in self.symbols},
            'ofi': {symbol: [] for symbol in self.symbols},
            'cvd': {symbol: [] for symbol in self.symbols},
            'fusion': {symbol: [] for symbol in self.symbols},
            'events': {symbol: [] for symbol in self.symbols}
        }
        
        # OFI计算参数
        self.ofi_config = {
            'levels': 5,
            'weights': [0.4, 0.3, 0.2, 0.08, 0.02],
            'z_window': 100,
            'ema_alpha': 0.1
        }
        
        # CVD计算参数
        self.cvd_config = {
            'z_window': 100,
            'ema_alpha': 0.1
        }
        
        # 订单簿缓存
        self.orderbooks = {symbol: {} for symbol in self.symbols}
        
        # 统计信息
        self.stats = {
            'total_trades': {symbol: 0 for symbol in self.symbols},
            'total_ofi': {symbol: 0 for symbol in self.symbols},
            'total_cvd': {symbol: 0 for symbol in self.symbols},
            'total_events': {symbol: 0 for symbol in self.symbols}
        }
        
        logger.info(f"初始化完整版采集器: {self.symbols}, 运行{run_hours}小时")
    
    def _create_directory_structure(self):
        """创建目录结构"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        for symbol in self.symbols:
            for kind in ['prices', 'ofi', 'cvd', 'fusion', 'events']:
                dir_path = self.output_dir / f"date={today}" / f"symbol={symbol}" / f"kind={kind}"
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # 创建artifacts目录
        (Path("artifacts") / "run_logs").mkdir(parents=True, exist_ok=True)
        (Path("artifacts") / "dq_reports").mkdir(parents=True, exist_ok=True)
    
    def _calculate_ofi(self, symbol: str, orderbook: Dict) -> Optional[Dict]:
        """计算OFI指标"""
        try:
            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                return None
            
            bids = orderbook['bids'][:self.ofi_config['levels']]
            asks = orderbook['asks'][:self.ofi_config['levels']]
            
            # 填充到5档
            while len(bids) < self.ofi_config['levels']:
                bids.append([0.0, 0.0])
            while len(asks) < self.ofi_config['levels']:
                asks.append([0.0, 0.0])
            
            # 计算OFI
            ofi_value = 0.0
            for i in range(self.ofi_config['levels']):
                bid_qty = max(0, bids[i][1])
                ask_qty = max(0, asks[i][1])
                weight = self.ofi_config['weights'][i]
                ofi_value += weight * (bid_qty - ask_qty)
            
            # 简化的Z-score计算
            ofi_z = ofi_value / 1000.0  # 简化处理
            
            return {
                'ofi_value': ofi_value,
                'ofi_z': ofi_z,
                'scale': 1000.0,
                'regime': 'normal'
            }
        except Exception as e:
            logger.error(f"OFI计算错误 {symbol}: {e}")
            return None
    
    def _calculate_cvd(self, symbol: str, trade_data: Dict) -> Optional[Dict]:
        """计算CVD指标"""
        try:
            # 简化的CVD计算
            price = trade_data.get('price', 0)
            qty = trade_data.get('qty', 0)
            is_buy = not trade_data.get('is_buyer_maker', False)
            
            # 累积成交量差
            delta = qty if is_buy else -qty
            
            # 简化的Z-score计算
            cvd_z = delta / 100.0  # 简化处理
            
            return {
                'cvd': delta,
                'delta': delta,
                'z_raw': cvd_z,
                'z_cvd': cvd_z,
                'scale': 100.0,
                'sigma_floor': 0.0,
                'floor_used': False,
                'regime': 'normal'
            }
        except Exception as e:
            logger.error(f"CVD计算错误 {symbol}: {e}")
            return None
    
    def _calculate_fusion(self, ofi_result: Dict, cvd_result: Dict) -> Optional[Dict]:
        """计算融合指标"""
        try:
            if not ofi_result or not cvd_result:
                return None
            
            # 融合分数
            score = 0.6 * ofi_result.get('ofi_z', 0) + 0.4 * cvd_result.get('z_cvd', 0)
            score_z = score  # 简化处理
            
            return {
                'score': score,
                'score_z': score_z,
                'regime': 'normal'
            }
        except Exception as e:
            logger.error(f"融合计算错误: {e}")
            return None
    
    def _detect_events(self, ofi_result: Dict, cvd_result: Dict, price: float) -> List[Dict]:
        """检测事件（背离/枢轴/异常）"""
        events = []
        
        try:
            if not ofi_result or not cvd_result:
                return events
            
            ofi_z = ofi_result.get('ofi_z', 0)
            cvd_z = cvd_result.get('z_cvd', 0)
            
            # 检测背离
            if abs(ofi_z) > 2.0 and abs(cvd_z) > 2.0:
                if (ofi_z > 0 and cvd_z < 0) or (ofi_z < 0 and cvd_z > 0):
                    events.append({
                        'event_type': 'divergence',
                        'meta_json': json.dumps({
                            'ofi_z': ofi_z,
                            'cvd_z': cvd_z,
                            'price': price
                        })
                    })
            
            # 检测异常
            if abs(ofi_z) > 3.0 or abs(cvd_z) > 3.0:
                events.append({
                    'event_type': 'anomaly',
                    'meta_json': json.dumps({
                        'ofi_z': ofi_z,
                        'cvd_z': cvd_z,
                        'price': price
                    })
                })
            
        except Exception as e:
            logger.error(f"事件检测错误: {e}")
        
        return events
    
    async def _process_trade_data(self, symbol: str, trade_data: Dict):
        """处理交易数据"""
        try:
            current_time = int(time.time() * 1000)
            
            # 1. 保存价格数据
            price_data = {
                'ts_ms': trade_data.get('event_ts_ms', current_time),
                'event_ts_ms': trade_data.get('event_ts_ms', current_time),
                'symbol': symbol,
                'price': trade_data.get('price', 0),
                'qty': trade_data.get('qty', 0),
                'agg_trade_id': trade_data.get('trade_id', 0),
                'latency_ms': current_time - trade_data.get('event_ts_ms', current_time),
                'recv_rate_tps': 1.0  # 简化处理
            }
            self.data_buffers['prices'][symbol].append(price_data)
            
            # 2. 计算OFI（如果有订单簿数据）
            ofi_result = None
            if symbol in self.orderbooks and self.orderbooks[symbol]:
                ofi_result = self._calculate_ofi(symbol, self.orderbooks[symbol])
                if ofi_result:
                    ofi_data = {
                        'ts_ms': current_time,
                        'symbol': symbol,
                        'ofi_value': ofi_result['ofi_value'],
                        'ofi_z': ofi_result['ofi_z'],
                        'scale': ofi_result['scale'],
                        'regime': ofi_result['regime']
                    }
                    self.data_buffers['ofi'][symbol].append(ofi_data)
                    self.stats['total_ofi'][symbol] += 1
            
            # 3. 计算CVD
            cvd_result = self._calculate_cvd(symbol, trade_data)
            if cvd_result:
                cvd_data = {
                    'ts_ms': current_time,
                    'symbol': symbol,
                    'cvd': cvd_result['cvd'],
                    'delta': cvd_result['delta'],
                    'z_raw': cvd_result['z_raw'],
                    'z_cvd': cvd_result['z_cvd'],
                    'scale': cvd_result['scale'],
                    'sigma_floor': cvd_result['sigma_floor'],
                    'floor_used': cvd_result['floor_used'],
                    'regime': cvd_result['regime']
                }
                self.data_buffers['cvd'][symbol].append(cvd_data)
                self.stats['total_cvd'][symbol] += 1
            
            # 4. 计算融合指标
            fusion_result = self._calculate_fusion(ofi_result, cvd_result)
            if fusion_result:
                fusion_data = {
                    'ts_ms': current_time,
                    'symbol': symbol,
                    'score': fusion_result['score'],
                    'score_z': fusion_result['score_z'],
                    'regime': fusion_result['regime']
                }
                self.data_buffers['fusion'][symbol].append(fusion_data)
            
            # 5. 检测事件
            events = self._detect_events(ofi_result, cvd_result, trade_data.get('price', 0))
            for event in events:
                event_data = {
                    'ts_ms': current_time,
                    'symbol': symbol,
                    'event_type': event['event_type'],
                    'meta_json': event['meta_json']
                }
                self.data_buffers['events'][symbol].append(event_data)
                self.stats['total_events'][symbol] += 1
            
            self.stats['total_trades'][symbol] += 1
            
        except Exception as e:
            logger.error(f"处理交易数据错误 {symbol}: {e}")
    
    async def _process_orderbook_data(self, symbol: str, orderbook_data: Dict):
        """处理订单簿数据"""
        try:
            if 'bids' in orderbook_data and 'asks' in orderbook_data:
                self.orderbooks[symbol] = orderbook_data
        except Exception as e:
            logger.error(f"处理订单簿数据错误 {symbol}: {e}")
    
    async def _save_data(self, symbol: str, kind: str):
        """保存数据到Parquet文件"""
        if not self.data_buffers[kind][symbol]:
            return
        
        try:
            # 创建DataFrame
            df = pd.DataFrame(self.data_buffers[kind][symbol])
            
            # 添加日期分区
            df['date'] = pd.to_datetime(df['ts_ms'], unit='ms').dt.date.astype(str)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"part-{timestamp}.parquet"
            
            # 确定保存路径
            today = datetime.now().strftime("%Y-%m-%d")
            filepath = self.output_dir / f"date={today}" / f"symbol={symbol}" / f"kind={kind}" / filename
            
            # 保存为Parquet
            df.to_parquet(filepath, compression='snappy', index=False)
            
            logger.info(f"保存数据: {symbol}-{kind}, {len(df)}条记录, {filepath}")
            
            # 清空缓冲区
            self.data_buffers[kind][symbol] = []
            
        except Exception as e:
            logger.error(f"保存数据错误 {symbol}-{kind}: {e}")
    
    async def connect_trade_stream(self, symbol: str):
        """连接交易流"""
        stream_name = f"{symbol.lower()}@trade"
        url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        
        logger.info(f"连接交易流: {symbol}")
        
        try:
            async with websockets.connect(url, ping_interval=20) as websocket:
                logger.info(f"交易流连接成功: {symbol}")
                
                async for message in websocket:
                    if time.time() > self.end_time:
                        logger.info(f"达到运行时间限制，停止采集: {symbol}")
                        break
                    
                    try:
                        data = json.loads(message)
                        
                        # 处理交易数据
                        trade_data = {
                            'event_ts_ms': data.get('E', 0),
                            'symbol': symbol,
                            'price': float(data.get('p', 0)),
                            'qty': float(data.get('q', 0)),
                            'trade_id': data.get('a', 0),
                            'is_buyer_maker': data.get('m', False)
                        }
                        
                        await self._process_trade_data(symbol, trade_data)
                        
                        # 每50条数据保存一次
                        if len(self.data_buffers['prices'][symbol]) >= 50:
                            for kind in ['prices', 'ofi', 'cvd', 'fusion', 'events']:
                                await self._save_data(symbol, kind)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析错误: {e}")
                    except Exception as e:
                        logger.error(f"处理消息错误: {e}")
                        
        except Exception as e:
            logger.error(f"交易流连接错误: {e}")
            raise
    
    async def connect_orderbook_stream(self, symbol: str):
        """连接订单簿流"""
        stream_name = f"{symbol.lower()}@depth5@100ms"
        url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        
        logger.info(f"连接订单簿流: {symbol}")
        
        try:
            async with websockets.connect(url, ping_interval=20) as websocket:
                logger.info(f"订单簿流连接成功: {symbol}")
                
                async for message in websocket:
                    if time.time() > self.end_time:
                        break
                    
                    try:
                        data = json.loads(message)
                        
                        # 处理订单簿数据
                        orderbook_data = {
                            'bids': data.get('bids', []),
                            'asks': data.get('asks', [])
                        }
                        
                        await self._process_orderbook_data(symbol, orderbook_data)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"订单簿JSON解析错误: {e}")
                    except Exception as e:
                        logger.error(f"处理订单簿消息错误: {e}")
                        
        except Exception as e:
            logger.error(f"订单簿流连接错误: {e}")
            # 订单簿流失败不影响交易流
    
    async def run(self):
        """运行完整版采集器"""
        logger.info("开始完整版数据采集...")
        
        # 创建任务
        tasks = []
        for symbol in self.symbols:
            # 交易流任务
            trade_task = asyncio.create_task(self.connect_trade_stream(symbol))
            tasks.append(trade_task)
            
            # 订单簿流任务
            orderbook_task = asyncio.create_task(self.connect_orderbook_stream(symbol))
            tasks.append(orderbook_task)
        
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
                for kind in ['prices', 'ofi', 'cvd', 'fusion', 'events']:
                    if self.data_buffers[kind][symbol]:
                        await self._save_data(symbol, kind)
            
            # 打印统计信息
            logger.info("数据采集完成，统计信息:")
            for symbol in self.symbols:
                logger.info(f"{symbol}: 交易{self.stats['total_trades'][symbol]}, "
                          f"OFI{self.stats['total_ofi'][symbol]}, "
                          f"CVD{self.stats['total_cvd'][symbol]}, "
                          f"事件{self.stats['total_events'][symbol]}")

async def main():
    """主函数"""
    # 从环境变量获取配置
    symbols = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT').split(',')
    run_hours = int(os.getenv('RUN_HOURS', '2'))
    output_dir = os.getenv('OUTPUT_DIR', 'data/ofi_cvd')
    
    # 创建完整版采集器
    harvester = CompleteOFICVDHarvester(symbols, run_hours, output_dir)
    
    # 运行采集
    await harvester.run()

if __name__ == "__main__":
    asyncio.run(main())
