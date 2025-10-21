# -*- coding: utf-8 -*-
"""
Binance WebSocket Adapter for OFI+CVD Data Harvesting
简化版WebSocket适配器，用于数据采集
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, Any, Callable, Optional

class BinanceWebSocketAdapter:
    """Binance WebSocket适配器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connections = {}
        
    async def subscribe_trades(self, symbol: str, on_trade: Callable, on_reconnect: Callable = None, 
                             ping_interval: int = 20, heartbeat_timeout: int = 30, 
                             reconnect_delay: float = 1.0, max_reconnect_attempts: int = 10):
        """订阅交易数据流"""
        stream_name = f"{symbol.lower()}@trade"
        url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        
        self.logger.info(f"连接WebSocket: {url}")
        
        try:
            async with websockets.connect(url, ping_interval=ping_interval) as websocket:
                self.connections[symbol] = websocket
                self.logger.info(f"WebSocket连接成功: {symbol}")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        # 转换数据格式
                        trade_data = {
                            'event_ts_ms': data.get('E', 0),
                            'ts_ms': data.get('T', 0),
                            'symbol': data.get('s', symbol),
                            'price': float(data.get('p', 0)),
                            'qty': float(data.get('q', 0)),
                            'agg_trade_id': data.get('a', 0),
                            'recv_ts_ms': int(asyncio.get_event_loop().time() * 1000)
                        }
                        
                        # 调用回调函数
                        on_trade(symbol, trade_data)
                        
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON解析错误: {e}")
                    except Exception as e:
                        self.logger.error(f"处理消息错误: {e}")
                        
        except Exception as e:
            self.logger.error(f"WebSocket连接错误: {e}")
            if on_reconnect:
                on_reconnect(symbol)
            raise

