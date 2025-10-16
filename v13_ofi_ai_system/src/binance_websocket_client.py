#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""币安WebSocket客户端 - 接收真实订单簿数据

这个模块实现了币安期货WebSocket客户端，用于接收实时的订单簿数据。
支持5档订单簿深度，更新频率100ms。

Author: V13 OFI+AI System
Created: 2025-01-17
"""

import websocket
import json
from datetime import datetime
from collections import deque
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceOrderBookStream:
    """币安订单簿WebSocket流客户端
    
    这个类负责连接币安期货WebSocket，接收实时的订单簿数据。
    订单簿包含5档买单和5档卖单，更新频率为100ms。
    
    Attributes:
        symbol (str): 交易对符号，如'ethusdt'
        depth_levels (int): 订单簿深度档位，默认5档
        ws_url (str): WebSocket连接URL
        order_book_history (deque): 历史订单簿数据缓存
        ws (WebSocketApp): WebSocket连接对象
    """
    
    def __init__(self, symbol='ethusdt', depth_levels=5):
        """初始化币安WebSocket客户端
        
        Args:
            symbol (str): 交易对符号，默认'ethusdt'（ETHUSDT永续合约）
            depth_levels (int): 订单簿深度档位，默认5档
            
        Example:
            >>> client = BinanceOrderBookStream('ethusdt', 5)
            >>> # 客户端已初始化，准备连接
        """
        self.symbol = symbol.lower()
        self.depth_levels = depth_levels
        
        # 构建WebSocket URL
        # 格式: wss://fstream.binance.com/ws/{symbol}@depth{levels}@100ms
        self.ws_url = f"wss://fstream.binance.com/ws/{self.symbol}@depth{depth_levels}@100ms"
        
        # 订单簿历史数据缓存（最多保存10000条）
        self.order_book_history = deque(maxlen=10000)
        
        # WebSocket连接对象（初始化为None）
        self.ws = None
        
        logger.info(f"BinanceOrderBookStream initialized for {symbol.upper()}")
        logger.info(f"WebSocket URL: {self.ws_url}")
    
    def __repr__(self):
        """对象的字符串表示"""
        return f"BinanceOrderBookStream(symbol='{self.symbol}', depth_levels={self.depth_levels})"
    
    def __str__(self):
        """对象的可读字符串表示"""
        status = "connected" if self.ws else "not connected"
        history_size = len(self.order_book_history)
        return f"BinanceOrderBookStream({self.symbol.upper()}, {status}, {history_size} records)"

