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
    
    def on_open(self, ws):
        """WebSocket连接成功时的回调
        
        Args:
            ws: WebSocket连接对象
        """
        logger.info(f"✅ WebSocket连接成功: {self.symbol.upper()}")
        logger.info(f"订阅订单簿流: {self.depth_levels}档深度, 100ms更新")
    
    def on_message(self, ws, message):
        """接收到WebSocket消息时的回调
        
        这个方法在每次收到订单簿更新时被调用（约100ms一次）。
        消息格式为JSON字符串，包含时间戳和订单簿数据。
        
        Args:
            ws: WebSocket连接对象
            message (str): 接收到的JSON消息
            
        币安订单簿消息格式:
        {
            "e": "depthUpdate",        // 事件类型
            "E": 1234567890123,        // 事件时间（毫秒）
            "s": "ETHUSDT",           // 交易对
            "U": 123456789,           // 第一个更新ID
            "u": 123456799,           // 最后一个更新ID
            "b": [                    // 买单（bids）
                ["3245.50", "10.5"],  // [价格, 数量]
                ["3245.40", "8.3"],
                ...
            ],
            "a": [                    // 卖单（asks）
                ["3245.60", "11.2"],
                ["3245.70", "9.5"],
                ...
            ]
        }
        """
        try:
            # 1. 验证消息格式
            if not message or not isinstance(message, str):
                logger.warning(f"收到无效消息格式: {type(message)}")
                return
            
            # 2. 解析JSON数据
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}")
                return
            
            # 3. 验证必需字段
            if 'E' not in data or 'b' not in data or 'a' not in data:
                logger.warning(f"消息缺少必需字段: {data.keys()}")
                return
            
            # 4. 提取时间戳（毫秒转datetime）
            timestamp_ms = data['E']
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
            
            # 5. 提取买单（bids）- 5档
            bids = []
            for bid in data['b'][:self.depth_levels]:
                price = float(bid[0])
                quantity = float(bid[1])
                bids.append([price, quantity])
            
            # 6. 提取卖单（asks）- 5档
            asks = []
            for ask in data['a'][:self.depth_levels]:
                price = float(ask[0])
                quantity = float(ask[1])
                asks.append([price, quantity])
            
            # 7. 验证数据完整性
            if len(bids) < self.depth_levels or len(asks) < self.depth_levels:
                logger.warning(f"订单簿深度不足: bids={len(bids)}, asks={len(asks)}")
                return
            
            # 8. 构建订单簿数据结构
            order_book = {
                'timestamp': timestamp,
                'symbol': self.symbol.upper(),
                'bids': bids,
                'asks': asks,
                'event_time': timestamp_ms
            }
            
            # 9. 存储到历史记录
            self.order_book_history.append(order_book)
            
            # 10. 定期打印统计信息
            current_count = len(self.order_book_history)
            if current_count % 100 == 0:  # 每100条打印一次
                logger.info(f"已接收 {current_count} 条订单簿数据")
                logger.debug(f"最新数据 - Bid1: {bids[0][0]:.2f}@{bids[0][1]:.4f}, "
                           f"Ask1: {asks[0][0]:.2f}@{asks[0][1]:.4f}, "
                           f"Spread: {(asks[0][0] - bids[0][0]):.2f}")
                
        except Exception as e:
            logger.error(f"处理消息时出错: {e}", exc_info=True)
    
    def on_error(self, ws, error):
        """WebSocket连接出错时的回调
        
        Args:
            ws: WebSocket连接对象
            error: 错误对象或错误消息
        """
        logger.error(f"❌ WebSocket错误: {error}")
        
        # 根据错误类型进行分类处理
        error_str = str(error)
        if "Connection refused" in error_str:
            logger.error("连接被拒绝，请检查网络或URL")
        elif "timeout" in error_str.lower():
            logger.error("连接超时，可能网络不稳定")
        elif "SSL" in error_str or "Certificate" in error_str:
            logger.error("SSL证书错误")
        else:
            logger.error(f"未知错误类型: {error_str}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket连接关闭时的回调
        
        Args:
            ws: WebSocket连接对象
            close_status_code: 关闭状态码
            close_msg: 关闭消息
        """
        logger.warning(f"⚠️ WebSocket连接已关闭")
        logger.warning(f"状态码: {close_status_code}")
        logger.warning(f"关闭消息: {close_msg}")
        
        # 记录连接统计
        total_records = len(self.order_book_history)
        logger.info(f"本次会话共接收 {total_records} 条订单簿数据")
    
    def get_latest_order_book(self):
        """获取最新的订单簿数据
        
        Returns:
            dict: 最新的订单簿数据，如果没有数据则返回None
            
        Example:
            >>> client = BinanceOrderBookStream()
            >>> # ... 运行一段时间后 ...
            >>> latest = client.get_latest_order_book()
            >>> if latest:
            ...     print(f"Bid1: {latest['bids'][0]}")
            ...     print(f"Ask1: {latest['asks'][0]}")
        """
        if len(self.order_book_history) == 0:
            return None
        return self.order_book_history[-1]
    
    def get_order_book_count(self):
        """获取已接收的订单簿数据总数
        
        Returns:
            int: 订单簿数据数量
        """
        return len(self.order_book_history)
    
    def run(self, reconnect=True):
        """启动WebSocket连接
        
        这个方法会建立到币安WebSocket的连接，并开始接收订单簿数据。
        连接是阻塞的，会一直运行直到手动停止或发生错误。
        
        Args:
            reconnect (bool): 是否在断线后自动重连，默认True
            
        Example:
            >>> client = BinanceOrderBookStream('ethusdt')
            >>> client.run()  # 开始接收数据，阻塞运行
        """
        logger.info("=" * 60)
        logger.info(f"启动币安WebSocket客户端")
        logger.info(f"交易对: {self.symbol.upper()}")
        logger.info(f"订单簿深度: {self.depth_levels}档")
        logger.info(f"WebSocket URL: {self.ws_url}")
        logger.info(f"自动重连: {'开启' if reconnect else '关闭'}")
        logger.info("=" * 60)
        
        # 创建WebSocket连接
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # 启动连接（阻塞运行）
        try:
            self.ws.run_forever(
                reconnect=5 if reconnect else 0  # 重连间隔5秒
            )
        except KeyboardInterrupt:
            logger.info("用户中断，正在关闭连接...")
            self.ws.close()
        except Exception as e:
            logger.error(f"运行时发生异常: {e}", exc_info=True)
            raise

