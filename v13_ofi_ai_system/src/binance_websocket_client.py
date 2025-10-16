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
            
        Note:
            消息解析将在Task_1.1.3中实现
        """
        try:
            # 记录接收到的消息（暂不解析，留给下一个任务）
            logger.debug(f"收到消息: {len(message)} bytes")
            
            # 简单验证消息格式
            if not message or not isinstance(message, str):
                logger.warning(f"收到无效消息格式: {type(message)}")
                return
            
            # 在Task_1.1.3中将实现详细的数据解析
            # 这里暂时只记录消息数量
            current_count = len(self.order_book_history)
            if current_count % 100 == 0:  # 每100条消息打印一次
                logger.info(f"已接收 {current_count} 条订单簿数据")
                
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

