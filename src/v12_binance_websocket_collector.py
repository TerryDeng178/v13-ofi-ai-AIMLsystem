"""
V12 币安WebSocket订单簿数据收集器
基于V9 OFI策略参数，收集真实订单簿数据进行OFI计算
"""

import websocket
import json
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
from collections import deque
import queue

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V12BinanceWebSocketCollector:
    """
    V12 币安WebSocket订单簿数据收集器
    基于V9 OFI策略参数设计
    """
    
    def __init__(self, symbol="ETHUSDT", depth_levels=5):
        """
        初始化WebSocket收集器
        
        Args:
            symbol: 交易对符号
            depth_levels: 订单簿深度级别 (基于V9: 5档)
        """
        self.symbol = symbol.lower()
        self.depth_levels = depth_levels
        
        # WebSocket配置
        self.ws_url = f"wss://fstream.binance.com/ws/{self.symbol}@depth20@100ms"
        self.ws = None
        self.is_connected = False
        self.is_running = False
        
        # 数据存储 (基于V9参数优化)
        self.order_book_data = deque(maxlen=10000)  # 存储最近10000条数据
        self.ofi_data = deque(maxlen=7200)         # 存储最近2小时OFI数据 (7200秒)
        self.cvd_data = deque(maxlen=7200)         # 存储最近2小时CVD数据
        
        # V9 OFI参数
        self.ofi_window_seconds = 2               # V9: 2秒滚动窗口
        self.z_window = 1200                      # V9: 20分钟Z-score窗口
        self.ofi_levels = 5                       # V9: 5档深度
        
        # 数据回调函数
        self.data_callbacks = []
        
        # 线程安全队列
        self.data_queue = queue.Queue(maxsize=1000)
        
        # 统计信息
        self.stats = {
            'messages_received': 0,
            'ofis_calculated': 0,
            'cvds_calculated': 0,
            'connection_errors': 0,
            'last_update': None
        }
        
        logger.info(f"V12 WebSocket收集器初始化完成 - 交易对: {self.symbol}, 深度: {self.depth_levels}")
    
    def add_data_callback(self, callback: Callable):
        """添加数据回调函数"""
        self.data_callbacks.append(callback)
        logger.info(f"添加数据回调函数: {callback.__name__}")
    
    def start_collection(self):
        """开始数据收集"""
        if self.is_running:
            logger.warning("数据收集已在运行中")
            return
        
        self.is_running = True
        
        # 启动WebSocket连接
        self._connect_websocket()
        
        # 启动数据处理线程
        self._start_data_processing()
        
        logger.info("V12 WebSocket数据收集已启动")
    
    def stop_collection(self):
        """停止数据收集"""
        self.is_running = False
        
        if self.ws:
            self.ws.close()
        
        logger.info("V12 WebSocket数据收集已停止")
    
    def _connect_websocket(self):
        """连接WebSocket"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # 在新线程中运行WebSocket
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
        except Exception as e:
            logger.error(f"WebSocket连接失败: {e}")
            self.stats['connection_errors'] += 1
    
    def _on_open(self, ws):
        """WebSocket连接打开"""
        self.is_connected = True
        logger.info(f"WebSocket连接已建立 - {self.symbol} 订单簿数据流")
    
    def _on_message(self, ws, message):
        """处理WebSocket消息"""
        try:
            data = json.loads(message)
            
            # 更新统计信息
            self.stats['messages_received'] += 1
            self.stats['last_update'] = datetime.now()
            
            # 将数据放入队列
            if not self.data_queue.full():
                self.data_queue.put(data)
            else:
                logger.warning("数据队列已满，丢弃消息")
                
        except Exception as e:
            logger.error(f"处理WebSocket消息失败: {e}")
    
    def _on_error(self, ws, error):
        """WebSocket错误处理"""
        logger.error(f"WebSocket错误: {error}")
        self.stats['connection_errors'] += 1
        self.is_connected = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket连接关闭"""
        logger.info(f"WebSocket连接关闭: {close_status_code} - {close_msg}")
        self.is_connected = False
    
    def _start_data_processing(self):
        """启动数据处理线程"""
        def process_data():
            while self.is_running:
                try:
                    # 从队列获取数据
                    if not self.data_queue.empty():
                        data = self.data_queue.get(timeout=1)
                        
                        # 处理订单簿数据
                        processed_data = self._process_order_book_data(data)
                        
                        if processed_data:
                            # 计算OFI和CVD
                            ofi_value = self._calculate_ofi(processed_data)
                            cvd_value = self._calculate_cvd(processed_data)
                            
                            # 存储数据
                            self._store_data(processed_data, ofi_value, cvd_value)
                            
                            # 调用回调函数
                            self._notify_callbacks(processed_data, ofi_value, cvd_value)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"数据处理错误: {e}")
                    time.sleep(0.1)
        
        # 启动数据处理线程
        processing_thread = threading.Thread(target=process_data)
        processing_thread.daemon = True
        processing_thread.start()
    
    def _process_order_book_data(self, data):
        """处理订单簿数据"""
        try:
            # 提取订单簿数据
            timestamp = datetime.now()
            
            # 解析买卖盘数据
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            if not bids or not asks:
                return None
            
            # 构建订单簿数据结构
            order_book = {
                'timestamp': timestamp,
                'symbol': self.symbol
            }
            
            # 提取前N档数据 (基于V9: 5档)
            for i in range(min(self.depth_levels, len(bids))):
                order_book[f'bid{i+1}_price'] = float(bids[i][0])
                order_book[f'bid{i+1}_size'] = float(bids[i][1])
            
            for i in range(min(self.depth_levels, len(asks))):
                order_book[f'ask{i+1}_price'] = float(asks[i][0])
                order_book[f'ask{i+1}_size'] = float(asks[i][1])
            
            # 计算中间价和价差
            order_book['mid_price'] = (order_book['bid1_price'] + order_book['ask1_price']) / 2
            order_book['spread'] = order_book['ask1_price'] - order_book['bid1_price']
            order_book['spread_bps'] = (order_book['spread'] / order_book['mid_price']) * 10000
            
            return order_book
            
        except Exception as e:
            logger.error(f"处理订单簿数据失败: {e}")
            return None
    
    def _calculate_ofi(self, order_book_data):
        """计算OFI (基于V9参数)"""
        try:
            # 如果数据不足，返回0
            if len(self.order_book_data) < 2:
                return 0.0
            
            # 获取当前和前一时刻的数据
            current_data = order_book_data
            prev_data = self.order_book_data[-1] if self.order_book_data else None
            
            if not prev_data:
                return 0.0
            
            ofi_total = 0.0
            
            # 基于V9的5档加权计算
            weights = [1.0, 0.5, 0.33, 0.25, 0.2]  # V9权重
            
            for level in range(self.ofi_levels):
                weight = weights[level]
                
                # 获取当前档位数据
                bid_price_key = f'bid{level+1}_price'
                ask_price_key = f'ask{level+1}_price'
                bid_size_key = f'bid{level+1}_size'
                ask_size_key = f'ask{level+1}_size'
                
                if all(key in current_data and key in prev_data for key in [bid_price_key, ask_price_key, bid_size_key, ask_size_key]):
                    # 检查价格改进
                    bid_improved = current_data[bid_price_key] > prev_data[bid_price_key]
                    ask_improved = current_data[ask_price_key] > prev_data[ask_price_key]
                    
                    # 计算数量变化
                    bid_delta = current_data[bid_size_key] - prev_data[bid_size_key]
                    ask_delta = current_data[ask_size_key] - prev_data[ask_size_key]
                    
                    # OFI贡献
                    ofi_contribution = weight * (
                        bid_delta * bid_improved - ask_delta * ask_improved
                    )
                    ofi_total += ofi_contribution
            
            self.stats['ofis_calculated'] += 1
            return ofi_total
            
        except Exception as e:
            logger.error(f"计算OFI失败: {e}")
            return 0.0
    
    def _calculate_cvd(self, order_book_data):
        """计算CVD (累积成交量差值)"""
        try:
            # 简化的CVD计算 (基于订单簿变化)
            # 在实际应用中，需要交易数据来计算真实的CVD
            
            if len(self.order_book_data) < 2:
                return 0.0
            
            current_data = order_book_data
            prev_data = self.order_book_data[-1]
            
            # 基于买卖盘变化估算CVD
            bid_size_change = sum([
                current_data.get(f'bid{i+1}_size', 0) - prev_data.get(f'bid{i+1}_size', 0)
                for i in range(self.ofi_levels)
            ])
            
            ask_size_change = sum([
                current_data.get(f'ask{i+1}_size', 0) - prev_data.get(f'ask{i+1}_size', 0)
                for i in range(self.ofi_levels)
            ])
            
            cvd_change = bid_size_change - ask_size_change
            
            # 累积CVD
            prev_cvd = self.cvd_data[-1] if self.cvd_data else 0.0
            cvd_total = prev_cvd + cvd_change
            
            self.stats['cvds_calculated'] += 1
            return cvd_total
            
        except Exception as e:
            logger.error(f"计算CVD失败: {e}")
            return 0.0
    
    def _store_data(self, order_book_data, ofi_value, cvd_value):
        """存储数据"""
        # 存储订单簿数据
        self.order_book_data.append(order_book_data)
        
        # 存储OFI数据
        ofi_record = {
            'timestamp': order_book_data['timestamp'],
            'ofi': ofi_value,
            'mid_price': order_book_data['mid_price'],
            'spread_bps': order_book_data['spread_bps']
        }
        self.ofi_data.append(ofi_record)
        
        # 存储CVD数据
        cvd_record = {
            'timestamp': order_book_data['timestamp'],
            'cvd': cvd_value,
            'mid_price': order_book_data['mid_price']
        }
        self.cvd_data.append(cvd_record)
    
    def _notify_callbacks(self, order_book_data, ofi_value, cvd_value):
        """通知回调函数"""
        for callback in self.data_callbacks:
            try:
                callback(order_book_data, ofi_value, cvd_value)
            except Exception as e:
                logger.error(f"回调函数执行失败: {e}")
    
    def get_latest_ofi_data(self, window_seconds=None):
        """获取最新的OFI数据"""
        if not window_seconds:
            window_seconds = self.ofi_window_seconds
        
        if not self.ofi_data:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(list(self.ofi_data))
        
        if df.empty:
            return df
        
        # 按时间窗口过滤
        cutoff_time = df['timestamp'].max() - timedelta(seconds=window_seconds)
        recent_data = df[df['timestamp'] >= cutoff_time]
        
        return recent_data
    
    def get_latest_cvd_data(self, window_seconds=None):
        """获取最新的CVD数据"""
        if not window_seconds:
            window_seconds = self.ofi_window_seconds
        
        if not self.cvd_data:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(list(self.cvd_data))
        
        if df.empty:
            return df
        
        # 按时间窗口过滤
        cutoff_time = df['timestamp'].max() - timedelta(seconds=window_seconds)
        recent_data = df[df['timestamp'] >= cutoff_time]
        
        return recent_data
    
    def calculate_ofi_zscore(self, window_seconds=None):
        """计算OFI Z-score (基于V9参数)"""
        if not window_seconds:
            window_seconds = self.z_window
        
        if not self.ofi_data:
            return 0.0
        
        # 获取历史数据
        df = pd.DataFrame(list(self.ofi_data))
        
        if len(df) < 10:  # 数据不足
            return 0.0
        
        # 按时间窗口过滤
        cutoff_time = df['timestamp'].max() - timedelta(seconds=window_seconds)
        window_data = df[df['timestamp'] >= cutoff_time]
        
        if len(window_data) < 5:
            return 0.0
        
        # 计算Z-score
        latest_ofi = df['ofi'].iloc[-1]
        window_mean = window_data['ofi'].mean()
        window_std = window_data['ofi'].std()
        
        if window_std == 0:
            return 0.0
        
        z_score = (latest_ofi - window_mean) / window_std
        return z_score
    
    def calculate_cvd_zscore(self, window_seconds=None):
        """计算CVD Z-score (基于V9参数)"""
        if not window_seconds:
            window_seconds = self.z_window
        
        if not self.cvd_data:
            return 0.0
        
        # 获取历史数据
        df = pd.DataFrame(list(self.cvd_data))
        
        if len(df) < 10:
            return 0.0
        
        # 按时间窗口过滤
        cutoff_time = df['timestamp'].max() - timedelta(seconds=window_seconds)
        window_data = df[df['timestamp'] >= cutoff_time]
        
        if len(window_data) < 5:
            return 0.0
        
        # 计算Z-score
        latest_cvd = df['cvd'].iloc[-1]
        window_mean = window_data['cvd'].mean()
        window_std = window_data['cvd'].std()
        
        if window_std == 0:
            return 0.0
        
        z_score = (latest_cvd - window_mean) / window_std
        return z_score
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'messages_received': self.stats['messages_received'],
            'ofis_calculated': self.stats['ofis_calculated'],
            'cvds_calculated': self.stats['cvds_calculated'],
            'connection_errors': self.stats['connection_errors'],
            'last_update': self.stats['last_update'],
            'order_book_data_count': len(self.order_book_data),
            'ofi_data_count': len(self.ofi_data),
            'cvd_data_count': len(self.cvd_data)
        }
    
    def export_data(self, filepath_prefix="v12_websocket_data"):
        """导出数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 导出订单簿数据
        if self.order_book_data:
            order_book_df = pd.DataFrame(list(self.order_book_data))
            order_book_file = f"{filepath_prefix}_orderbook_{timestamp}.csv"
            order_book_df.to_csv(order_book_file, index=False)
            logger.info(f"订单簿数据已导出: {order_book_file}")
        
        # 导出OFI数据
        if self.ofi_data:
            ofi_df = pd.DataFrame(list(self.ofi_data))
            ofi_file = f"{filepath_prefix}_ofi_{timestamp}.csv"
            ofi_df.to_csv(ofi_file, index=False)
            logger.info(f"OFI数据已导出: {ofi_file}")
        
        # 导出CVD数据
        if self.cvd_data:
            cvd_df = pd.DataFrame(list(self.cvd_data))
            cvd_file = f"{filepath_prefix}_cvd_{timestamp}.csv"
            cvd_df.to_csv(cvd_file, index=False)
            logger.info(f"CVD数据已导出: {cvd_file}")


def test_v12_websocket_collector():
    """测试V12 WebSocket收集器"""
    logger.info("开始测试V12 WebSocket收集器...")
    
    # 创建收集器
    collector = V12BinanceWebSocketCollector(symbol="ETHUSDT", depth_levels=5)
    
    # 添加数据回调
    def data_callback(order_book_data, ofi_value, cvd_value):
        logger.info(f"收到数据 - 时间: {order_book_data['timestamp']}, OFI: {ofi_value:.4f}, CVD: {cvd_value:.4f}")
    
    collector.add_data_callback(data_callback)
    
    # 开始收集
    collector.start_collection()
    
    try:
        # 运行30秒
        time.sleep(30)
        
        # 获取统计信息
        stats = collector.get_statistics()
        logger.info(f"收集统计: {stats}")
        
        # 计算Z-score
        ofi_z = collector.calculate_ofi_zscore()
        cvd_z = collector.calculate_cvd_zscore()
        logger.info(f"OFI Z-score: {ofi_z:.4f}, CVD Z-score: {cvd_z:.4f}")
        
        # 导出数据
        collector.export_data()
        
    finally:
        # 停止收集
        collector.stop_collection()
    
    logger.info("V12 WebSocket收集器测试完成")


if __name__ == "__main__":
    test_v12_websocket_collector()
