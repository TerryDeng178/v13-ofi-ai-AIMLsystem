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
import pandas as pd
import numpy as np
import os
from pathlib import Path
import threading

# 配置日志（基础配置）
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
        # 使用备用域名 binancefuture.com（测试验证可用）
        # 格式: wss://fstream.binancefuture.com/ws/{symbol}@depth{levels}@100ms
        self.ws_url = f"wss://fstream.binancefuture.com/ws/{self.symbol}@depth{depth_levels}@100ms"
        
        # 订单簿历史数据缓存（最多保存10000条）
        self.order_book_history = deque(maxlen=10000)
        
        # WebSocket连接对象（初始化为None）
        self.ws = None
        
        # 数据存储相关
        self.save_interval = 60  # 每60秒保存一次
        self.last_save_time = datetime.now()
        
        # 三层存储目录
        self.data_dir = Path("v13_ofi_ai_system/data/order_book")
        self.ndjson_dir = self.data_dir / "ndjson"  # Layer 1: 原始流
        self.parquet_dir = self.data_dir / "parquet"  # Layer 2: 分析存储
        self.csv_dir = self.data_dir / "csv"  # Legacy: CSV备份
        
        self.ndjson_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        
        # 序列号和时延跟踪
        self.message_seq = 0  # 消息序列号
        self.last_order_book = None  # 上一个订单簿状态（用于增量检测）
        self.last_update_id = None  # 上一个更新ID（pu字段）
        
        # 统计数据
        self.stats = {
            'total_messages': 0,
            'start_time': datetime.now(),
            'latency_list': [],
            'last_print_time': datetime.now(),
            'last_metrics_time': datetime.now(),
            # 序列一致性统计（期货WS严格对齐）
            'gaps': 0,  # 连续区间内的缺口计数（区间内u-U-1的累计）
            'max_gap': 0,  # 单次最大缺口
            'resync': 0,  # resync次数（pu != last_u）
            'reconnects': 0,  # 重连次数
            'last_u': None,  # 上一次的u值（用于pu对齐检查）
        }
        
        # 配置增强日志系统
        self._setup_logging()
        
        logger.info(f"="*60)
        logger.info(f"BinanceOrderBookStream initialized for {symbol.upper()}")
        logger.info(f"WebSocket URL: {self.ws_url}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"="*60)
    
    def __repr__(self):
        """对象的字符串表示"""
        return f"BinanceOrderBookStream(symbol='{self.symbol}', depth_levels={self.depth_levels})"
    
    def __str__(self):
        """对象的可读字符串表示"""
        status = "connected" if self.ws else "not connected"
        history_size = len(self.order_book_history)
        return f"BinanceOrderBookStream({self.symbol.upper()}, {status}, {history_size} records)"
    
    def _setup_logging(self):
        """配置增强的日志系统（控制台+文件）"""
        # 创建日志目录
        log_dir = Path("v13_ofi_ai_system/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件名（按日期）
        log_file = log_dir / f"{self.symbol}_{datetime.now().strftime('%Y%m%d')}.log"
        
        # 配置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        
        logger.info(f"日志系统已配置: {log_file}")
    
    def print_order_book(self, order_book):
        """实时打印订单簿数据（格式化显示）"""
        print()
        print("=" * 80)
        print(f"📊 实时订单簿 - {order_book['symbol']} - Seq: {order_book['seq']}")
        print("=" * 80)
        print(f"⏰ 时间: {order_book['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print(f"📡 时延: {order_book['latency_ms']:.2f}ms")
        print()
        
        # 打印买单
        print("💚 买单（Bids）:")
        print(f"  {'档位':<8} {'价格':>12} {'数量':>15} {'总额':>15}")
        print("-" * 55)
        for i, (price, qty) in enumerate(order_book['bids'], 1):
            total = price * qty
            print(f"  档位{i:<5} {price:>12.2f} {qty:>15.4f} {total:>15.2f}")
        
        print()
        
        # 打印卖单
        print("❤️  卖单（Asks）:")
        print(f"  {'档位':<8} {'价格':>12} {'数量':>15} {'总额':>15}")
        print("-" * 55)
        for i, (price, qty) in enumerate(order_book['asks'], 1):
            total = price * qty
            print(f"  档位{i:<5} {price:>12.2f} {qty:>15.4f} {total:>15.2f}")
        
        # 打印价差
        spread = order_book['asks'][0][0] - order_book['bids'][0][0]
        mid_price = (order_book['asks'][0][0] + order_book['bids'][0][0]) / 2
        spread_bps = (spread / mid_price) * 10000
        
        print()
        print(f"📈 价差: {spread:.2f} USDT ({spread_bps:.2f} bps)")
        print(f"📊 中间价: {mid_price:.2f} USDT")
        print("=" * 80)
        print()
    
    def print_statistics(self):
        """打印统计信息（增强版：包含分位数和序列一致性）"""
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        if elapsed == 0:
            return
        
        rate = self.stats['total_messages'] / elapsed
        avg_latency = sum(self.stats['latency_list']) / len(self.stats['latency_list']) if self.stats['latency_list'] else 0
        
        print()
        print("=" * 80)
        print("📊 运行统计")
        print("=" * 80)
        print(f"⏱️  运行时间: {elapsed:.1f}秒")
        print(f"📨 接收消息: {self.stats['total_messages']} 条")
        print(f"⚡ 接收速率: {rate:.2f} 条/秒")
        print(f"📡 平均时延: {avg_latency:.2f}ms")
        
        # 时延分位数（硬标准2）
        if self.stats['latency_list']:
            percentiles = self.calculate_percentiles()
            print(f"📊 时延分位:")
            print(f"   - P50 (中位数): {percentiles['p50']:.2f}ms")
            print(f"   - P95: {percentiles['p95']:.2f}ms")
            print(f"   - P99: {percentiles['p99']:.2f}ms")
            print(f"📉 最小时延: {min(self.stats['latency_list']):.2f}ms")
            print(f"📈 最大时延: {max(self.stats['latency_list']):.2f}ms")
        
        # 序列一致性统计（硬标准3 - 期货WS严格对齐）
        print(f"🔗 序列一致性 (期货WS严格对齐):")
        print(f"   - Gaps (区间缺口): {self.stats['gaps']} 个updateId")
        print(f"   - Max Gap (最大区间): {self.stats['max_gap']} 个updateId")
        print(f"   - Resync (对齐中断): {self.stats['resync']} 次")
        print(f"   - Reconnects (重连): {self.stats['reconnects']} 次")
        
        print(f"💾 缓存数据: {len(self.order_book_history)} 条")
        print("=" * 80)
        print()
    
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
            
            # 4. 计算接收时延（增强版）
            receive_time = datetime.now()
            ts_recv = receive_time.timestamp() * 1000  # 接收时间戳（毫秒）
            timestamp_ms = data['E']  # 事件时间戳（毫秒）
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
            
            # 计算两种时延
            latency_event_ms = (receive_time - timestamp).total_seconds() * 1000  # 事件时延
            pipeline_start = datetime.now()
            
            # 5. 递增序列号
            self.message_seq += 1
            
            # 6. 提取更新ID字段（U, u）
            U = data.get('U', 0)  # 第一个更新ID
            u = data.get('u', 0)   # 最后一个更新ID
            pu = data.get('pu', None)  # 消息自带的pu（实际不存在，这里用last_u模拟）
            
            # 7. 期货WS严格对齐检测（pu == last_u 连续性）
            if self.stats['last_u'] is not None:
                # 检查连续性：pu应该等于last_u
                if pu is None:
                    # 币安实际不发送pu，需要自己检测 U == last_u + 1
                    pu_expected = self.stats['last_u']
                    if U != pu_expected + 1:
                        # 触发resync
                        self.stats['resync'] += 1
                        logger.warning(f"⚠️ Resync触发! last_u={self.stats['last_u']}, U={U}, gap={U - self.stats['last_u'] - 1}")
            
            # 8. 计算连续区间内的缺口（u - U + 1 = 实际更新数，理想应该连续）
            # 如果区间内有缺口，说明某些updateId被跳过
            interval_updates = u - U + 1  # 区间包含的更新ID数量
            # 实际上每个消息都是聚合的，不一定连续，但我们统计区间内理论缺口
            if interval_updates > 1:
                # 区间内的缺口 = (u - U) - 聚合数 + 1，这里简化为 u - U（因为理想连续时u-U=0）
                interval_gap = u - U  # 区间跨度（0表示单个更新，>0表示有缺口）
                if interval_gap > 0:
                    self.stats['gaps'] += interval_gap
                    self.stats['max_gap'] = max(self.stats['max_gap'], interval_gap)
            
            # 更新last_u和pu
            self.stats['last_u'] = u
            prev_update_id = self.last_update_id
            self.last_update_id = u
            
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
            
            # 8. 计算管道时延
            latency_pipeline_ms = (datetime.now() - pipeline_start).total_seconds() * 1000
            
            # 9. 构建订单簿数据结构（完整版 - 满足NDJSON字段要求）
            order_book = {
                'seq': self.message_seq,  # 序列号
                'timestamp': timestamp,
                'symbol': self.symbol.upper(),
                'bids': bids,
                'asks': asks,
                # 新增完整字段
                'ts_recv': ts_recv,  # 接收时间戳（毫秒）
                'E': timestamp_ms,  # 事件时间（保留原字段名）
                'U': U,  # 第一个更新ID
                'u': u,  # 最后一个更新ID
                'pu': prev_update_id,  # 上一个更新ID（实际是上一条消息的u）
                'latency_event_ms': round(latency_event_ms, 2),  # 事件时延
                'latency_pipeline_ms': round(latency_pipeline_ms, 2),  # 管道时延
                # 保留兼容字段
                'event_time': timestamp_ms,
                'latency_ms': round(latency_event_ms, 2),
                'receive_time': receive_time
            }
            
            # 10. 更新统计数据
            self.stats['total_messages'] += 1
            self.stats['latency_list'].append(latency_event_ms)
            # 只保留最近1000个时延数据（滚动窗口）
            if len(self.stats['latency_list']) > 1000:
                self.stats['latency_list'] = self.stats['latency_list'][-1000:]
            
            # 10. 存储到历史记录
            self.order_book_history.append(order_book)
            
            # 11. 实时写入NDJSON
            self._write_to_ndjson(order_book)
            
            # 12. 定期打印订单簿和保存指标（每10秒一次）
            time_since_print = (datetime.now() - self.stats['last_print_time']).total_seconds()
            if time_since_print >= 10:
                self.print_order_book(order_book)
                self.print_statistics()
                self.save_metrics_json()  # 新增：保存metrics.json
                self.stats['last_print_time'] = datetime.now()
            
            # 13. 日志记录（每100条一次）
            if self.stats['total_messages'] % 100 == 0:
                logger.info(f"已接收 {self.stats['total_messages']} 条订单簿数据, "
                           f"速率: {self.stats['total_messages'] / (datetime.now() - self.stats['start_time']).total_seconds():.2f} 条/秒")
                
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
        
        # 记录重连次数（硬标准3）
        self.stats['reconnects'] += 1
        logger.info(f"重连次数: {self.stats['reconnects']}")
        
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
    
    def calculate_percentiles(self):
        """计算时延分位数（硬标准2：p50/p95/p99）
        
        Returns:
            dict: 包含p50, p95, p99的字典
        """
        if not self.stats['latency_list']:
            return {'p50': 0, 'p95': 0, 'p99': 0}
        
        latencies = np.array(self.stats['latency_list'])
        
        return {
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
    
    def save_metrics_json(self):
        """保存指标到metrics.json文件（硬标准4：周期产物）
        
        每10秒刷新一次，保存当前运行统计和分位数
        """
        try:
            elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
            rate = self.stats['total_messages'] / elapsed if elapsed > 0 else 0
            
            # 计算分位数
            percentiles = self.calculate_percentiles()
            
            # 构建指标数据
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'runtime_seconds': round(elapsed, 2),
                'total_messages': self.stats['total_messages'],
                'message_rate': round(rate, 2),
                'latency': {
                    'avg_ms': round(sum(self.stats['latency_list']) / len(self.stats['latency_list']), 2) if self.stats['latency_list'] else 0,
                    'min_ms': round(min(self.stats['latency_list']), 2) if self.stats['latency_list'] else 0,
                    'max_ms': round(max(self.stats['latency_list']), 2) if self.stats['latency_list'] else 0,
                    'p50_ms': round(percentiles['p50'], 2),
                    'p95_ms': round(percentiles['p95'], 2),
                    'p99_ms': round(percentiles['p99'], 2)
                },
                'sequence_consistency': {
                    'gaps': self.stats['gaps'],
                    'max_gap': self.stats['max_gap'],
                    'resync': self.stats['resync'],
                    'reconnects': self.stats['reconnects']
                },
                'cache_size': len(self.order_book_history),
                'symbol': self.symbol.upper()
            }
            
            # 保存到文件
            metrics_file = self.data_dir / 'metrics.json'
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"指标已保存到 {metrics_file}")
            
        except Exception as e:
            logger.error(f"保存metrics.json失败: {e}", exc_info=True)
    
    def _write_to_ndjson(self, order_book):
        """实时写入NDJSON文件（追加模式）
        
        Args:
            order_book (dict): 订单簿数据
            
        Note:
            NDJSON格式：每行一个JSON对象，便于流式处理和回放
            完整字段：ts_recv, E, U, u, pu, latency_event_ms, latency_pipeline_ms
        """
        try:
            # 生成今天的文件名
            date_str = datetime.now().strftime('%Y%m%d')
            ndjson_file = self.ndjson_dir / f"{self.symbol}_{date_str}.ndjson"
            
            # 准备写入的数据（完整版，包含所有必需字段）
            record = {
                'seq': order_book['seq'],
                'timestamp': order_book['timestamp'].isoformat(),
                'symbol': order_book['symbol'],
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                # 必需字段（硬标准1）
                'ts_recv': order_book['ts_recv'],
                'E': order_book['E'],
                'U': order_book['U'],
                'u': order_book['u'],
                'pu': order_book['pu'],
                'latency_event_ms': order_book['latency_event_ms'],
                'latency_pipeline_ms': order_book['latency_pipeline_ms']
            }
            
            # 追加写入（每行一个JSON）
            with open(ndjson_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"写入NDJSON失败: {e}", exc_info=True)
    
    def convert_ndjson_to_parquet(self, ndjson_file=None):
        """将NDJSON文件转换为Parquet格式
        
        Args:
            ndjson_file (Path): NDJSON文件路径，默认转换今天的文件
            
        Returns:
            str: 生成的Parquet文件路径
            
        Note:
            Parquet列式存储，压缩率高，查询快，适合OFI计算
        """
        try:
            # 如果没有指定文件，使用今天的
            if ndjson_file is None:
                date_str = datetime.now().strftime('%Y%m%d')
                ndjson_file = self.ndjson_dir / f"{self.symbol}_{date_str}.ndjson"
            
            if not ndjson_file.exists():
                logger.warning(f"NDJSON文件不存在: {ndjson_file}")
                return None
            
            # 读取NDJSON文件
            records = []
            with open(ndjson_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        # 展平数据结构
                        flat_record = {
                            'seq': record['seq'],
                            'timestamp': record['timestamp'],
                            'event_time': record['event_time'],
                            'latency_ms': record['latency_ms'],
                            'symbol': record['symbol']
                        }
                        
                        # 添加5档买单
                        for i, (price, qty) in enumerate(record['bids'], 1):
                            flat_record[f'bid_price_{i}'] = price
                            flat_record[f'bid_qty_{i}'] = qty
                        
                        # 添加5档卖单
                        for i, (price, qty) in enumerate(record['asks'], 1):
                            flat_record[f'ask_price_{i}'] = price
                            flat_record[f'ask_qty_{i}'] = qty
                        
                        records.append(flat_record)
            
            if not records:
                logger.warning("NDJSON文件为空")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 生成Parquet文件名
            date_str = datetime.now().strftime('%Y%m%d')
            parquet_file = self.parquet_dir / f"{self.symbol}_{date_str}.parquet"
            
            # 保存为Parquet（使用snappy压缩）
            df.to_parquet(parquet_file, engine='pyarrow', compression='snappy', index=False)
            
            logger.info(f"✅ NDJSON→Parquet转换成功: {parquet_file} ({len(df)} 条记录)")
            return str(parquet_file)
            
        except Exception as e:
            logger.error(f"NDJSON→Parquet转换失败: {e}", exc_info=True)
            return None
    
    def save_to_csv(self, force=False):
        """保存订单簿数据到CSV文件
        
        Args:
            force (bool): 是否强制保存（忽略时间间隔）
            
        Returns:
            str: 保存的文件路径，如果没有保存则返回None
            
        Note:
            默认每60秒自动保存一次，避免频繁IO操作
        """
        # 检查是否需要保存
        if not force:
            time_since_last_save = (datetime.now() - self.last_save_time).total_seconds()
            if time_since_last_save < self.save_interval:
                return None
        
        # 检查是否有数据
        if len(self.order_book_history) == 0:
            logger.warning("没有数据可保存")
            return None
        
        try:
            # 将数据转换为DataFrame格式
            data_list = []
            for ob in self.order_book_history:
                # 展平订单簿数据
                row = {
                    'timestamp': ob['timestamp'],
                    'symbol': ob['symbol'],
                    'event_time': ob['event_time']
                }
                
                # 添加买单数据（5档）
                for i, (price, qty) in enumerate(ob['bids'], 1):
                    row[f'bid_price_{i}'] = price
                    row[f'bid_qty_{i}'] = qty
                
                # 添加卖单数据（5档）
                for i, (price, qty) in enumerate(ob['asks'], 1):
                    row[f'ask_price_{i}'] = price
                    row[f'ask_qty_{i}'] = qty
                
                data_list.append(row)
            
            # 创建DataFrame
            df = pd.DataFrame(data_list)
            
            # 生成文件名（按日期和时间）
            filename = f"{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.data_dir / filename
            
            # 保存到CSV
            df.to_csv(filepath, index=False)
            
            # 更新最后保存时间
            self.last_save_time = datetime.now()
            
            logger.info(f"✅ 数据已保存: {filepath} ({len(df)} 条记录)")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"保存CSV失败: {e}", exc_info=True)
            return None
    
    def auto_save_loop(self):
        """自动保存循环（在后台线程中运行）"""
        while self.ws and self.ws.keep_running:
            try:
                # 每60秒尝试保存一次
                import time
                time.sleep(self.save_interval)
                self.save_to_csv(force=False)
            except Exception as e:
                logger.error(f"自动保存异常: {e}")
                break
    
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

