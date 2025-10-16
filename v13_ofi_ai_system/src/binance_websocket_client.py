#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""甯佸畨WebSocket瀹㈡埛绔?- 鎺ユ敹鐪熷疄璁㈠崟绨挎暟鎹?

杩欎釜妯″潡瀹炵幇浜嗗竵瀹夋湡璐ebSocket瀹㈡埛绔紝鐢ㄤ簬鎺ユ敹瀹炴椂鐨勮鍗曠翱鏁版嵁銆?
鏀寔5妗ｈ鍗曠翱娣卞害锛屾洿鏂伴鐜?00ms銆?

Task 1.1.6 鏂板:
- 寮傛鏃ュ織锛圦ueueHandler/Listener锛岄潪闃诲锛?
- 鏃ュ織杞浆涓庝繚鐣欙紙鏃堕棿/澶у皬涓ょ妯″紡锛?
- log_queue鐩戞帶鎸囨爣
- 鍛戒护琛屽弬鏁版敮鎸?

Author: V13 OFI+AI System
Created: 2025-01-17
Updated: 2025-10-17 (Task 1.1.6)
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
import argparse
import sys
import time

# 瀵煎叆寮傛鏃ュ織宸ュ叿
try:
    from utils.async_logging import setup_async_logging, sample_queue_metrics
except ImportError:
    # 鍏煎涓嶅悓杩愯璺緞
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.async_logging import setup_async_logging, sample_queue_metrics

# 閰嶇疆鏃ュ織锛堝皢鍦ㄧ被鍒濆鍖栨椂閲嶆柊閰嶇疆涓哄紓姝ユ棩蹇楋級
logger = logging.getLogger(__name__)


class BinanceOrderBookStream:
    """甯佸畨璁㈠崟绨縒ebSocket娴佸鎴风
    
    杩欎釜绫昏礋璐ｈ繛鎺ュ竵瀹夋湡璐ebSocket锛屾帴鏀跺疄鏃剁殑璁㈠崟绨挎暟鎹€?
    璁㈠崟绨垮寘鍚?妗ｄ拱鍗曞拰5妗ｅ崠鍗曪紝鏇存柊棰戠巼涓?00ms銆?
    
    Attributes:
        symbol (str): 浜ゆ槗瀵圭鍙凤紝濡?ethusdt'
        depth_levels (int): 璁㈠崟绨挎繁搴︽。浣嶏紝榛樿5妗?
        ws_url (str): WebSocket杩炴帴URL
        order_book_history (deque): 鍘嗗彶璁㈠崟绨挎暟鎹紦瀛?
        ws (WebSocketApp): WebSocket杩炴帴瀵硅薄
    """
    
    def __init__(self, symbol='ethusdt', depth_levels=5, 
                 rotate='interval', rotate_sec=60, max_bytes=5_000_000, backups=7,
                 print_interval=10):
        """鍒濆鍖栧竵瀹塛ebSocket瀹㈡埛绔?
        
        Args:
            symbol (str): 浜ゆ槗瀵圭鍙凤紝榛樿'ethusdt'锛圗THUSDT姘哥画鍚堢害锛?
            depth_levels (int): 璁㈠崟绨挎繁搴︽。浣嶏紝榛樿5妗?
            rotate (str): 鏃ュ織杞浆妯″紡 ('interval' 鎴?'size')
            rotate_sec (int): 鏃堕棿杞浆闂撮殧锛堢锛夛紝榛樿60绉?
            max_bytes (int): 澶у皬杞浆闃堝€硷紙瀛楄妭锛夛紝榛樿5MB
            backups (int): 淇濈暀澶囦唤鏁伴噺锛岄粯璁?涓?
            print_interval (int): 鎵撳嵃闂撮殧锛堢锛夛紝榛樿10绉?
            
        Example:
            >>> client = BinanceOrderBookStream('ethusdt', 5)
            >>> # 瀹㈡埛绔凡鍒濆鍖栵紝鍑嗗杩炴帴
        """
        self.symbol = symbol.lower()
        self.depth_levels = depth_levels
        self.rotate = rotate
        self.rotate_sec = rotate_sec
        self.max_bytes = max_bytes
        self.backups = backups
        self.print_interval = print_interval
        
        # 鏋勫缓WebSocket URL
        # 浣跨敤澶囩敤鍩熷悕 binancefuture.com锛堟祴璇曢獙璇佸彲鐢級
        # 鏍煎紡: wss://fstream.binancefuture.com/ws/{symbol}@depth{levels}@100ms
        self.ws_url = f"wss://fstream.binancefuture.com/ws/{self.symbol}@depth{depth_levels}@100ms"
        
        # 璁㈠崟绨垮巻鍙叉暟鎹紦瀛橈紙鏈€澶氫繚瀛?0000鏉★級
        self.order_book_history = deque(maxlen=10000)
        
        # WebSocket杩炴帴瀵硅薄锛堝垵濮嬪寲涓篘one锛?
        self.ws = None
        
        # 鏁版嵁瀛樺偍鐩稿叧
        self.save_interval = 60  # 姣?0绉掍繚瀛樹竴娆?
        self.last_save_time = datetime.now()
        
        # 涓夊眰瀛樺偍鐩綍
        self.data_dir = Path("v13_ofi_ai_system/data/order_book")
        self.ndjson_dir = self.data_dir / "ndjson"  # Layer 1: 鍘熷娴?
        self.parquet_dir = self.data_dir / "parquet"  # Layer 2: 鍒嗘瀽瀛樺偍
        self.csv_dir = self.data_dir / "csv"  # Legacy: CSV澶囦唤
        
        self.ndjson_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        
        # 搴忓垪鍙峰拰鏃跺欢璺熻釜
        self.message_seq = 0  # 娑堟伅搴忓垪鍙?
        self.last_order_book = None  # 涓婁竴涓鍗曠翱鐘舵€侊紙鐢ㄤ簬澧為噺妫€娴嬶級
        self.last_update_id = None  # 涓婁竴涓洿鏂癐D锛坧u瀛楁锛?
        
        # 缁熻鏁版嵁
        self.stats = {
            'total_messages': 0,
            'start_time': datetime.now(),
            'latency_list': [],
            'last_print_time': datetime.now(),
            'last_metrics_time': datetime.now(),
            # 搴忓垪涓€鑷存€х粺璁★紙鏈熻揣WS涓ユ牸瀵归綈 - 淇鐗?锛?
            'resyncs': 0,  # resync娆℃暟锛堜繚鐣欏瓧娈碉紝瀹為檯濮嬬粓涓?锛?
            'reconnects': 0,  # 閲嶈繛娆℃暟
            'batch_span_sum': 0,  # batch_span绱锛堜粎瑙傛祴锛?
            'batch_span_max': 0,  # 鏈€澶atch_span
            'batch_span_list': deque(maxlen=1000),  # 鐢ㄤ簬璁＄畻P95
            'last_u': None,  # 涓婁竴娆＄殑u鍊硷紙鐢ㄤ簬pu瀛楁璁板綍锛?
            # log_queue鎸囨爣
            'log_queue_depth_list': deque(maxlen=1000),  # 闃熷垪娣卞害鍘嗗彶
            'log_queue_max_depth': 0,  # 闃熷垪鏈€澶ф繁搴?
            'log_drops': 0,  # 鏃ュ織涓㈠純鏁?
        }
        
        # 閰嶇疆寮傛鏃ュ織绯荤粺锛圱ask 1.1.6锛?
        self.logger, self.listener, self.queue_handler = self._setup_logging()
        
        self.logger.info(f"="*60)
        self.logger.info(f"BinanceOrderBookStream initialized for {symbol.upper()}")
        self.logger.info(f"WebSocket URL: {self.ws_url}")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"="*60)
    
    def __repr__(self):
        """瀵硅薄鐨勫瓧绗︿覆琛ㄧず"""
        return f"BinanceOrderBookStream(symbol='{self.symbol}', depth_levels={self.depth_levels})"
    
    def __str__(self):
        """瀵硅薄鐨勫彲璇诲瓧绗︿覆琛ㄧず"""
        status = "connected" if self.ws else "not connected"
        history_size = len(self.order_book_history)
        return f"BinanceOrderBookStream({self.symbol.upper()}, {status}, {history_size} records)"
    
    def _setup_logging(self):
        """閰嶇疆寮傛鏃ュ織绯荤粺锛圦ueueHandler + 杞浆淇濈暀锛?
        
        Task 1.1.6: 浣跨敤QueueHandler/Listener瀹炵幇闈為樆濉炴棩蹇?
        """
        # 鍒涘缓鏃ュ織鐩綍
        log_dir = Path("v13_ofi_ai_system/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 鏃ュ織鏂囦欢鍚嶏紙鎸夋棩鏈?+ 杞浆锛?
        log_file = log_dir / f"{self.symbol}_{datetime.now().strftime('%Y%m%d')}.log"
        
        # 浣跨敤寮傛鏃ュ織宸ュ叿璁剧疆
        logger_instance, listener, queue_handler = setup_async_logging(
            name=__name__,
            log_path=str(log_file),
            rotate=self.rotate,
            rotate_sec=self.rotate_sec,
            max_bytes=self.max_bytes,
            backups=self.backups,
            level=logging.INFO,
            queue_max=10000,
            to_console=False  # 鎺у埗鍙拌緭鍑虹敱print_statistics鎺у埗
        )
        
        logger_instance.info(f"寮傛鏃ュ織绯荤粺宸查厤缃? {log_file}")
        logger_instance.info(f"杞浆妯″紡: {self.rotate}, " + 
                           (f"闂撮殧={self.rotate_sec}s" if self.rotate=='interval' else f"澶у皬={self.max_bytes}瀛楄妭"))
        logger_instance.info(f"淇濈暀澶囦唤鏁? {self.backups}")
        
        return logger_instance, listener, queue_handler
    
    def print_order_book(self, order_book):
        """瀹炴椂鎵撳嵃璁㈠崟绨挎暟鎹紙鏍煎紡鍖栨樉绀猴級"""
        print()
        print("=" * 80)
        print(f"馃搳 瀹炴椂璁㈠崟绨?- {order_book['symbol']} - Seq: {order_book['seq']}")
        print("=" * 80)
        print(f"鈴?鏃堕棿: {order_book['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print(f"馃摗 鏃跺欢: {order_book['latency_ms']:.2f}ms")
        print()
        
        # 鎵撳嵃涔板崟
        print("馃挌 涔板崟锛圔ids锛?")
        print(f"  {'妗ｄ綅':<8} {'浠锋牸':>12} {'鏁伴噺':>15} {'鎬婚':>15}")
        print("-" * 55)
        for i, (price, qty) in enumerate(order_book['bids'], 1):
            total = price * qty
            print(f"  妗ｄ綅{i:<5} {price:>12.2f} {qty:>15.4f} {total:>15.2f}")
        
        print()
        
        # 鎵撳嵃鍗栧崟
        print("鉂わ笍  鍗栧崟锛圓sks锛?")
        print(f"  {'妗ｄ綅':<8} {'浠锋牸':>12} {'鏁伴噺':>15} {'鎬婚':>15}")
        print("-" * 55)
        for i, (price, qty) in enumerate(order_book['asks'], 1):
            total = price * qty
            print(f"  妗ｄ綅{i:<5} {price:>12.2f} {qty:>15.4f} {total:>15.2f}")
        
        # 鎵撳嵃浠峰樊
        spread = order_book['asks'][0][0] - order_book['bids'][0][0]
        mid_price = (order_book['asks'][0][0] + order_book['bids'][0][0]) / 2
        spread_bps = (spread / mid_price) * 10000
        
        print()
        print(f"馃搱 浠峰樊: {spread:.2f} USDT ({spread_bps:.2f} bps)")
        print(f"馃搳 涓棿浠? {mid_price:.2f} USDT")
        print("=" * 80)
        print()
    
    def print_statistics(self):
        """鎵撳嵃缁熻淇℃伅锛堝寮虹増锛氬寘鍚垎浣嶆暟鍜屽簭鍒椾竴鑷存€э級"""
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        if elapsed == 0:
            return
        
        rate = self.stats['total_messages'] / elapsed
        avg_latency = sum(self.stats['latency_list']) / len(self.stats['latency_list']) if self.stats['latency_list'] else 0
        
        print()
        print("=" * 80)
        print("馃搳 杩愯缁熻")
        print("=" * 80)
        print(f"鈴憋笍  杩愯鏃堕棿: {elapsed:.1f}绉?)
        print(f"馃摠 鎺ユ敹娑堟伅: {self.stats['total_messages']} 鏉?)
        print(f"鈿?鎺ユ敹閫熺巼: {rate:.2f} 鏉?绉?)
        print(f"馃摗 骞冲潎鏃跺欢: {avg_latency:.2f}ms")
        
        # 鏃跺欢鍒嗕綅鏁帮紙纭爣鍑?锛?
        if self.stats['latency_list']:
            percentiles = self.calculate_percentiles()
            print(f"馃搳 鏃跺欢鍒嗕綅:")
            print(f"   - P50 (涓綅鏁?: {percentiles['p50']:.2f}ms")
            print(f"   - P95: {percentiles['p95']:.2f}ms")
            print(f"   - P99: {percentiles['p99']:.2f}ms")
            print(f"馃搲 鏈€灏忔椂寤? {min(self.stats['latency_list']):.2f}ms")
            print(f"馃搱 鏈€澶ф椂寤? {max(self.stats['latency_list']):.2f}ms")
        
        # 搴忓垪涓€鑷存€х粺璁★紙纭爣鍑? - 鏈熻揣WS涓ユ牸瀵归綈 v2锛?
        print(f"馃敆 搴忓垪涓€鑷存€?(pu==last_u):")
        print(f"   - Resyncs (杩炵画鎬ф柇瑁?: {self.stats['resyncs']} 娆?)
        print(f"   - Reconnects (閲嶈繛): {self.stats['reconnects']} 娆?)
        print(f"   - Batch Span (瑙傛祴): avg={self.stats['batch_span_sum'] / self.stats['total_messages']:.1f}, max={self.stats['batch_span_max']}")
        
        print(f"馃捑 缂撳瓨鏁版嵁: {len(self.order_book_history)} 鏉?)
        print("=" * 80)
        print()
    
    def on_open(self, ws):
        """WebSocket杩炴帴鎴愬姛鏃剁殑鍥炶皟
        
        Args:
            ws: WebSocket杩炴帴瀵硅薄
        """.logger.info(f"鉁?WebSocket杩炴帴鎴愬姛: {self.symbol.upper()}").logger.info(f"璁㈤槄璁㈠崟绨挎祦: {self.depth_levels}妗ｆ繁搴? 100ms鏇存柊")
    
    def on_message(self, ws, message):
        """鎺ユ敹鍒癢ebSocket娑堟伅鏃剁殑鍥炶皟
        
        杩欎釜鏂规硶鍦ㄦ瘡娆℃敹鍒拌鍗曠翱鏇存柊鏃惰璋冪敤锛堢害100ms涓€娆★級銆?
        娑堟伅鏍煎紡涓篔SON瀛楃涓诧紝鍖呭惈鏃堕棿鎴冲拰璁㈠崟绨挎暟鎹€?
        
        Args:
            ws: WebSocket杩炴帴瀵硅薄
            message (str): 鎺ユ敹鍒扮殑JSON娑堟伅
            
        甯佸畨璁㈠崟绨挎秷鎭牸寮?
        {
            "e": "depthUpdate",        // 浜嬩欢绫诲瀷
            "E": 1234567890123,        // 浜嬩欢鏃堕棿锛堟绉掞級
            "s": "ETHUSDT",           // 浜ゆ槗瀵?
            "U": 123456789,           // 绗竴涓洿鏂癐D
            "u": 123456799,           // 鏈€鍚庝竴涓洿鏂癐D
            "b": [                    // 涔板崟锛坆ids锛?
                ["3245.50", "10.5"],  // [浠锋牸, 鏁伴噺]
                ["3245.40", "8.3"],
                ...
            ],
            "a": [                    // 鍗栧崟锛坅sks锛?
                ["3245.60", "11.2"],
                ["3245.70", "9.5"],
                ...
            ]
        }
        """
        try:
            # 1. 楠岃瘉娑堟伅鏍煎紡
            if not message or not isinstance(message, str):
                self.logger.warning(f"鏀跺埌鏃犳晥娑堟伅鏍煎紡: {type(message)}")
                return
            
            # 2. 瑙ｆ瀽JSON鏁版嵁
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:.logger.error(f"JSON瑙ｆ瀽澶辫触: {e}")
                return
            
            # 3. 楠岃瘉蹇呴渶瀛楁
            if 'E' not in data or 'b' not in data or 'a' not in data:.logger.warning(f"娑堟伅缂哄皯蹇呴渶瀛楁: {data.keys()}")
                return
            
            # 4. 璁＄畻鎺ユ敹鏃跺欢锛堝寮虹増锛?
            receive_time = datetime.now()
            ts_recv = receive_time.timestamp() * 1000  # 鎺ユ敹鏃堕棿鎴筹紙姣锛?
            timestamp_ms = data['E']  # 浜嬩欢鏃堕棿鎴筹紙姣锛?
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
            
            # 璁＄畻涓ょ鏃跺欢
            latency_event_ms = (receive_time - timestamp).total_seconds() * 1000  # 浜嬩欢鏃跺欢
            pipeline_start = datetime.now()
            
            # 5. 閫掑搴忓垪鍙?
            self.message_seq += 1
            
            # 6. 鎻愬彇鏇存柊ID瀛楁锛圲, u, pu锛? 鎸塀inance瀹樻柟瑙勮寖
            U = data.get('U', 0)   # 绗竴涓洿鏂癐D
            u = data.get('u', 0)   # 鏈€鍚庝竴涓洿鏂癐D
            pu = data.get('pu', None)  # Previous final update ID锛堝竵瀹夊畼鏂瑰瓧娈碉級
            
            # 7. 璁＄畻batch_span锛堜粎瑙傛祴锛屼笉褰撻敊璇級
            batch_span = u - U + 1
            self.stats['batch_span_sum'] += batch_span
            self.stats['batch_span_max'] = max(self.stats['batch_span_max'], batch_span)
            self.stats['batch_span_list'].append(batch_span)  # 鐢ㄤ簬P95璁＄畻
            
            # 8. 杩炵画鎬ф娴嬶細pu == last_u锛堟寜Binance瀹樻柟瑙勮寖锛?
            if self.stats['last_u'] is not None and pu is not None:
                if pu != self.stats['last_u']:
                    # 瑙﹀彂resync - 杩炵画鎬ф柇瑁?
                    self.stats['resyncs'] += 1.logger.warning(f"鈿狅笍 Resync瑙﹀彂! pu={pu}, last_u={self.stats['last_u']}, break={abs(pu - self.stats['last_u'])}")
            
            # 9. 鏇存柊last_u鍜宲rev_update_id
            prev_update_id = self.last_update_id if self.last_update_id is not None else pu
            self.stats['last_u'] = u
            self.last_update_id = u
            
            # 5. 鎻愬彇涔板崟锛坆ids锛? 5妗?
            bids = []
            for bid in data['b'][:self.depth_levels]:
                price = float(bid[0])
                quantity = float(bid[1])
                bids.append([price, quantity])
            
            # 6. 鎻愬彇鍗栧崟锛坅sks锛? 5妗?
            asks = []
            for ask in data['a'][:self.depth_levels]:
                price = float(ask[0])
                quantity = float(ask[1])
                asks.append([price, quantity])
            
            # 7. 楠岃瘉鏁版嵁瀹屾暣鎬?
            if len(bids) < self.depth_levels or len(asks) < self.depth_levels:.logger.warning(f"璁㈠崟绨挎繁搴︿笉瓒? bids={len(bids)}, asks={len(asks)}")
                return
            
            # 8. 璁＄畻绠￠亾鏃跺欢
            latency_pipeline_ms = (datetime.now() - pipeline_start).total_seconds() * 1000
            
            # 9. 鏋勫缓璁㈠崟绨挎暟鎹粨鏋勶紙瀹屾暣鐗?- 婊¤冻NDJSON瀛楁瑕佹眰锛?
            order_book = {
                'seq': self.message_seq,  # 搴忓垪鍙?
                'timestamp': timestamp,
                'symbol': self.symbol.upper(),
                'bids': bids,
                'asks': asks,
                # 鏂板瀹屾暣瀛楁
                'ts_recv': ts_recv,  # 鎺ユ敹鏃堕棿鎴筹紙姣锛?
                'E': timestamp_ms,  # 浜嬩欢鏃堕棿锛堜繚鐣欏師瀛楁鍚嶏級
                'U': U,  # 绗竴涓洿鏂癐D
                'u': u,  # 鏈€鍚庝竴涓洿鏂癐D
                'pu': prev_update_id,  # 涓婁竴涓洿鏂癐D锛堝疄闄呮槸涓婁竴鏉℃秷鎭殑u锛?
                'latency_event_ms': round(latency_event_ms, 2),  # 浜嬩欢鏃跺欢
                'latency_pipeline_ms': round(latency_pipeline_ms, 2),  # 绠￠亾鏃跺欢
                # 淇濈暀鍏煎瀛楁
                'event_time': timestamp_ms,
                'latency_ms': round(latency_event_ms, 2),
                'receive_time': receive_time
            }
            
            # 10. 鏇存柊缁熻鏁版嵁
            self.stats['total_messages'] += 1
            self.stats['latency_list'].append(latency_event_ms)
            # 鍙繚鐣欐渶杩?000涓椂寤舵暟鎹紙婊氬姩绐楀彛锛?
            if len(self.stats['latency_list']) > 1000:
                self.stats['latency_list'] = self.stats['latency_list'][-1000:]
            
            # 10. 瀛樺偍鍒板巻鍙茶褰?
            self.order_book_history.append(order_book)
            
            # 11. 瀹炴椂鍐欏叆NDJSON
            self._write_to_ndjson(order_book)
            
            # 12. 閲囨牱log_queue鎸囨爣锛圱ask 1.1.6锛?
            queue_metrics = sample_queue_metrics(self.queue_handler)
            self.stats['log_queue_depth_list'].append(queue_metrics['depth'])
            self.stats['log_queue_max_depth'] = max(self.stats['log_queue_max_depth'], queue_metrics['max_depth'])
            self.stats['log_drops'] = queue_metrics['drops']
            
            # 13. 瀹氭湡鎵撳嵃璁㈠崟绨垮拰淇濆瓨鎸囨爣锛堟瘡print_interval绉掍竴娆★級
            time_since_print = (datetime.now() - self.stats['last_print_time']).total_seconds()
            if time_since_print >= self.print_interval:
                self.print_statistics()  # 鍙墦鍗版憳瑕侊紙涓嶆墦鍗板ぇ琛ㄦ牸锛?
                self.save_metrics_json()  # 淇濆瓨metrics.json
                self.stats['last_print_time'] = datetime.now()
            
            # 13. 鏃ュ織璁板綍锛堟瘡100鏉′竴娆★級
            if self.stats['total_messages'] % 100 == 0:.logger.info(f"宸叉帴鏀?{self.stats['total_messages']} 鏉¤鍗曠翱鏁版嵁, "
                           f"閫熺巼: {self.stats['total_messages'] / (datetime.now() - self.stats['start_time']).total_seconds():.2f} 鏉?绉?)
                
        except Exception as e:.logger.error(f"澶勭悊娑堟伅鏃跺嚭閿? {e}", exc_info=True)
    
    def on_error(self, ws, error):
        """WebSocket杩炴帴鍑洪敊鏃剁殑鍥炶皟
        
        Args:
            ws: WebSocket杩炴帴瀵硅薄
            error: 閿欒瀵硅薄鎴栭敊璇秷鎭?
        """.logger.error(f"鉂?WebSocket閿欒: {error}")
        
        # 鏍规嵁閿欒绫诲瀷杩涜鍒嗙被澶勭悊
        error_str = str(error)
        if "Connection refused" in error_str:.logger.error("杩炴帴琚嫆缁濓紝璇锋鏌ョ綉缁滄垨URL")
        elif "timeout" in error_str.lower():.logger.error("杩炴帴瓒呮椂锛屽彲鑳界綉缁滀笉绋冲畾")
        elif "SSL" in error_str or "Certificate" in error_str:.logger.error("SSL璇佷功閿欒")
        else:.logger.error(f"鏈煡閿欒绫诲瀷: {error_str}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket杩炴帴鍏抽棴鏃剁殑鍥炶皟
        
        Args:
            ws: WebSocket杩炴帴瀵硅薄
            close_status_code: 鍏抽棴鐘舵€佺爜
            close_msg: 鍏抽棴娑堟伅
        """.logger.warning(f"鈿狅笍 WebSocket杩炴帴宸插叧闂?).logger.warning(f"鐘舵€佺爜: {close_status_code}").logger.warning(f"鍏抽棴娑堟伅: {close_msg}")
        
        # 璁板綍閲嶈繛娆℃暟锛堢‖鏍囧噯3锛?
        self.stats['reconnects'] += 1.logger.info(f"閲嶈繛娆℃暟: {self.stats['reconnects']}")
        
        # 璁板綍杩炴帴缁熻
        total_records = len(self.order_book_history).logger.info(f"鏈浼氳瘽鍏辨帴鏀?{total_records} 鏉¤鍗曠翱鏁版嵁")
    
    def get_latest_order_book(self):
        """鑾峰彇鏈€鏂扮殑璁㈠崟绨挎暟鎹?
        
        Returns:
            dict: 鏈€鏂扮殑璁㈠崟绨挎暟鎹紝濡傛灉娌℃湁鏁版嵁鍒欒繑鍥濶one
            
        Example:
            >>> client = BinanceOrderBookStream()
            >>> # ... 杩愯涓€娈垫椂闂村悗 ...
            >>> latest = client.get_latest_order_book()
            >>> if latest:
            ...     print(f"Bid1: {latest['bids'][0]}")
            ...     print(f"Ask1: {latest['asks'][0]}")
        """
        if len(self.order_book_history) == 0:
            return None
        return self.order_book_history[-1]
    
    def get_order_book_count(self):
        """鑾峰彇宸叉帴鏀剁殑璁㈠崟绨挎暟鎹€绘暟
        
        Returns:
            int: 璁㈠崟绨挎暟鎹暟閲?
        """
        return len(self.order_book_history)
    
    def calculate_percentiles(self):
        """璁＄畻鏃跺欢鍒嗕綅鏁帮紙纭爣鍑?锛歱50/p95/p99锛?
        
        Returns:
            dict: 鍖呭惈p50, p95, p99鐨勫瓧鍏?
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
        """淇濆瓨鎸囨爣鍒癿etrics.json鏂囦欢锛堢‖鏍囧噯4锛氬懆鏈熶骇鐗╋級
        
        姣?0绉掑埛鏂颁竴娆★紝淇濆瓨褰撳墠杩愯缁熻鍜屽垎浣嶆暟
        """
        try:
            elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
            rate = self.stats['total_messages'] / elapsed if elapsed > 0 else 0
            
            # 璁＄畻鍒嗕綅鏁?
            percentiles = self.calculate_percentiles()
            
            # 鏋勫缓鎸囨爣鏁版嵁
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
                    'resyncs': self.stats['resyncs'],
                    'reconnects': self.stats['reconnects'],
                    'batch_span_avg': round(self.stats['batch_span_sum'] / self.stats['total_messages'], 2) if self.stats['total_messages'] > 0 else 0,
                    'batch_span_max': self.stats['batch_span_max']
                },
                'cache_size': len(self.order_book_history),
                'symbol': self.symbol.upper()
            }
            
            # 淇濆瓨鍒版枃浠?
            metrics_file = self.data_dir / 'metrics.json'
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False).logger.debug(f"鎸囨爣宸蹭繚瀛樺埌 {metrics_file}")
            
        except Exception as e:.logger.error(f"淇濆瓨metrics.json澶辫触: {e}", exc_info=True)
    
    def _write_to_ndjson(self, order_book):
        """瀹炴椂鍐欏叆NDJSON鏂囦欢锛堣拷鍔犳ā寮忥級
        
        Args:
            order_book (dict): 璁㈠崟绨挎暟鎹?
            
        Note:
            NDJSON鏍煎紡锛氭瘡琛屼竴涓狫SON瀵硅薄锛屼究浜庢祦寮忓鐞嗗拰鍥炴斁
            瀹屾暣瀛楁锛歵s_recv, E, U, u, pu, latency_event_ms, latency_pipeline_ms
        """
        try:
            # 鐢熸垚浠婂ぉ鐨勬枃浠跺悕
            date_str = datetime.now().strftime('%Y%m%d')
            ndjson_file = self.ndjson_dir / f"{self.symbol}_{date_str}.ndjson"
            
            # 鍑嗗鍐欏叆鐨勬暟鎹紙瀹屾暣鐗堬紝鍖呭惈鎵€鏈夊繀闇€瀛楁锛?
            record = {
                'seq': order_book['seq'],
                'timestamp': order_book['timestamp'].isoformat(),
                'symbol': order_book['symbol'],
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                # 蹇呴渶瀛楁锛堢‖鏍囧噯1锛?
                'ts_recv': order_book['ts_recv'],
                'E': order_book['E'],
                'U': order_book['U'],
                'u': order_book['u'],
                'pu': order_book['pu'],
                'latency_event_ms': order_book['latency_event_ms'],
                'latency_pipeline_ms': order_book['latency_pipeline_ms']
            }
            
            # 杩藉姞鍐欏叆锛堟瘡琛屼竴涓狫SON锛?
            with open(ndjson_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
        except Exception as e:.logger.error(f"鍐欏叆NDJSON澶辫触: {e}", exc_info=True)
    
    def convert_ndjson_to_parquet(self, ndjson_file=None):
        """灏哊DJSON鏂囦欢杞崲涓篜arquet鏍煎紡
        
        Args:
            ndjson_file (Path): NDJSON鏂囦欢璺緞锛岄粯璁よ浆鎹粖澶╃殑鏂囦欢
            
        Returns:
            str: 鐢熸垚鐨凱arquet鏂囦欢璺緞
            
        Note:
            Parquet鍒楀紡瀛樺偍锛屽帇缂╃巼楂橈紝鏌ヨ蹇紝閫傚悎OFI璁＄畻
        """
        try:
            # 濡傛灉娌℃湁鎸囧畾鏂囦欢锛屼娇鐢ㄤ粖澶╃殑
            if ndjson_file is None:
                date_str = datetime.now().strftime('%Y%m%d')
                ndjson_file = self.ndjson_dir / f"{self.symbol}_{date_str}.ndjson"
            
            if not ndjson_file.exists():.logger.warning(f"NDJSON鏂囦欢涓嶅瓨鍦? {ndjson_file}")
                return None
            
            # 璇诲彇NDJSON鏂囦欢
            records = []
            with open(ndjson_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        # 灞曞钩鏁版嵁缁撴瀯
                        flat_record = {
                            'seq': record['seq'],
                            'timestamp': record['timestamp'],
                            'event_time': record['event_time'],
                            'latency_ms': record['latency_ms'],
                            'symbol': record['symbol']
                        }
                        
                        # 娣诲姞5妗ｄ拱鍗?
                        for i, (price, qty) in enumerate(record['bids'], 1):
                            flat_record[f'bid_price_{i}'] = price
                            flat_record[f'bid_qty_{i}'] = qty
                        
                        # 娣诲姞5妗ｅ崠鍗?
                        for i, (price, qty) in enumerate(record['asks'], 1):
                            flat_record[f'ask_price_{i}'] = price
                            flat_record[f'ask_qty_{i}'] = qty
                        
                        records.append(flat_record)
            
            if not records:.logger.warning("NDJSON鏂囦欢涓虹┖")
                return None
            
            # 杞崲涓篋ataFrame
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 鐢熸垚Parquet鏂囦欢鍚?
            date_str = datetime.now().strftime('%Y%m%d')
            parquet_file = self.parquet_dir / f"{self.symbol}_{date_str}.parquet"
            
            # 淇濆瓨涓篜arquet锛堜娇鐢╯nappy鍘嬬缉锛?
            df.to_parquet(parquet_file, engine='pyarrow', compression='snappy', index=False).logger.info(f"鉁?NDJSON鈫扨arquet杞崲鎴愬姛: {parquet_file} ({len(df)} 鏉¤褰?")
            return str(parquet_file)
            
        except Exception as e:.logger.error(f"NDJSON鈫扨arquet杞崲澶辫触: {e}", exc_info=True)
            return None
    
    def save_to_csv(self, force=False):
        """淇濆瓨璁㈠崟绨挎暟鎹埌CSV鏂囦欢
        
        Args:
            force (bool): 鏄惁寮哄埗淇濆瓨锛堝拷鐣ユ椂闂撮棿闅旓級
            
        Returns:
            str: 淇濆瓨鐨勬枃浠惰矾寰勶紝濡傛灉娌℃湁淇濆瓨鍒欒繑鍥濶one
            
        Note:
            榛樿姣?0绉掕嚜鍔ㄤ繚瀛樹竴娆★紝閬垮厤棰戠箒IO鎿嶄綔
        """
        # 妫€鏌ユ槸鍚﹂渶瑕佷繚瀛?
        if not force:
            time_since_last_save = (datetime.now() - self.last_save_time).total_seconds()
            if time_since_last_save < self.save_interval:
                return None
        
        # 妫€鏌ユ槸鍚︽湁鏁版嵁
        if len(self.order_book_history) == 0:.logger.warning("娌℃湁鏁版嵁鍙繚瀛?)
            return None
        
        try:
            # 灏嗘暟鎹浆鎹负DataFrame鏍煎紡
            data_list = []
            for ob in self.order_book_history:
                # 灞曞钩璁㈠崟绨挎暟鎹?
                row = {
                    'timestamp': ob['timestamp'],
                    'symbol': ob['symbol'],
                    'event_time': ob['event_time']
                }
                
                # 娣诲姞涔板崟鏁版嵁锛?妗ｏ級
                for i, (price, qty) in enumerate(ob['bids'], 1):
                    row[f'bid_price_{i}'] = price
                    row[f'bid_qty_{i}'] = qty
                
                # 娣诲姞鍗栧崟鏁版嵁锛?妗ｏ級
                for i, (price, qty) in enumerate(ob['asks'], 1):
                    row[f'ask_price_{i}'] = price
                    row[f'ask_qty_{i}'] = qty
                
                data_list.append(row)
            
            # 鍒涘缓DataFrame
            df = pd.DataFrame(data_list)
            
            # 鐢熸垚鏂囦欢鍚嶏紙鎸夋棩鏈熷拰鏃堕棿锛?
            filename = f"{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.data_dir / filename
            
            # 淇濆瓨鍒癈SV
            df.to_csv(filepath, index=False)
            
            # 鏇存柊鏈€鍚庝繚瀛樻椂闂?
            self.last_save_time = datetime.now().logger.info(f"鉁?鏁版嵁宸蹭繚瀛? {filepath} ({len(df)} 鏉¤褰?")
            return str(filepath)
            
        except Exception as e:.logger.error(f"淇濆瓨CSV澶辫触: {e}", exc_info=True)
            return None
    
    def auto_save_loop(self):
        """鑷姩淇濆瓨寰幆锛堝湪鍚庡彴绾跨▼涓繍琛岋級"""
        while self.ws and self.ws.keep_running:
            try:
                # 姣?0绉掑皾璇曚繚瀛樹竴娆?
                import time
                time.sleep(self.save_interval)
                self.save_to_csv(force=False)
            except Exception as e:.logger.error(f"鑷姩淇濆瓨寮傚父: {e}")
                break
    
    def run(self, reconnect=True):
        """鍚姩WebSocket杩炴帴
        
        杩欎釜鏂规硶浼氬缓绔嬪埌甯佸畨WebSocket鐨勮繛鎺ワ紝骞跺紑濮嬫帴鏀惰鍗曠翱鏁版嵁銆?
        杩炴帴鏄樆濉炵殑锛屼細涓€鐩磋繍琛岀洿鍒版墜鍔ㄥ仠姝㈡垨鍙戠敓閿欒銆?
        
        Args:
            reconnect (bool): 鏄惁鍦ㄦ柇绾垮悗鑷姩閲嶈繛锛岄粯璁rue
            
        Example:
            >>> client = BinanceOrderBookStream('ethusdt')
            >>> client.run()  # 寮€濮嬫帴鏀舵暟鎹紝闃诲杩愯
        """.logger.info("=" * 60).logger.info(f"鍚姩甯佸畨WebSocket瀹㈡埛绔?).logger.info(f"浜ゆ槗瀵? {self.symbol.upper()}").logger.info(f"璁㈠崟绨挎繁搴? {self.depth_levels}妗?).logger.info(f"WebSocket URL: {self.ws_url}").logger.info(f"鑷姩閲嶈繛: {'寮€鍚? if reconnect else '鍏抽棴'}").logger.info("=" * 60)
        
        # 鍒涘缓WebSocket杩炴帴
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # 鍚姩杩炴帴锛堥樆濉炶繍琛岋級
        try:
            self.ws.run_forever(
                reconnect=5 if reconnect else 0  # 閲嶈繛闂撮殧5绉?
            )
        except KeyboardInterrupt:.logger.info("鐢ㄦ埛涓柇锛屾鍦ㄥ叧闂繛鎺?..")
            self.ws.close()
        except Exception as e:.logger.error(f"杩愯鏃跺彂鐢熷紓甯? {e}", exc_info=True)
            raise


