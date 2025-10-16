#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安测试网集成系统
支持ETHUSDT永续合约的实时数据获取和交易
"""

import hmac
import hashlib
import time
import requests
import json
import websocket
import threading
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceTestnetAPI:
    """币安测试网API客户端"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://testnet.binancefuture.com"
        self.ws_url = "wss://stream.binancefuture.com"
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        })
        
    def _generate_signature(self, params: str) -> str:
        """生成签名"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        return str(int(time.time() * 1000))
    
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        timestamp = self._get_timestamp()
        params = f"timestamp={timestamp}"
        signature = self._generate_signature(params)
        
        url = f"{self.base_url}/fapi/v2/account?{params}&signature={signature}"
        response = self.session.get(url)
        return response.json()
    
    def get_position_info(self, symbol: str = "ETHUSDT") -> List[Dict]:
        """获取持仓信息"""
        timestamp = self._get_timestamp()
        params = f"symbol={symbol}&timestamp={timestamp}"
        signature = self._generate_signature(params)
        
        url = f"{self.base_url}/fapi/v2/positionRisk?{params}&signature={signature}"
        response = self.session.get(url)
        return response.json()
    
    def get_orderbook(self, symbol: str = "ETHUSDT", limit: int = 1000) -> Dict:
        """获取订单簿"""
        url = f"{self.base_url}/fapi/v1/depth?symbol={symbol}&limit={limit}"
        response = self.session.get(url)
        return response.json()
    
    def get_klines(self, symbol: str = "ETHUSDT", interval: str = "1s", limit: int = 1000) -> List[List]:
        """获取K线数据"""
        url = f"{self.base_url}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = self.session.get(url)
        return response.json()
    
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, 
                   price: Optional[float] = None, time_in_force: str = "GTC") -> Dict:
        """下单"""
        timestamp = self._get_timestamp()
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'timestamp': timestamp
        }
        
        if price:
            params['price'] = price
        if time_in_force:
            params['timeInForce'] = time_in_force
            
        params_str = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = self._generate_signature(params_str)
        
        url = f"{self.base_url}/fapi/v1/order?{params_str}&signature={signature}"
        response = self.session.post(url)
        return response.json()
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """取消订单"""
        timestamp = self._get_timestamp()
        params = f"symbol={symbol}&orderId={order_id}&timestamp={timestamp}"
        signature = self._generate_signature(params)
        
        url = f"{self.base_url}/fapi/v1/order?{params}&signature={signature}"
        response = self.session.delete(url)
        return response.json()
    
    def get_open_orders(self, symbol: str = "ETHUSDT") -> List[Dict]:
        """获取未成交订单"""
        timestamp = self._get_timestamp()
        params = f"symbol={symbol}&timestamp={timestamp}"
        signature = self._generate_signature(params)
        
        url = f"{self.base_url}/fapi/v1/openOrders?{params}&signature={signature}"
        response = self.session.get(url)
        return response.json()

class BinanceWebSocketClient:
    """币安WebSocket客户端"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.ws = None
        self.is_connected = False
        self.callbacks = {}
        
    def connect(self, streams: List[str]):
        """连接WebSocket"""
        stream_names = '/'.join(streams)
        ws_url = f"wss://stream.binancefuture.com/stream?streams={stream_names}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                stream = data.get('stream')
                if stream in self.callbacks:
                    self.callbacks[stream](data['data'])
            except Exception as e:
                logger.error(f"WebSocket消息处理错误: {e}")
                logger.error(f"原始消息: {message}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            
        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")
            self.is_connected = False
            
        def on_open(ws):
            logger.info("WebSocket connection opened")
            self.is_connected = True
            
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # 在新线程中运行WebSocket
        def run_ws():
            self.ws.run_forever()
            
        ws_thread = threading.Thread(target=run_ws)
        ws_thread.daemon = True
        ws_thread.start()
        
    def subscribe(self, stream: str, callback: Callable):
        """订阅数据流"""
        self.callbacks[stream] = callback
        
    def disconnect(self):
        """断开连接"""
        if self.ws:
            self.ws.close()
            self.is_connected = False

class BinanceDataProcessor:
    """币安数据处理器"""
    
    def __init__(self):
        self.market_data = []
        self.orderbook_data = []
        self.trade_data = []
        
    def process_ticker(self, data: Dict):
        """处理ticker数据"""
        ticker = {
            'timestamp': datetime.now(),
            'symbol': data['s'],
            'price': float(data['c']),
            'bid': float(data['b']),
            'ask': float(data['a']),
            'volume': float(data['v']),
            'change': float(data['P']),
            'high': float(data['h']),
            'low': float(data['l'])
        }
        self.market_data.append(ticker)
        logger.info(f"Ticker: {ticker['symbol']} = {ticker['price']}")
        
    def process_orderbook(self, data: Dict):
        """处理订单簿数据"""
        orderbook = {
            'timestamp': datetime.now(),
            'symbol': data['s'],
            'bids': [[float(bid[0]), float(bid[1])] for bid in data['b'][:5]],
            'asks': [[float(ask[0]), float(ask[1])] for ask in data['a'][:5]]
        }
        self.orderbook_data.append(orderbook)
        logger.info(f"Orderbook: {orderbook['symbol']} - Bids: {len(orderbook['bids'])}, Asks: {len(orderbook['asks'])}")
        
    def process_trade(self, data: Dict):
        """处理交易数据"""
        trade = {
            'timestamp': datetime.now(),
            'symbol': data['s'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'side': data['S'],
            'trade_id': data['t']
        }
        self.trade_data.append(trade)
        logger.info(f"Trade: {trade['symbol']} {trade['side']} {trade['quantity']} @ {trade['price']}")
        
    def get_latest_data(self, data_type: str = 'market') -> List[Dict]:
        """获取最新数据"""
        if data_type == 'market':
            return self.market_data[-100:] if self.market_data else []
        elif data_type == 'orderbook':
            return self.orderbook_data[-100:] if self.orderbook_data else []
        elif data_type == 'trade':
            return self.trade_data[-100:] if self.trade_data else []
        return []

class BinanceTradingBot:
    """币安交易机器人"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api = BinanceTestnetAPI(api_key, secret_key)
        self.ws_client = BinanceWebSocketClient(api_key, secret_key)
        self.data_processor = BinanceDataProcessor()
        self.symbol = "ETHUSDT"
        self.is_running = False
        
    def start(self):
        """启动交易机器人"""
        logger.info("启动币安交易机器人...")
        
        # 连接WebSocket
        streams = [
            f"{self.symbol.lower()}@ticker",
            f"{self.symbol.lower()}@depth5@100ms",
            f"{self.symbol.lower()}@trade"
        ]
        
        self.ws_client.connect(streams)
        
        # 订阅数据流
        self.ws_client.subscribe(f"{self.symbol.lower()}@ticker", self.data_processor.process_ticker)
        self.ws_client.subscribe(f"{self.symbol.lower()}@depth5@100ms", self.data_processor.process_orderbook)
        self.ws_client.subscribe(f"{self.symbol.lower()}@trade", self.data_processor.process_trade)
        
        self.is_running = True
        
        # 获取账户信息
        try:
            account_info = self.api.get_account_info()
            logger.info(f"账户余额: {account_info.get('totalWalletBalance', 'N/A')} USDT")
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            
        # 获取持仓信息
        try:
            positions = self.api.get_position_info(self.symbol)
            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    logger.info(f"持仓: {pos['symbol']} {pos['positionAmt']} {pos['unrealizedPnl']}")
        except Exception as e:
            logger.error(f"获取持仓信息失败: {e}")
    
    def stop(self):
        """停止交易机器人"""
        logger.info("停止币安交易机器人...")
        self.is_running = False
        self.ws_client.disconnect()
    
    def get_market_data(self) -> List[Dict]:
        """获取市场数据"""
        return self.data_processor.get_latest_data('market')
    
    def get_orderbook_data(self) -> List[Dict]:
        """获取订单簿数据"""
        return self.data_processor.get_latest_data('orderbook')
    
    def get_trade_data(self) -> List[Dict]:
        """获取交易数据"""
        return self.data_processor.get_latest_data('trade')
    
    def place_market_order(self, side: str, quantity: float) -> Dict:
        """下市价单"""
        try:
            result = self.api.place_order(
                symbol=self.symbol,
                side=side,
                order_type="MARKET",
                quantity=quantity
            )
            logger.info(f"下单成功: {result}")
            return result
        except Exception as e:
            logger.error(f"下单失败: {e}")
            return {}
    
    def place_limit_order(self, side: str, quantity: float, price: float) -> Dict:
        """下限价单"""
        try:
            result = self.api.place_order(
                symbol=self.symbol,
                side=side,
                order_type="LIMIT",
                quantity=quantity,
                price=price
            )
            logger.info(f"下单成功: {result}")
            return result
        except Exception as e:
            logger.error(f"下单失败: {e}")
            return {}
    
    def get_open_orders(self) -> List[Dict]:
        """获取未成交订单"""
        try:
            return self.api.get_open_orders(self.symbol)
        except Exception as e:
            logger.error(f"获取未成交订单失败: {e}")
            return []
    
    def cancel_all_orders(self):
        """取消所有订单"""
        try:
            open_orders = self.get_open_orders()
            for order in open_orders:
                self.api.cancel_order(self.symbol, order['orderId'])
                logger.info(f"取消订单: {order['orderId']}")
        except Exception as e:
            logger.error(f"取消订单失败: {e}")

def main():
    """主函数"""
    # 币安测试网API配置
    API_KEY = "tKs2nQW3t3rcHnUCGOPWLIYsDotF6ot1GRZzo5n3QOsNRqbPTxO06TC8QglxtUYy"
    SECRET_KEY = "I5SqluL8yqvkcSSb0hBBMr80cHmNndxrl6DAufw9t6myjlRBQDvPxUmabHNJpkZJ"
    
    # 创建交易机器人
    bot = BinanceTradingBot(API_KEY, SECRET_KEY)
    
    try:
        # 启动机器人
        bot.start()
        
        # 运行一段时间
        import time
        time.sleep(30)  # 运行30秒
        
        # 获取数据
        market_data = bot.get_market_data()
        orderbook_data = bot.get_orderbook_data()
        trade_data = bot.get_trade_data()
        
        logger.info(f"市场数据: {len(market_data)}条")
        logger.info(f"订单簿数据: {len(orderbook_data)}条")
        logger.info(f"交易数据: {len(trade_data)}条")
        
        # 显示最新数据
        if market_data:
            latest = market_data[-1]
            logger.info(f"最新价格: {latest['price']}")
            logger.info(f"买卖价差: {latest['ask'] - latest['bid']}")
        
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        # 停止机器人
        bot.stop()

if __name__ == "__main__":
    main()
