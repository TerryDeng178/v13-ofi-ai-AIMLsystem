#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试订单簿流连接
"""

import asyncio
import websockets
import json

async def test_orderbook_connection():
    """测试订单簿流连接"""
    try:
        print("测试订单簿流连接...")
        async with websockets.connect('wss://stream.binance.com:9443/ws/btcusdt@depth5@100ms', ping_interval=20) as ws:
            print("✅ 订单簿流连接成功")
            
            # 接收几条消息测试
            for i in range(3):
                message = await ws.recv()
                data = json.loads(message)
                print(f"收到消息 {i+1}: bids={len(data.get('bids', []))}, asks={len(data.get('asks', []))}")
            
            return True
    except Exception as e:
        print(f"❌ 订单簿流连接失败: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_orderbook_connection())
