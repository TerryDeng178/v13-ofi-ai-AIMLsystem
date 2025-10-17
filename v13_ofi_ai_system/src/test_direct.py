#!/usr/bin/env python3
from binance_websocket_client import BinanceOrderBookStream
import time, threading

print("直接测试 binance_websocket_client.py")
print("=" * 60)

c = BinanceOrderBookStream('ETHUSDT', print_interval=5)
print("客户端创建成功")

t = threading.Thread(target=c.run, daemon=True)
t.start()
print("WebSocket启动...")

print("\n等待30秒...\n")
time.sleep(30)

print(f"\n结果:")
print(f"  总消息: {c.stats['total_messages']}")
print(f"  已同步: {c.synced}")
print(f"  Resyncs: {c.stats['resyncs']}")

if c.ws:
    c.ws.close()
c.listener.stop()

