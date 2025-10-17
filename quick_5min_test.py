#!/usr/bin/env python3
"""5分钟快速测试"""
import sys, time, signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "v13_ofi_ai_system" / "src"))
from binance_websocket_client import BinanceOrderBookStream

print("="*80)
print("5分钟快速测试开始")
print("="*80)
print(f"开始时间: {time.strftime('%H:%M:%S')}")
print(f"预计结束: {time.strftime('%H:%M:%S', time.localtime(time.time()+300))}")
print("="*80 + "\n")

client = BinanceOrderBookStream(symbol="ETHUSDT", depth_levels=5, print_interval=10)

def stop(sig, frame):
    print("\n\n" + "="*80)
    print("停止测试...")
    print("="*80)
    if client.ws: 
        try: client.ws.close()
        except: pass
    try: client.listener.stop()
    except: pass
    sys.exit(0)

signal.signal(signal.SIGINT, stop)

import threading
t = threading.Thread(target=client.run, kwargs={"reconnect": True}, daemon=True)
t.start()

try:
    time.sleep(300)  # 5分钟
except KeyboardInterrupt:
    pass

print("\n\n" + "="*80)
print("5分钟测试完成")
print("="*80)
if client.ws:
    try: client.ws.close()
    except: pass
try: client.listener.stop()
except: pass
time.sleep(2)
print("测试结束！请查看 v13_ofi_ai_system/data/order_book/metrics.json")

