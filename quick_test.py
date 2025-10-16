#!/usr/bin/env python3
"""Quick test script for binance_websocket_client.py"""
import sys
import time
import threading
from pathlib import Path

# Import the client
from binance_websocket_client import BinanceOrderBookStream

print("=" * 60)
print("Quick Test: 20 seconds")
print("=" * 60)

# Create client
client = BinanceOrderBookStream(
    symbol="ETHUSDT",
    depth_levels=5,
    rotate="interval",
    rotate_sec=60,
    backups=7,
    print_interval=5  # 5 seconds
)

# Run in thread
t = threading.Thread(target=client.run, kwargs={"reconnect": False}, daemon=True)
t.start()

print("测试运行中...")
time.sleep(20)

# Stop
if client.ws:
    try:
        client.ws.close()
    except:
        pass

# Stop listener
try:
    client.listener.stop()
except:
    pass

print("\n" + "=" * 60)
print("Test completed! Checking files...")
print("=" * 60)

# Check files
log_dir = Path("v13_ofi_ai_system/logs")
data_dir = Path("v13_ofi_ai_system/data/order_book")

logs = list(log_dir.glob("*.log*"))
print(f"\nLog files: {len(logs)}")
for log in logs[:3]:
    print(f"  - {log.name} ({log.stat().st_size} bytes)")

if (data_dir / "metrics.json").exists():
    import json
    with open(data_dir / "metrics.json") as f:
        metrics = json.load(f)
    print(f"\nmetrics.json:")
    print(f"  - Messages: {metrics.get('total_messages', 0)}")
    print(f"  - Rate: {metrics.get('recv_rate', 0)} msg/s")
    print(f"  - Resyncs: {metrics['continuity'].get('resyncs', 0)}")
    print(f"  - Log drops: {metrics['log_queue'].get('drops', 0)}")
else:
    print("\nmetrics.json not found")

if (data_dir / "ethusdt_depth.ndjson.gz").exists():
    import gzip
    with gzip.open(data_dir / "ethusdt_depth.ndjson.gz", "rt") as f:
        lines = f.readlines()
    print(f"\nNDJSON: {len(lines)} lines")
else:
    print("\nNDJSON not found")

print("\nTest completed OK!")

