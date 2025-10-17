#!/usr/bin/env python3
"""
30分钟稳态测试脚本
Task 1.1.6验收测试
"""
import sys
import time
import signal
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "v13_ofi_ai_system" / "src"))

from binance_websocket_client import BinanceOrderBookStream

print("=" * 80)
print("Task 1.1.6: 30分钟稳态测试")
print("=" * 80)
print("\n测试参数:")
print("  - 交易对: ETHUSDT")
print("  - 测试时长: 30分钟")
print("  - 日志轮转: 大小模式（5MB）")
print("  - 保留备份: 7个")
print("  - 打印间隔: 10秒")
print("\n测试开始时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
print("预计结束时间:", time.strftime("%Y-%m-%d %H:%M:%S", 
                                    time.localtime(time.time() + 30*60)))
print("=" * 80)
print("\n提示: 按Ctrl+C可以提前停止测试\n")

# 创建客户端
client = BinanceOrderBookStream(
    symbol="ETHUSDT",
    depth_levels=5,
    rotate="size",
    rotate_sec=60,
    max_bytes=5_000_000,
    backups=7,
    print_interval=10
)

# 处理Ctrl+C
def signal_handler(sig, frame):
    print("\n\n" + "=" * 80)
    print("用户中断测试，正在停止...")
    print("=" * 80)
    if client.ws:
        try:
            client.ws.close()
        except:
            pass
    try:
        client.listener.stop()
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# 启动测试
import threading
test_thread = threading.Thread(target=client.run, kwargs={"reconnect": True}, daemon=True)
test_thread.start()

# 等待30分钟
try:
    time.sleep(30 * 60)
except KeyboardInterrupt:
    pass

# 停止
print("\n\n" + "=" * 80)
print("30分钟测试完成，正在停止...")
print("=" * 80)

if client.ws:
    try:
        client.ws.close()
    except:
        pass

try:
    client.listener.stop()
except:
    pass

# 等待线程结束
time.sleep(2)

print("\n测试结束！")
print("请查看以下文件验证结果:")
print("  - v13_ofi_ai_system/logs/*.log")
print("  - v13_ofi_ai_system/data/order_book/metrics.json")
print("  - v13_ofi_ai_system/data/order_book/ethusdt_depth.ndjson.gz")

