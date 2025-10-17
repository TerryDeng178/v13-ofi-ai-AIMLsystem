#!/usr/bin/env python3
"""最小化调试测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "v13_ofi_ai_system" / "src"))

print("=" * 80)
print("开始调试测试")
print("=" * 80)

from binance_websocket_client import BinanceOrderBookStream
import time

# 创建客户端
print("\n1. 创建客户端...")
client = BinanceOrderBookStream(
    symbol="ETHUSDT",
    depth_levels=5,
    print_interval=5  # 5秒
)

print("2. 客户端创建成功")
print(f"   - WebSocket URL: {client.ws_url}")
print(f"   - 打印间隔: {client.print_interval}秒")
print(f"   - 初始synced状态: {client.synced}")

# 手动测试对齐逻辑
print("\n3. 测试对齐逻辑...")
client.load_snapshot()
print(f"   - REST快照lastUpdateId: {client.last_update_id}")

# 模拟一个对齐事件
if client.last_update_id:
    L = client.last_update_id
    test_U = L - 100
    test_u = L + 100
    result = client._try_align_first_event(test_U, test_u)
    print(f"   - 测试对齐 U={test_U}, u={test_u}, L={L}")
    print(f"   - 对齐结果: {result} (应该为True)")

print("\n4. 启动WebSocket连接（30秒测试）...")
print("   每5秒应该看到一次SUMMARY输出\n")

import threading
t = threading.Thread(target=client.run, kwargs={"reconnect": False}, daemon=True)
t.start()

# 等待30秒
for i in range(30):
    time.sleep(1)
    if i % 5 == 4:  # 每5秒检查一次
        print(f"\n[{i+1}秒] 统计信息:")
        print(f"   - 已同步: {client.synced}")
        print(f"   - 总消息数: {client.stats['total_messages']}")
        print(f"   - 最后u: {client.last_u}")
        print(f"   - Resyncs: {client.stats['resyncs']}")

print("\n\n5. 停止测试...")
if client.ws:
    try:
        client.ws.close()
    except:
        pass

try:
    client.listener.stop()
except:
    pass

time.sleep(1)

print("\n6. 最终统计:")
print(f"   - 总消息数: {client.stats['total_messages']}")
print(f"   - 已同步: {client.synced}")
print(f"   - Resyncs: {client.stats['resyncs']}")
print(f"   - Reconnects: {client.stats['reconnects']}")

if client.stats['total_messages'] == 0:
    print("\n⚠️  问题：没有收到任何消息！")
    print("   可能原因：")
    print("   1. WebSocket连接失败")
    print("   2. 对齐一直失败")
    print("   3. 数据被过滤掉了")
elif not client.synced:
    print("\n⚠️  问题：收到消息但一直没有对齐！")
else:
    print("\n✅ 测试成功！")

print("\n" + "=" * 80)

