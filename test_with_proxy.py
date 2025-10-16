#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试带代理的币安WebSocket连接"""
import sys, io, time, threading
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*80)
print("🔧 币安WebSocket代理配置测试")
print("="*80)
print()

# 首先测试基本网络连接
print("📡 步骤1: 测试币安REST API连接（不需要WebSocket）")
print("-" * 80)

import requests

def test_rest_api():
    """测试REST API连接"""
    url = "https://fapi.binance.com/fapi/v1/ping"
    
    try:
        print(f"🔍 测试URL: {url}")
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            print("✅ REST API连接成功！")
            print(f"   状态码: {response.status_code}")
            return True
        else:
            print(f"❌ REST API连接失败")
            print(f"   状态码: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print("❌ 连接超时（5秒）")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False

# 测试REST API
rest_ok = test_rest_api()
print()

if not rest_ok:
    print("⚠️  无法连接到币安服务器！")
    print()
    print("可能的原因和解决方案:")
    print("1. 网络连接问题 - 请检查您的网络")
    print("2. 防火墙阻止 - 请检查防火墙设置")
    print("3. 地区限制 - 需要使用代理或VPN")
    print()
    print("💡 配置代理的方法:")
    print()
    print("方法1: 系统环境变量（推荐）")
    print("  Windows PowerShell:")
    print('  $env:HTTP_PROXY="http://proxy_server:port"')
    print('  $env:HTTPS_PROXY="http://proxy_server:port"')
    print()
    print("方法2: Python代码中配置")
    print("  proxies = {")
    print('    "http": "http://proxy_server:port",')
    print('    "https": "http://proxy_server:port"')
    print("  }")
    print()
    print("方法3: 使用VPN")
    print("  连接到可访问币安的VPN服务器")
    print()
    sys.exit(1)

# REST API连接成功，继续测试WebSocket
print("🚀 步骤2: 测试WebSocket连接")
print("-" * 80)

from v13_ofi_ai_system.src.binance_websocket_client import BinanceOrderBookStream

# 创建客户端
client = BinanceOrderBookStream(symbol='ethusdt', depth_levels=5)
print(f"✅ 客户端创建成功")
print(f"   交易对: {client.symbol.upper()}")
print(f"   WebSocket URL: {client.ws_url}")
print()

def run_websocket():
    """在单独线程中运行WebSocket"""
    try:
        client.run(reconnect=False)
    except Exception as e:
        print(f"❌ WebSocket异常: {e}")

# 启动WebSocket
print("⏳ 正在连接WebSocket服务器...")
print("   如果10秒内没有看到'连接成功'消息，说明WebSocket被阻止")
print()

ws_thread = threading.Thread(target=run_websocket, daemon=True)
ws_thread.start()

# 等待10秒
for i in range(10):
    time.sleep(1)
    count = client.get_order_book_count()
    if count > 0:
        print(f"✅ WebSocket连接成功！已接收 {count} 条数据")
        break
    if i == 4:
        print(f"⏳ [{i+1}秒] 仍在等待连接...")
    if i == 9:
        print(f"❌ [{i+1}秒] WebSocket连接超时")

# 检查结果
print()
print("="*80)
print("📊 测试结果总结")
print("="*80)

count = client.get_order_book_count()
if count > 0:
    print(f"✅ 测试成功！")
    print(f"   REST API: ✅ 正常")
    print(f"   WebSocket: ✅ 正常")
    print(f"   接收数据: {count} 条")
    print()
    
    latest = client.get_latest_order_book()
    if latest:
        print("📊 最新订单簿数据:")
        print(f"   时间: {latest['timestamp']}")
        print(f"   买一: {latest['bids'][0][0]:.2f} USDT")
        print(f"   卖一: {latest['asks'][0][0]:.2f} USDT")
        print(f"   价差: {latest['asks'][0][0] - latest['bids'][0][0]:.2f} USDT")
    print()
    print("🎉 可以正常使用WebSocket实时数据！")
else:
    print(f"⚠️  测试结果：")
    print(f"   REST API: ✅ 正常")
    print(f"   WebSocket: ❌ 无法连接")
    print()
    print("📋 WebSocket无法连接的原因分析：")
    print()
    print("1. WebSocket协议被阻止")
    print("   - 某些网络环境只允许HTTP/HTTPS，不允许WebSocket")
    print("   - 需要使用支持WebSocket的代理")
    print()
    print("2. 币安WebSocket服务器访问受限")
    print("   - 某些地区无法访问币安WebSocket")
    print("   - 需要使用VPN或代理")
    print()
    print("💡 解决方案建议：")
    print()
    print("方案A: 使用REST API替代方案（推荐）")
    print("  - 使用REST API定期轮询订单簿数据")
    print("  - 虽然不是实时，但足够用于策略开发")
    print("  - 可以下载历史数据进行回测")
    print()
    print("方案B: 配置WebSocket代理")
    print("  - 使用支持WebSocket的HTTP代理")
    print("  - 配置方法：")
    print("    import websocket")
    print("    ws = websocket.WebSocketApp(url,")
    print('        http_proxy_host="proxy_server",')
    print('        http_proxy_port=port)')
    print()
    print("方案C: 使用VPN")
    print("  - 连接到可访问币安的VPN")
    print("  - 重新测试连接")

# 关闭WebSocket
if client.ws:
    client.ws.close()

print()
print("测试完成。")

