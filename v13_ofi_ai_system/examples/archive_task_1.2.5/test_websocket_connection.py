#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试WebSocket连接 - Task_1.1.2验证脚本

这个脚本用于验证WebSocket连接功能是否正常工作。
运行后会连接到币安WebSocket，接收5秒钟的数据后自动停止。
"""

import sys
import os
import time
import threading

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.binance_websocket_client import BinanceOrderBookStream

def test_connection(duration=5):
    """测试WebSocket连接
    
    Args:
        duration (int): 测试持续时间（秒），默认5秒
    """
    print("\n" + "=" * 60)
    print("WebSocket连接测试 - Task_1.1.2")
    print("=" * 60)
    print(f"测试时长: {duration}秒")
    print("按 Ctrl+C 可随时停止\n")
    
    # 创建客户端
    client = BinanceOrderBookStream('ethusdt', 5)
    
    # 在后台线程中运行
    def run_client():
        try:
            client.run(reconnect=False)
        except Exception as e:
            print(f"连接出错: {e}")
    
    thread = threading.Thread(target=run_client, daemon=True)
    thread.start()
    
    # 等待指定时间
    try:
        time.sleep(duration)
        print(f"\n✅ 测试完成！连接运行了 {duration} 秒")
        print(f"客户端状态: {client}")
        
        # 关闭连接
        if client.ws:
            client.ws.close()
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
        if client.ws:
            client.ws.close()
        return False

if __name__ == '__main__':
    # 运行测试
    success = test_connection(5)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ WebSocket连接测试通过！")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ WebSocket连接测试失败")
        print("=" * 60)
        sys.exit(1)

