#!/usr/bin/env python3
"""
V10.0 增强客户端启动脚本
连接V10增强实时市场模拟器，支持3级加权OFI和深度学习信号监控
"""

import asyncio
import argparse
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from client_v10_enhanced import V10EnhancedClient

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='V10.0 增强实时市场模拟器客户端')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8765, help='服务器端口')
    parser.add_argument('--mode', type=str, choices=['interactive', 'automated'], 
                       default='interactive', help='运行模式')
    parser.add_argument('--duration', type=int, default=60, help='自动化模式运行时长(秒)')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 构建WebSocket URI
    uri = f"ws://{args.host}:{args.port}"
    
    if args.verbose:
        print("V10.0 增强实时市场模拟器客户端配置:")
        print(f"  服务器: {uri}")
        print(f"  模式: {args.mode}")
        if args.mode == 'automated':
            print(f"  运行时长: {args.duration}秒")
    
    # 创建客户端
    client = V10EnhancedClient(uri)
    
    print("="*60)
    print("V10.0 增强实时市场模拟器客户端")
    print("="*60)
    print(f"连接地址: {uri}")
    print(f"运行模式: {args.mode}")
    print("支持功能:")
    print("  [OK] 实时市场数据监控")
    print("  [OK] 3级加权OFI分析")
    print("  [OK] 深度学习信号生成")
    print("  [OK] 实时性能统计")
    print("  [OK] 数据可视化")
    print("="*60)
    
    async def run_client():
        """运行客户端"""
        # 连接到服务器
        if not await client.connect():
            print("连接失败，退出")
            return
            
        try:
            if args.mode == 'interactive':
                # 交互式模式
                await client.run_interactive()
            else:
                # 自动化模式
                await client.run_automated(args.duration)
        except Exception as e:
            print(f"客户端运行失败: {e}")
        finally:
            await client.disconnect()
    
    try:
        # 运行客户端
        asyncio.run(run_client())
    except KeyboardInterrupt:
        print("\n客户端已停止")
    except Exception as e:
        print(f"客户端启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
