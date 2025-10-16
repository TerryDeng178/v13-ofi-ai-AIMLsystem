#!/usr/bin/env python3
"""
V10.0 增强实时市场模拟器启动脚本
一键启动服务器和客户端
"""

import subprocess
import sys
import time
import os
import argparse
from pathlib import Path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='V10.0 增强实时市场模拟器启动器')
    parser.add_argument('--mode', type=str, choices=['server', 'client', 'both'], 
                       default='both', help='启动模式')
    parser.add_argument('--config', type=str, 
                       default='config/params_v10_enhanced.yaml', 
                       help='配置文件路径')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器主机')
    parser.add_argument('--port', type=int, default=8765, help='服务器端口')
    parser.add_argument('--duration', type=int, default=60, help='客户端运行时长(秒)')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent
    examples_dir = project_root / "examples"
    
    print("="*60)
    print("V10.0 增强实时市场模拟器启动器")
    print("="*60)
    print(f"项目目录: {project_root}")
    print(f"配置文件: {args.config}")
    print(f"启动模式: {args.mode}")
    print("="*60)
    
    if args.mode in ['server', 'both']:
        print("启动V10.0增强服务器...")
        server_cmd = [
            sys.executable,
            str(examples_dir / "run_v10_enhanced_server.py"),
            "--config", args.config,
            "--host", args.host,
            "--port", str(args.port)
        ]
        if args.verbose:
            server_cmd.append("--verbose")
            
        print(f"服务器命令: {' '.join(server_cmd)}")
        
        # 启动服务器
        server_process = subprocess.Popen(server_cmd)
        print(f"服务器已启动 (PID: {server_process.pid})")
        
        # 等待服务器启动
        time.sleep(3)
        
    if args.mode in ['client', 'both']:
        print("启动V10.0增强客户端...")
        client_cmd = [
            sys.executable,
            str(examples_dir / "run_v10_enhanced_client.py"),
            "--host", args.host,
            "--port", str(args.port),
            "--mode", "automated",
            "--duration", str(args.duration)
        ]
        if args.verbose:
            client_cmd.append("--verbose")
            
        print(f"客户端命令: {' '.join(client_cmd)}")
        
        # 启动客户端
        client_process = subprocess.Popen(client_cmd)
        print(f"客户端已启动 (PID: {client_process.pid})")
        
        # 等待客户端完成
        try:
            client_process.wait()
        except KeyboardInterrupt:
            print("\n用户中断，停止客户端")
            client_process.terminate()
            
    if args.mode == 'both':
        # 等待服务器进程
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n用户中断，停止服务器")
            server_process.terminate()
            
    print("V10.0增强实时市场模拟器已停止")

if __name__ == "__main__":
    main()
