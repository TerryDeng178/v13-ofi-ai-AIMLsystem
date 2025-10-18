#!/usr/bin/env python3
"""
干净金测监控脚本
监控：丢弃率、时长、Z质量、系统资源
"""

import time
import psutil
import os
import json
from pathlib import Path

def monitor_test():
    """监控测试进度"""
    print("🔍 开始监控干净金测...")
    print("=" * 60)
    
    start_time = time.time()
    last_check = start_time
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # 系统资源
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        print(f"\r⏱️  运行时长: {elapsed/60:.1f}分钟 | "
              f"CPU: {cpu_percent:.1f}% | "
              f"内存: {memory.percent:.1f}% | "
              f"可用: {memory.available/(1024**3):.1f}GB", end="")
        
        # 检查数据文件
        data_dir = Path("data")
        if data_dir.exists():
            parquet_files = list(data_dir.glob("**/*.parquet"))
            if parquet_files:
                latest_file = max(parquet_files, key=os.path.getmtime)
                file_size = latest_file.stat().st_size / (1024*1024)  # MB
                print(f" | 数据: {file_size:.1f}MB", end="")
        
        # 每5分钟输出一次详细状态
        if current_time - last_check >= 300:  # 5分钟
            print(f"\n📊 详细状态 [{time.strftime('%H:%M:%S')}]:")
            print(f"  运行时长: {elapsed/60:.1f}分钟")
            print(f"  CPU: {cpu_percent:.1f}%")
            print(f"  内存: {memory.percent:.1f}% ({memory.available/(1024**3):.1f}GB可用)")
            
            # 检查数据文件
            if data_dir.exists():
                parquet_files = list(data_dir.glob("**/*.parquet"))
                if parquet_files:
                    latest_file = max(parquet_files, key=os.path.getmtime)
                    file_size = latest_file.stat().st_size / (1024*1024)
                    print(f"  数据文件: {latest_file.name} ({file_size:.1f}MB)")
            
            last_check = current_time
        
        time.sleep(10)  # 每10秒检查一次

if __name__ == "__main__":
    try:
        monitor_test()
    except KeyboardInterrupt:
        print("\n\n🛑 监控已停止")
