#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据采集状态监控脚本
"""

import os
import time
import psutil
from pathlib import Path
import pandas as pd

def check_process():
    """检查Python进程"""
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'create_time', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe':
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'simple_harvest.py' in cmdline:
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'create_time': proc.info['create_time'],
                        'cmdline': cmdline
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return python_processes

def check_data_files():
    """检查数据文件"""
    data_dir = Path("data/ofi_cvd")
    if not data_dir.exists():
        return []
    
    files = list(data_dir.glob("*.parquet"))
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[:10]  # 返回最新的10个文件

def check_log_files():
    """检查日志文件"""
    log_dir = Path("artifacts/run_logs")
    if not log_dir.exists():
        return []
    
    files = list(log_dir.glob("simple_harvest_*.log"))
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[:3]  # 返回最新的3个日志文件

def get_file_size_mb(file_path):
    """获取文件大小（MB）"""
    try:
        return round(file_path.stat().st_size / (1024 * 1024), 2)
    except:
        return 0

def main():
    """主监控循环"""
    print("=" * 60)
    print("OFI+CVD 数据采集监控")
    print("=" * 60)
    
    while True:
        try:
            # 检查进程
            processes = check_process()
            print(f"\n[{time.strftime('%H:%M:%S')}] Python进程状态:")
            if processes:
                for proc in processes:
                    print(f"  PID: {proc['pid']}, 启动时间: {time.ctime(proc['create_time'])}")
            else:
                print("  没有找到运行中的数据采集进程")
            
            # 检查数据文件
            data_files = check_data_files()
            print(f"\n数据文件状态 (最新10个):")
            if data_files:
                total_size = 0
                for file in data_files:
                    size_mb = get_file_size_mb(file)
                    total_size += size_mb
                    print(f"  {file.name}: {size_mb}MB")
                print(f"  总大小: {total_size}MB")
            else:
                print("  没有找到数据文件")
            
            # 检查日志文件
            log_files = check_log_files()
            print(f"\n日志文件状态 (最新3个):")
            if log_files:
                for file in log_files:
                    size_mb = get_file_size_mb(file)
                    print(f"  {file.name}: {size_mb}MB")
            else:
                print("  没有找到日志文件")
            
            print("\n" + "=" * 60)
            print("按 Ctrl+C 停止监控")
            
            # 等待30秒
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception as e:
            print(f"监控错误: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()

