#!/usr/bin/env python3
"""
48小时数据收集监控脚本
实时监控数据收集进度和状态
"""

import os
import time
import glob
import pandas as pd
from datetime import datetime, timedelta
import json

def get_data_stats():
    """获取数据收集统计信息"""
    stats = {}
    
    # 检查数据目录
    data_dir = "data/ofi_cvd"
    if not os.path.exists(data_dir):
        return {"error": "数据目录不存在"}
    
    # 统计各类型数据
    data_types = ['prices', 'ofi', 'cvd', 'fusion', 'events']
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    for symbol in symbols:
        stats[symbol] = {}
        for data_type in data_types:
            files = glob.glob(f"{data_dir}/date=*/symbol={symbol}/kind={data_type}/*.parquet")
            if files:
                total_rows = 0
                for file in files:
                    try:
                        df = pd.read_parquet(file)
                        total_rows += len(df)
                    except:
                        pass
                stats[symbol][data_type] = {
                    'files': len(files),
                    'rows': total_rows,
                    'latest_file': max(files, key=os.path.getmtime) if files else None
                }
            else:
                stats[symbol][data_type] = {'files': 0, 'rows': 0, 'latest_file': None}
    
    return stats

def get_log_info():
    """获取日志信息"""
    log_dir = "artifacts/run_logs"
    if not os.path.exists(log_dir):
        return {"error": "日志目录不存在"}
    
    log_files = glob.glob(f"{log_dir}/harvest_48h_*.log")
    if not log_files:
        return {"error": "未找到48小时收集日志"}
    
    latest_log = max(log_files, key=os.path.getmtime)
    
    # 读取最后几行日志
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            last_lines = lines[-10:] if len(lines) >= 10 else lines
        return {
            'log_file': latest_log,
            'last_lines': [line.strip() for line in last_lines],
            'file_size': os.path.getsize(latest_log)
        }
    except Exception as e:
        return {"error": f"读取日志失败: {e}"}

def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def main():
    """主监控循环"""
    print("=" * 60)
    print("48小时OFI+CVD数据收集监控")
    print("=" * 60)
    print(f"监控开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("按 Ctrl+C 停止监控")
    print("=" * 60)
    
    try:
        while True:
            # 清屏（Windows）
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 60)
            print("48小时OFI+CVD数据收集监控")
            print("=" * 60)
            print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # 获取数据统计
            print("📊 数据收集统计:")
            stats = get_data_stats()
            
            if 'error' in stats:
                print(f"❌ {stats['error']}")
            else:
                for symbol, data in stats.items():
                    print(f"\n🔸 {symbol}:")
                    for data_type, info in data.items():
                        if info['rows'] > 0:
                            latest_time = ""
                            if info['latest_file']:
                                mtime = os.path.getmtime(info['latest_file'])
                                latest_time = f" (最新: {datetime.fromtimestamp(mtime).strftime('%H:%M:%S')})"
                            print(f"  {data_type}: {info['rows']}行, {info['files']}文件{latest_time}")
                        else:
                            print(f"  {data_type}: 无数据")
            
            # 获取日志信息
            print("\n📝 日志信息:")
            log_info = get_log_info()
            
            if 'error' in log_info:
                print(f"❌ {log_info['error']}")
            else:
                print(f"日志文件: {log_info['log_file']}")
                print(f"文件大小: {format_size(log_info['file_size'])}")
                print("\n最新日志:")
                for line in log_info['last_lines']:
                    if line:
                        print(f"  {line}")
            
            print("\n" + "=" * 60)
            print("下次更新: 30秒后 (按 Ctrl+C 停止)")
            
            # 等待30秒
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")
    except Exception as e:
        print(f"\n监控出错: {e}")

if __name__ == "__main__":
    main()

