#!/usr/bin/env python3
"""
检查数据收集状态
"""

import os
import glob
from datetime import datetime

def check_data_status():
    """检查数据收集状态"""
    print("=" * 60)
    print("数据收集状态检查")
    print("=" * 60)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查数据目录
    data_root = "data/ofi_cvd"
    if not os.path.exists(data_root):
        print("[ERROR] 数据目录不存在")
        return
    
    # 检查日期目录
    date_dirs = glob.glob(f"{data_root}/date=*")
    if not date_dirs:
        print("[ERROR] 没有找到日期目录")
        return
    
    print(f"[DATE] 找到日期目录: {len(date_dirs)} 个")
    for date_dir in date_dirs:
        date_name = os.path.basename(date_dir)
        print(f"  - {date_name}")
    
    print()
    
    # 检查最新日期的数据
    latest_date = max(date_dirs, key=os.path.getmtime)
    date_name = os.path.basename(latest_date)
    print(f"[LATEST] 最新数据日期: {date_name}")
    
    # 检查交易对
    symbol_dirs = glob.glob(f"{latest_date}/symbol=*")
    if not symbol_dirs:
        print("[ERROR] 没有找到交易对目录")
        return
    
    print(f"[SYMBOL] 交易对: {len(symbol_dirs)} 个")
    for symbol_dir in symbol_dirs:
        symbol_name = os.path.basename(symbol_dir)
        print(f"  - {symbol_name}")
    
    print()
    
    # 检查每个交易对的数据
    for symbol_dir in symbol_dirs:
        symbol_name = os.path.basename(symbol_dir)
        print(f"[DATA] {symbol_name} 数据统计:")
        
        # 检查数据类型
        data_types = ['prices', 'ofi', 'cvd', 'fusion', 'events']
        for data_type in data_types:
            type_dir = f"{symbol_dir}/kind={data_type}"
            if os.path.exists(type_dir):
                parquet_files = glob.glob(f"{type_dir}/*.parquet")
                file_count = len(parquet_files)
                
                # 获取最新文件时间
                if parquet_files:
                    latest_file = max(parquet_files, key=os.path.getmtime)
                    latest_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
                    print(f"  {data_type}: {file_count} 个文件, 最新: {latest_time.strftime('%H:%M:%S')}")
                else:
                    print(f"  {data_type}: 0 个文件")
            else:
                print(f"  {data_type}: 目录不存在")
        
        print()
    
    # 检查进程状态
    print("[PROCESS] 进程状态:")
    try:
        import psutil
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'harvest' in cmdline or 'run_success' in cmdline:
                        python_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if python_processes:
            print("  [OK] 发现数据收集进程:")
            for proc in python_processes:
                print(f"    PID: {proc.pid}, 命令: {' '.join(proc.cmdline()[:3])}")
        else:
            print("  [NO] 没有发现数据收集进程")
    except ImportError:
        print("  [WARN] 无法检查进程状态 (需要安装psutil)")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    check_data_status()
