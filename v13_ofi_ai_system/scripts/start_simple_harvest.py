#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化数据收集脚本 - 启动ETH和BTC数据收集
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def start_simple_harvest():
    """启动简化的数据收集"""
    print("开始启动简化数据收集...")
    
    # 设置环境变量
    os.environ['SYMBOLS'] = 'BTCUSDT,ETHUSDT'
    os.environ['RUN_HOURS'] = '48'
    os.environ['PARQUET_ROTATE_SEC'] = '60'
    os.environ['WSS_PING_INTERVAL'] = '20'
    os.environ['DEDUP_LRU'] = '10000'
    os.environ['Z_MODE'] = 'rolling'
    os.environ['SCALE_MODE'] = 'mad'
    os.environ['MAD_MULTIPLIER'] = '3.0'
    os.environ['SCALE_FAST_WEIGHT'] = '0.3'
    os.environ['HALF_LIFE_SEC'] = '300'
    os.environ['WINSOR_LIMIT'] = '3.0'
    
    print("环境变量设置完成")
    
    # 启动数据收集
    try:
        print("启动数据收集进程...")
        process = subprocess.Popen([
            sys.executable, 'examples/run_success_harvest.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print(f"数据收集进程已启动，PID: {process.pid}")
        
        # 保存进程信息
        process_info = {
            'pid': process.pid,
            'start_time': datetime.now().isoformat(),
            'status': 'running'
        }
        
        os.makedirs('artifacts/canary', exist_ok=True)
        with open('artifacts/canary/process_info.json', 'w') as f:
            import json
            json.dump(process_info, f, indent=2)
        
        print("进程信息已保存")
        
        # 等待一段时间检查状态
        print("等待数据收集启动...")
        time.sleep(10)
        
        # 检查进程状态
        if process.poll() is None:
            print("数据收集进程运行正常")
            return True
        else:
            print("数据收集进程已停止")
            stdout, stderr = process.communicate()
            print(f"标准输出: {stdout}")
            print(f"错误输出: {stderr}")
            return False
            
    except Exception as e:
        print(f"启动数据收集失败: {e}")
        return False

def check_data_collection():
    """检查数据收集状态"""
    print("\n检查数据收集状态...")
    
    # 检查进程
    try:
        with open('artifacts/canary/process_info.json', 'r') as f:
            import json
            process_info = json.load(f)
        
        pid = process_info['pid']
        print(f"进程PID: {pid}")
        
        # 检查进程是否运行
        import psutil
        try:
            process = psutil.Process(pid)
            if process.is_running():
                print("进程运行正常")
            else:
                print("进程已停止")
                return False
        except psutil.NoSuchProcess:
            print("进程不存在")
            return False
            
    except Exception as e:
        print(f"检查进程失败: {e}")
        return False
    
    # 检查数据文件
    data_dir = 'data/ofi_cvd/date=2025-10-22'
    if not os.path.exists(data_dir):
        print("数据目录不存在")
        return False
    
    print("数据目录存在")
    
    # 检查各符号数据
    symbols = ['BTCUSDT', 'ETHUSDT']
    data_types = ['prices', 'ofi', 'cvd', 'fusion', 'events']
    
    for symbol in symbols:
        print(f"\n检查 {symbol} 数据:")
        symbol_dir = f'{data_dir}/symbol={symbol}'
        
        if not os.path.exists(symbol_dir):
            print(f"  {symbol} 目录不存在")
            continue
        
        for data_type in data_types:
            type_dir = f'{symbol_dir}/kind={data_type}'
            
            if not os.path.exists(type_dir):
                print(f"  {data_type}: 目录不存在")
                continue
            
            # 检查文件
            files = []
            for file in os.listdir(type_dir):
                if file.endswith('.parquet'):
                    files.append(file)
            
            if files:
                print(f"  {data_type}: {len(files)} 个文件")
            else:
                print(f"  {data_type}: 无文件")
    
    return True

def main():
    """主函数"""
    print("=== 简化数据收集启动脚本 ===")
    
    # 启动数据收集
    success = start_simple_harvest()
    
    if success:
        print("\n数据收集启动成功")
        
        # 检查状态
        check_data_collection()
        
        print("\n数据收集已启动，请等待数据生成...")
        print("建议等待5-10分钟后再次检查数据文件")
    else:
        print("\n数据收集启动失败")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
