#!/usr/bin/env python3
"""
检查48小时数据收集状态
"""

import os
import glob
import pandas as pd
from datetime import datetime

def check_harvest_status():
    """检查数据收集状态"""
    print("=" * 60)
    print("48小时OFI+CVD数据收集状态检查")
    print("=" * 60)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查数据目录
    data_dir = "data/ofi_cvd"
    if not os.path.exists(data_dir):
        print("❌ 数据目录不存在")
        return
    
    # 检查各类型数据
    data_types = ['prices', 'ofi', 'cvd', 'fusion', 'events']
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    total_stats = {}
    
    for symbol in symbols:
        print(f"📊 {symbol} 数据统计:")
        symbol_stats = {}
        
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
                
                # 获取最新文件时间
                latest_file = max(files, key=os.path.getmtime)
                latest_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
                
                print(f"  ✅ {data_type}: {total_rows}行, {len(files)}个文件, 最新: {latest_time.strftime('%H:%M:%S')}")
                symbol_stats[data_type] = total_rows
            else:
                print(f"  ❌ {data_type}: 无数据")
                symbol_stats[data_type] = 0
        
        total_stats[symbol] = symbol_stats
        print()
    
    # 计算总数据量
    print("📈 总体统计:")
    total_rows = sum(sum(stats.values()) for stats in total_stats.values())
    print(f"  总数据行数: {total_rows:,}")
    
    # 检查进程状态
    print("\n🔍 进程状态:")
    try:
        import subprocess
        result = subprocess.run(['tasklist', '/fi', 'imagename eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        if 'python.exe' in result.stdout:
            print("  ✅ Python进程正在运行")
        else:
            print("  ❌ 未发现Python进程")
    except:
        print("  ❓ 无法检查进程状态")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_harvest_status()

