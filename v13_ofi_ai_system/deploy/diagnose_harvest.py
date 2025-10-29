#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断脚本：分析 run_success_harvest.py 运行时的问题
需要在该进程运行时执行此脚本来收集诊断信息
"""
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

def analyze_file_writing_pattern():
    """分析文件写入模式"""
    print("=" * 60)
    print("1. 文件写入模式分析")
    print("=" * 60)
    
    base_dir = Path(__file__).parent / "data" / "ofi_cvd"
    if not base_dir.exists():
        print(f"数据目录不存在: {base_dir}")
        print("请确保采集进程运行中或检查路径")
        return
    
    # 查找所有 Parquet 文件
    pattern = "date=*/symbol=*/kind=*/part-*.parquet"
    files = sorted(base_dir.glob(pattern))
    
    if not files:
        print("未找到任何 Parquet 文件")
        return
    
    print(f"找到 {len(files)} 个 Parquet 文件")
    
    # 按类型分组统计
    kind_stats = {}
    for f in files:
        # 从路径提取信息：date=YYYY-MM-DD/symbol=XXX/kind=YYY/part-xxx.parquet
        parts = f.parts
        if len(parts) >= 3:
            date_part = parts[-4]
            symbol_part = parts[-3]
            kind_part = parts[-2]
            
            date_str = date_part.split('=')[1]
            symbol = symbol_part.split('=')[1]
            kind = kind_part.split('=')[1]
            
            if kind not in kind_stats:
                kind_stats[kind] = {'files': [], 'total_size': 0}
            
            kind_stats[kind]['files'].append(f)
            kind_stats[kind]['total_size'] += f.stat().st_size
    
    print("\n按类型统计:")
    for kind, stats in sorted(kind_stats.items()):
        print(f"  {kind:15s}: {len(stats['files']):4d} 文件, {stats['total_size'] / 1024 / 1024:8.2f} MB")
    
    # 检查最新的文件时间戳
    print("\n最新文件:")
    for kind in ['prices', 'ofi', 'cvd', 'orderbook']:
        if kind in kind_stats and kind_stats[kind]['files']:
            latest = max(kind_stats[kind]['files'], key=lambda f: f.stat().st_mtime)
            mtime = datetime.fromtimestamp(latest.stat().st_mtime)
            age_minutes = (datetime.now() - mtime).total_seconds() / 60
            
            print(f"  {kind:15s}: {latest.name[:60]}")
            print(f"               修改时间 {mtime.strftime('%Y-%m-%d %H:%M:%S')} ({age_minutes:.1f} 分钟前)")
            
            if age_minutes > 5:
                print(f"               [警告] 超过5分钟未更新文件！")
    
    print()

def analyze_data_integrity():
    """分析数据完整性"""
    print("=" * 60)
    print("2. 数据完整性分析")
    print("=" * 60)
    
    base_dir = Path(__file__).parent / "data" / "ofi_cvd"
    
    # 检查 prices 文件
    prices_files = sorted(base_dir.glob("date=*/symbol=*/kind=prices/*.parquet"))
    if not prices_files:
        print("未找到 prices 文件")
        print()
        return
    
    # 读取最新的文件
    latest_prices = prices_files[-1]
    print(f"检查最新 prices 文件: {latest_prices.name}")
    
    try:
        df = pd.read_parquet(latest_prices)
        print(f"  行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        
        if 'ts_ms' in df.columns:
            time_range = df['ts_ms'].max() - df['ts_ms'].min()
            print(f"  时间跨度: {time_range / 1000:.1f} 秒")
            
            latest_time = datetime.fromtimestamp(df['ts_ms'].max() / 1000)
            print(f"  最新数据时间: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            age_minutes = (datetime.now() - latest_time).total_seconds() / 60
            print(f"  数据年龄: {age_minutes:.1f} 分钟")
            
            if age_minutes > 5:
                print(f"  [警告] 数据年龄超过5分钟！")
        
        # 检查 is_buyer_maker 字段
        if 'is_buyer_maker' in df.columns:
            print("  is_buyer_maker: 存在")
            true_count = (df['is_buyer_maker'] == True).sum()
            false_count = (df['is_buyer_maker'] == False).sum()
            print(f"    True: {true_count}, False: {false_count}")
        else:
            print("  is_buyer_maker: 不存在")
    
    except Exception as e:
        print(f"  错误: {e}")
    
    print()

def suggest_diagnosis_steps():
    """建议诊断步骤"""
    print("=" * 60)
    print("3. 诊断建议")
    print("=" * 60)
    
    print("""
1. 监控进程内存使用:
   - 使用 top/htop 或任务管理器
   - 查看是否持续增长

2. 检查进程日志:
   - 查看保存错误信息
   - 查看轮转日志
   - 查看健康检查日志

3. 检查磁盘空间:
   - df -h (Linux) 或磁盘管理 (Windows)
   - 确保有足够空间

4. 检查文件权限:
   - 确保目录可写

5. 添加调试日志:
   - 在 _check_and_rotate_data 中添加缓冲区大小日志
   - 在 _save_data 中添加耗时日志

6. 查看 bug_analysis.md 了解可能的BUGS
    """)

def main():
    print("run_success_harvest.py 诊断工具")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    analyze_file_writing_pattern()
    analyze_data_integrity()
    suggest_diagnosis_steps()

if __name__ == "__main__":
    main()

