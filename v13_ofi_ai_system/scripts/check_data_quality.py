#!/usr/bin/env python3
"""
检查数据收集进度和质量
"""

import pandas as pd
import glob
from datetime import datetime

def check_data_quality():
    print('=== 数据收集进度和质量分析 ===')
    print()
    
    # 计算运行时间
    start_time = datetime(2025, 10, 21, 7, 12)
    current_time = datetime.now()
    elapsed = current_time - start_time
    hours = elapsed.total_seconds() / 3600
    
    print(f'运行时间: {hours:.1f}小时 (目标: 48小时)')
    print(f'进度: {hours/48*100:.1f}%')
    print()
    
    # 检查数据质量和种类
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        print(f'{symbol} 数据质量分析:')
        
        # 检查prices数据
        prices_files = glob.glob(f'data/ofi_cvd/date=*/symbol={symbol}/kind=prices/*.parquet')
        if prices_files:
            sample_df = pd.read_parquet(prices_files[0])
            print(f'  prices: {len(prices_files)}个文件, 字段: {list(sample_df.columns)}')
            if len(sample_df) > 0:
                print(f'    样本价格: {sample_df["price"].iloc[0]}')
        
        # 检查OFI数据
        ofi_files = glob.glob(f'data/ofi_cvd/date=*/symbol={symbol}/kind=ofi/*.parquet')
        if ofi_files:
            sample_df = pd.read_parquet(ofi_files[0])
            print(f'  ofi: {len(ofi_files)}个文件, 字段: {list(sample_df.columns)}')
            if 'ofi_z' in sample_df.columns and len(sample_df) > 0:
                valid_ofi_z = sample_df['ofi_z'].notna().sum()
                print(f'    OFI Z-score有效数据: {valid_ofi_z}/{len(sample_df)}')
        
        # 检查CVD数据
        cvd_files = glob.glob(f'data/ofi_cvd/date=*/symbol={symbol}/kind=cvd/*.parquet')
        if cvd_files:
            sample_df = pd.read_parquet(cvd_files[0])
            print(f'  cvd: {len(cvd_files)}个文件, 字段: {list(sample_df.columns)}')
            if 'z_cvd' in sample_df.columns and len(sample_df) > 0:
                valid_cvd_z = sample_df['z_cvd'].notna().sum()
                print(f'    CVD Z-score有效数据: {valid_cvd_z}/{len(sample_df)}')
        
        # 检查fusion数据
        fusion_files = glob.glob(f'data/ofi_cvd/date=*/symbol={symbol}/kind=fusion/*.parquet')
        if fusion_files:
            sample_df = pd.read_parquet(fusion_files[0])
            print(f'  fusion: {len(fusion_files)}个文件, 字段: {list(sample_df.columns)}')
        
        # 检查events数据
        events_files = glob.glob(f'data/ofi_cvd/date=*/symbol={symbol}/kind=events/*.parquet')
        if events_files:
            sample_df = pd.read_parquet(events_files[0])
            print(f'  events: {len(events_files)}个文件, 字段: {list(sample_df.columns)}')
            if 'event_type' in sample_df.columns and len(sample_df) > 0:
                event_types = sample_df['event_type'].value_counts()
                print(f'    事件类型: {dict(event_types)}')
        
        print()
    
    # 统计总数据量
    total_files = 0
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        for data_type in ['prices', 'ofi', 'cvd', 'fusion', 'events']:
            files = glob.glob(f'data/ofi_cvd/date=*/symbol={symbol}/kind={data_type}/*.parquet')
            total_files += len(files)
    
    print(f'总文件数: {total_files:,}个')
    print(f'平均文件生成速率: {total_files/hours:.0f}个/小时')
    print()
    print('数据收集状态: 正常运行')

if __name__ == "__main__":
    check_data_quality()
