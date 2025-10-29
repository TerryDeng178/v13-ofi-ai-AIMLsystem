#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查CVD数据量异常"""

import pandas as pd
import glob

print("="*60)
print("CVD Data Investigation")
print("="*60)

# 加载CVD数据
cvd_files = glob.glob("preview/ofi_cvd/date=*/symbol=*/kind=cvd/*.parquet")
print(f"\n找到 {len(cvd_files)} 个CVD文件")

if cvd_files:
    dfs = []
    for f in cvd_files:
        df = pd.read_parquet(f)
        print(f"文件: {f}, 行数: {len(df)}")
        dfs.append(df)
    
    cvd_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCVD总行数: {len(cvd_df)}")
    
    if not cvd_df.empty:
        print(f"\n时间范围:")
        print(f"  最早: {pd.to_datetime(cvd_df['ts_ms'].min(), unit='ms')}")
        print(f"  最晚: {pd.to_datetime(cvd_df['ts_ms'].max(), unit='ms')}")
        
        print(f"\n按symbol统计:")
        for sym in cvd_df['symbol'].unique():
            sym_data = cvd_df[cvd_df['symbol'] == sym]
            print(f"  {sym}: {len(sym_data)} 行")
            
            if len(sym_data) > 0:
                print(f"    时间间隔: {sym_data['ts_ms'].diff().describe()}")
        
        # 检查是否有特征列
        print(f"\nCVD数据列: {list(cvd_df.columns)}")
        
        # 检查z_cvd, z_raw是否存在
        if 'z_cvd' in cvd_df.columns and 'z_raw' in cvd_df.columns:
            print(f"\nZ-score统计:")
            print(f"  z_raw: mean={cvd_df['z_raw'].mean():.3f}, std={cvd_df['z_raw'].std():.3f}")
            print(f"  z_cvd: mean={cvd_df['z_cvd'].mean():.3f}, std={cvd_df['z_cvd'].std():.3f}")
        
        # 检查sample
        print(f"\n前5条样本:")
        print(cvd_df.head())
else:
    print("未找到CVD文件")

# 对比其他数据量
print("\n" + "="*60)
print("对比其他kind的数据量")
print("="*60)

for kind in ['prices', 'ofi', 'cvd']:
    if kind == 'prices':
        files = glob.glob("data/ofi_cvd/date=*/symbol=*/kind=prices/*.parquet")
    else:
        files = glob.glob(f"preview/ofi_cvd/date=*/symbol=*/kind={kind}/*.parquet")
    
    if files:
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"{kind}: {len(df)} 行")


