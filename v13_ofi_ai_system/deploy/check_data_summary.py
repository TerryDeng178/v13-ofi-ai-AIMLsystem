#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速数据质量检查报告"""

import pandas as pd
import glob
import json
from datetime import datetime

def check_data():
    """快速检查数据质量"""
    print("="*60)
    print("OFI+CVD Data Quality Quick Check")
    print("="*60)
    
    # 加载数据
    raw_dir = "data/ofi_cvd"
    preview_dir = "preview/ofi_cvd"
    
    print(f"\nRaw Dir: {raw_dir}")
    print(f"Preview Dir: {preview_dir}")
    
    # 统计文件
    for kind in ['prices', 'orderbook', 'ofi', 'cvd', 'fusion', 'features']:
        if kind in ['prices', 'orderbook']:
            files = glob.glob(f"{raw_dir}/date=*/symbol=*/kind={kind}/*.parquet")
        else:
            files = glob.glob(f"{preview_dir}/date=*/symbol=*/kind={kind}/*.parquet")
        
        if files:
            dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
            
            print(f"\n{kind.upper()}:")
            print(f"  Files: {len(files)}")
            print(f"  Rows: {len(df)}")
            
            if 'symbol' in df.columns:
                print(f"  Symbols: {df['symbol'].unique().tolist()}")
            
            # 检查去重
            if 'row_id' in df.columns:
                unique = df['row_id'].nunique()
                dup_rate = (len(df) - unique) / len(df)
                print(f"  Duplicate Rate: {dup_rate:.4f}")
            
            # 检查延迟
            if 'latency_ms' in df.columns:
                s = df['latency_ms'].dropna()
                if len(s) > 0:
                    print(f"  Latency (ms): P50={s.quantile(0.5):.0f}, P90={s.quantile(0.9):.0f}, P99={s.quantile(0.99):.0f}")
            
            # 检查场景覆盖
            if 'scenario_2x2' in df.columns:
                scenarios = df['scenario_2x2'].value_counts()
                print(f"  Scenarios: {scenarios.to_dict()}")
                
                required = {'A_H', 'A_L', 'Q_H', 'Q_L'}
                present = set(scenarios.index)
                missing = required - present
                if missing:
                    print(f"  WARNING: Missing scenarios: {missing}")
    
    print("\n" + "="*60)
    print("Check Complete!")
    print("="*60)

if __name__ == "__main__":
    check_data()


