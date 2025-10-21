#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据采集质量监控脚本
检查数据完整性、稳定性和质量指标
"""

import pandas as pd
import glob
import os
from datetime import datetime, timedelta
import json

def check_data_quality():
    """检查数据质量"""
    print("=" * 60)
    print("📊 数据采集质量监控报告")
    print("=" * 60)
    
    # 1. 文件统计
    print("\n📁 文件统计:")
    kinds = ['prices', 'ofi', 'cvd', 'fusion', 'events']
    for kind in kinds:
        files = glob.glob(f'data/ofi_cvd/date=*/symbol=*/kind={kind}/*.parquet')
        print(f"  {kind:8}: {len(files):3} 个文件")
    
    # 2. 数据量统计
    print("\n📈 数据量统计:")
    prices_files = glob.glob('data/ofi_cvd/date=*/symbol=*/kind=prices/*.parquet')
    cvd_files = glob.glob('data/ofi_cvd/date=*/symbol=*/kind=cvd/*.parquet')
    
    if prices_files:
        df_prices = pd.concat([pd.read_parquet(f) for f in prices_files])
        print(f"  总交易记录: {len(df_prices):,} 条")
        print(f"  交易对: {', '.join(df_prices['symbol'].unique())}")
        
        # 时间范围
        min_time = pd.to_datetime(df_prices['ts_ms'], unit='ms').min()
        max_time = pd.to_datetime(df_prices['ts_ms'], unit='ms').max()
        duration = max_time - min_time
        print(f"  时间范围: {min_time} - {max_time}")
        print(f"  采集时长: {duration}")
        
        # 价格范围
        print(f"  价格范围: {df_prices['price'].min():.2f} - {df_prices['price'].max():.2f}")
        
        # 按交易对统计
        print("\n📊 按交易对统计:")
        for symbol in df_prices['symbol'].unique():
            symbol_data = df_prices[df_prices['symbol'] == symbol]
            print(f"  {symbol}: {len(symbol_data):,} 条记录")
    
    if cvd_files:
        df_cvd = pd.concat([pd.read_parquet(f) for f in cvd_files])
        print(f"\n📊 CVD数据统计:")
        print(f"  总CVD记录: {len(df_cvd):,} 条")
        print(f"  CVD范围: {df_cvd['cvd'].min():.6f} - {df_cvd['cvd'].max():.6f}")
        print(f"  Z-score范围: {df_cvd['z_cvd'].min():.6f} - {df_cvd['z_cvd'].max():.6f}")
    
    # 3. 数据完整性检查
    print("\n🔍 数据完整性检查:")
    
    # 检查是否有OFI数据
    ofi_files = glob.glob('data/ofi_cvd/date=*/symbol=*/kind=ofi/*.parquet')
    if ofi_files:
        print("  ✅ OFI数据: 已生成")
    else:
        print("  ❌ OFI数据: 未生成 (可能订单簿流未连接)")
    
    # 检查是否有Fusion数据
    fusion_files = glob.glob('data/ofi_cvd/date=*/symbol=*/kind=fusion/*.parquet')
    if fusion_files:
        print("  ✅ Fusion数据: 已生成")
    else:
        print("  ❌ Fusion数据: 未生成 (需要OFI数据)")
    
    # 检查是否有Events数据
    events_files = glob.glob('data/ofi_cvd/date=*/symbol=*/kind=events/*.parquet')
    if events_files:
        print("  ✅ Events数据: 已生成")
    else:
        print("  ❌ Events数据: 未生成 (需要完整指标数据)")
    
    # 4. 系统稳定性检查
    print("\n⚡ 系统稳定性检查:")
    
    if prices_files:
        # 检查文件生成频率
        file_times = []
        for f in prices_files:
            file_time = os.path.getmtime(f)
            file_times.append(file_time)
        
        file_times.sort()
        if len(file_times) > 1:
            intervals = [file_times[i+1] - file_times[i] for i in range(len(file_times)-1)]
            avg_interval = sum(intervals) / len(intervals)
            print(f"  平均文件生成间隔: {avg_interval:.1f} 秒")
            
            if avg_interval < 30:
                print("  ✅ 文件生成频率正常")
            else:
                print("  ⚠️  文件生成频率较慢")
    
    # 5. 数据质量评分
    print("\n📊 数据质量评分:")
    
    score = 0
    max_score = 100
    
    # 基础数据 (40分)
    if prices_files and len(prices_files) >= 5:
        score += 20
        print("  ✅ 基础交易数据: 20/20")
    else:
        print("  ❌ 基础交易数据: 0/20")
    
    if cvd_files and len(cvd_files) >= 5:
        score += 20
        print("  ✅ CVD数据: 20/20")
    else:
        print("  ❌ CVD数据: 0/20")
    
    # 高级数据 (30分)
    if ofi_files:
        score += 15
        print("  ✅ OFI数据: 15/15")
    else:
        print("  ❌ OFI数据: 0/15")
    
    if fusion_files:
        score += 15
        print("  ✅ Fusion数据: 15/15")
    else:
        print("  ❌ Fusion数据: 0/15")
    
    # 系统稳定性 (30分)
    if prices_files and len(prices_files) >= 5:
        score += 30
        print("  ✅ 系统稳定性: 30/30")
    else:
        print("  ❌ 系统稳定性: 0/30")
    
    print(f"\n🎯 总体评分: {score}/{max_score} ({score/max_score*100:.1f}%)")
    
    if score >= 80:
        print("  🎉 数据质量优秀！")
    elif score >= 60:
        print("  ✅ 数据质量良好")
    elif score >= 40:
        print("  ⚠️  数据质量一般，需要改进")
    else:
        print("  ❌ 数据质量较差，需要修复")
    
    # 6. 建议
    print("\n💡 建议:")
    if not ofi_files:
        print("  - 检查订单簿流连接是否正常")
        print("  - 验证WebSocket连接稳定性")
    if not fusion_files:
        print("  - 需要等待OFI数据生成")
    if not events_files:
        print("  - 需要等待完整指标数据")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_data_quality()

