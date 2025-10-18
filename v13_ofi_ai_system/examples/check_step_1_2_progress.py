#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1.2 兜底方案测试进度监控脚本
"""

import os
import sys
import time
import json
import pandas as pd
from pathlib import Path
import io

# Windows Unicode 支持
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_step_1_2_progress():
    """检查Step 1.2测试进度"""
    
    # 输出目录
    output_dir = Path("../data/cvd_step_1_2_fallback_ethusdt")
    
    print("🔍 Step 1.2 兜底方案测试进度监控")
    print("=" * 60)
    print(f"📁 输出目录: {output_dir}")
    print(f"⏰ 检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查目录是否存在
    if not output_dir.exists():
        print("❌ 输出目录不存在，测试可能尚未开始")
        return
    
    # 查找最新的parquet文件
    parquet_files = list(output_dir.glob("*.parquet"))
    if not parquet_files:
        print("⏳ 尚未生成数据文件，测试可能正在启动...")
        return
    
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    print(f"📄 最新数据文件: {latest_file.name}")
    
    # 读取数据
    try:
        df = pd.read_parquet(latest_file)
        print(f"📊 数据点数: {len(df):,}")
        
        # 计算时间跨度
        if 'timestamp' in df.columns and len(df) > 0:
            time_span = (df['timestamp'].max() - df['timestamp'].min()) / 1000
            print(f"⏱️  时间跨度: {time_span:.1f} 秒 ({time_span/60:.1f} 分钟)")
        
        # 检查Z-score质量
        if 'z_cvd' in df.columns:
            z_data = df['z_cvd'].dropna()
            if len(z_data) > 0:
                print(f"📈 Z-score统计:")
                print(f"   - 中位数: {z_data.median():.6f}")
                print(f"   - median(|Z|): {z_data.abs().median():.6f}")
                print(f"   - P95(|Z|): {z_data.abs().quantile(0.95):.3f}")
                print(f"   - P99(|Z|): {z_data.abs().quantile(0.99):.3f}")
                
                # 计算目标指标
                p_gt2 = (z_data.abs() > 2).mean() * 100
                p_gt3 = (z_data.abs() > 3).mean() * 100
                print(f"   - P(|Z|>2): {p_gt2:.2f}% (目标: ≤8%)")
                print(f"   - P(|Z|>3): {p_gt3:.2f}% (目标: ≤2%)")
                
                # 状态评估
                print(f"\n🎯 Step 1.2 目标评估:")
                median_ok = z_data.abs().median() <= 1.0
                p_gt2_ok = p_gt2 <= 8.0
                p_gt3_ok = p_gt3 <= 2.0
                
                print(f"   - median(|Z|) ≤ 1.0: {'✅' if median_ok else '❌'}")
                print(f"   - P(|Z|>2) ≤ 8%: {'✅' if p_gt2_ok else '❌'}")
                print(f"   - P(|Z|>3) ≤ 2%: {'✅' if p_gt3_ok else '❌'}")
                
                if median_ok and p_gt2_ok and p_gt3_ok:
                    print(f"\n🎉 Step 1.2 兜底方案成功！所有目标指标达标！")
                else:
                    print(f"\n⚠️  Step 1.2 兜底方案部分达标，需要进一步优化")
        
        # 检查数据质量
        if 'agg_dup_count' in df.columns:
            dup_count = df['agg_dup_count'].iloc[-1] if len(df) > 0 else 0
            backward_count = df['agg_backward_count'].iloc[-1] if len(df) > 0 else 0
            print(f"\n🔍 数据质量:")
            print(f"   - aggTradeId重复: {dup_count}")
            print(f"   - aggTradeId倒序: {backward_count}")
        
    except Exception as e:
        print(f"❌ 读取数据时出错: {e}")
    
    print(f"\n⏰ 下次检查: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 30))}")

if __name__ == "__main__":
    check_step_1_2_progress()
