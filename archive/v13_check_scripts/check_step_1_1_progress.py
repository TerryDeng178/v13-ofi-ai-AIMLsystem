#!/usr/bin/env python3
"""
Step 1.1 修复测试进度监控脚本
"""

import sys
import io
import os
import time
import json
import pandas as pd
from pathlib import Path

# Windows Unicode支持
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_test_progress():
    """检查Step 1.1测试进度"""
    
    # 输出目录
    output_dir = Path("../data/cvd_step_1_1_fix_ethusdt")
    
    print("🔍 Step 1.1 修复测试进度监控")
    print("=" * 50)
    
    if not output_dir.exists():
        print("❌ 输出目录不存在，测试可能未开始")
        return
    
    # 查找最新的parquet文件
    parquet_files = list(output_dir.glob("*.parquet"))
    if not parquet_files:
        print("⏳ 等待数据文件生成...")
        return
    
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 最新数据文件: {latest_file.name}")
    
    try:
        # 读取数据
        df = pd.read_parquet(latest_file)
        total_records = len(df)
        
        print(f"📊 当前记录数: {total_records}")
        
        # 计算时间跨度
        if 'timestamp' in df.columns and len(df) > 1:
            time_span = (df['timestamp'].max() - df['timestamp'].min()) / 1000
            print(f"⏰ 时间跨度: {time_span:.1f} 秒 ({time_span/60:.1f} 分钟)")
        
        # 检查Z-score质量（如果有数据）
        if 'z_cvd' in df.columns and len(df) > 0:
            z_data = df['z_cvd'].dropna()
            if len(z_data) > 0:
                z_abs = z_data.abs()
                p_gt2 = (z_abs > 2).mean() * 100
                p_gt3 = (z_abs > 3).mean() * 100
                median_abs = z_abs.median()
                
                print(f"📈 Z-score质量预览:")
                print(f"   median(|Z|): {median_abs:.4f}")
                print(f"   P(|Z|>2): {p_gt2:.2f}%")
                print(f"   P(|Z|>3): {p_gt3:.2f}%")
                
                # 目标检查
                target_p2 = p_gt2 <= 8.0
                target_p3 = p_gt3 <= 2.0
                target_median = median_abs <= 1.0
                
                print(f"🎯 目标达成情况:")
                print(f"   P(|Z|>2) ≤ 8%: {'✅' if target_p2 else '❌'} ({p_gt2:.2f}%)")
                print(f"   P(|Z|>3) ≤ 2%: {'✅' if target_p3 else '❌'} ({p_gt3:.2f}%)")
                print(f"   median(|Z|) ≤ 1.0: {'✅' if target_median else '❌'} ({median_abs:.4f})")
        
        # 检查基础指标
        if 'meta' in df.columns:
            meta_data = df['meta'].apply(lambda x: x if isinstance(x, dict) else {})
            if len(meta_data) > 0:
                warmup_count = meta_data.apply(lambda x: x.get('warmup', False)).sum()
                std_zero_count = meta_data.apply(lambda x: x.get('std_zero', False)).sum()
                
                print(f"🔧 基础指标:")
                print(f"   Warmup记录: {warmup_count}")
                print(f"   Std_zero记录: {std_zero_count}")
        
        # 预计完成时间
        if total_records > 0:
            # 基于之前的测试，大约每分钟120-130条记录
            estimated_rate = 125  # 记录/分钟
            target_records = 2400  # 20分钟目标
            remaining_records = max(0, target_records - total_records)
            estimated_minutes = remaining_records / estimated_rate
            
            if remaining_records > 0:
                print(f"⏱️  预计还需: {estimated_minutes:.1f} 分钟")
            else:
                print("✅ 测试可能已完成")
        
    except Exception as e:
        print(f"❌ 读取数据时出错: {e}")
    
    print("=" * 50)

if __name__ == "__main__":
    check_test_progress()
