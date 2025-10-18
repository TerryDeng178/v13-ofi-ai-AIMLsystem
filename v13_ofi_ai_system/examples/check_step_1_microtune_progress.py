#!/usr/bin/env python3
"""
Step 1 微调测试进度监控脚本
监控Step 1微调测试的进度和Z-score质量
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
    """检查Step 1微调测试进度"""
    
    # 输出目录
    output_dir = Path("../data/cvd_step_1_microtune_ethusdt")
    
    print("🔍 Step 1 微调测试进度监控")
    print("=" * 50)
    print(f"输出目录: {output_dir}")
    
    if not output_dir.exists():
        print("❌ 输出目录不存在，测试可能未开始")
        return
    
    # 查找最新的parquet文件
    parquet_files = list(output_dir.glob("*.parquet"))
    if not parquet_files:
        print("❌ 未找到parquet文件，测试可能未开始")
        return
    
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 最新文件: {latest_file.name}")
    
    # 读取数据
    try:
        df = pd.read_parquet(latest_file)
        print(f"📊 记录数: {len(df)}")
        
        # 计算时间跨度
        if 'timestamp' in df.columns:
            time_span = (df['timestamp'].max() - df['timestamp'].min()) / 1000
            print(f"⏱️ 时间跨度: {time_span:.1f}秒")
        
        # Z-score质量分析
        if 'z_cvd' in df.columns:
            df_no_warmup = df[df.get('warmup', False) == False]
            if len(df_no_warmup) > 0:
                z_abs = df_no_warmup['z_cvd'].abs()
                
                # 关键指标
                p_gt2 = (z_abs > 2).mean() * 100
                p_gt3 = (z_abs > 3).mean() * 100
                median_abs = z_abs.median()
                p95 = z_abs.quantile(0.95)
                p99 = z_abs.quantile(0.99)
                
                print("\n📈 Z-score质量指标:")
                print(f"  median(|Z|): {median_abs:.4f}")
                print(f"  P95(|Z|): {p95:.4f}")
                print(f"  P99(|Z|): {p99:.4f}")
                print(f"  P(|Z|>2): {p_gt2:.2f}% {'✅' if p_gt2 <= 8 else '❌'} (目标≤8%)")
                print(f"  P(|Z|>3): {p_gt3:.2f}% {'✅' if p_gt3 <= 2 else '❌'} (目标≤2%)")
                
                # 通过状态
                pass_gt2 = p_gt2 <= 8
                pass_gt3 = p_gt3 <= 2
                pass_median = median_abs <= 1.0
                
                print(f"\n🎯 通过状态:")
                print(f"  median(|Z|)≤1.0: {'✅' if pass_median else '❌'}")
                print(f"  P(|Z|>2)≤8%: {'✅' if pass_gt2 else '❌'}")
                print(f"  P(|Z|>3)≤2%: {'✅' if pass_gt3 else '❌'}")
                
                overall_pass = pass_median and pass_gt2 and pass_gt3
                print(f"\n🏆 总体状态: {'✅ 通过' if overall_pass else '❌ 未达标'}")
                
                # 与Step 1基线对比
                print(f"\n📊 与Step 1基线对比:")
                print(f"  Step 1基线: P95=5.40, P(|Z|>2)≈?, P(|Z|>3)≈?")
                print(f"  当前微调:  P95={p95:.2f}, P(|Z|>2)={p_gt2:.2f}%, P(|Z|>3)={p_gt3:.2f}%")
                
                if p95 < 5.40:
                    print("  ✅ P95改善")
                else:
                    print("  ❌ P95未改善")
                    
            else:
                print("⚠️ 无有效Z-score数据（全部在warmup状态）")
        
        # 数据质量检查
        print(f"\n🔍 数据质量:")
        if 'agg_dup_count' in df.columns:
            dup_count = df['agg_dup_count'].iloc[-1] if len(df) > 0 else 0
            print(f"  重复ID: {dup_count}")
        if 'agg_backward_count' in df.columns:
            backward_count = df['agg_backward_count'].iloc[-1] if len(df) > 0 else 0
            print(f"  倒序ID: {backward_count}")
        
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")

if __name__ == "__main__":
    check_test_progress()
