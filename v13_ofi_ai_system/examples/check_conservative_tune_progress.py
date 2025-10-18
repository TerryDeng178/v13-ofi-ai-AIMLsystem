#!/usr/bin/env python3
"""
保守微调试验矩阵监控脚本
监控Step A和Step B的测试进度，重点关注"分母"和Z-score质量
"""

import os
import sys
import time
import json
import io
import numpy as np
from pathlib import Path

# Windows Unicode兼容性
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_conservative_tune_progress(test_group="A1"):
    """检查保守微调测试进度"""
    print(f"🔍 保守微调试验矩阵监控 - {test_group}")
    print("=" * 60)
    
    # 根据测试组确定输出目录
    if test_group == "A1":
        output_dir = Path("../data/cvd_p1_1_rollback")
        config_name = "回滚基线（300/8/5000/50）"
    elif test_group == "A2":
        output_dir = Path("../data/cvd_step_a_tune")
        config_name = "半衰期280（其余同A1）"
    elif test_group == "B1":
        output_dir = Path("../data/cvd_step_b_tune")
        config_name = "A2 + 稳健尺度（双EWMA或MAD地板）"
    else:
        print(f"❌ 未知测试组: {test_group}")
        return
    
    print(f"📋 测试配置: {config_name}")
    
    # 检查输出目录
    if not output_dir.exists():
        print("❌ 输出目录不存在，测试可能尚未开始")
        return
    
    # 查找最新的parquet文件
    parquet_files = list(output_dir.glob("*.parquet"))
    if not parquet_files:
        print("⏳ 等待数据文件生成...")
        return
    
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 最新数据文件: {latest_file.name}")
    
    # 检查文件大小和修改时间
    file_size = latest_file.stat().st_size
    mod_time = latest_file.stat().st_mtime
    current_time = time.time()
    age_seconds = current_time - mod_time
    
    print(f"📊 文件大小: {file_size:,} bytes")
    print(f"⏰ 最后更新: {age_seconds:.1f} 秒前")
    
    # 尝试读取parquet文件获取记录数
    try:
        import pandas as pd
        df = pd.read_parquet(latest_file)
        record_count = len(df)
        print(f"📈 当前记录数: {record_count:,}")
        
        # 检查是否有Z-score数据
        if 'z_cvd' in df.columns:
            z_data = df['z_cvd'].dropna()
            if len(z_data) > 0:
                median_z = abs(z_data).median()
                p_z_gt_2 = (abs(z_data) > 2).mean() * 100
                p_z_gt_3 = (abs(z_data) > 3).mean() * 100
                
                print(f"🎯 Z-score质量:")
                print(f"   median(|Z|): {median_z:.4f}")
                print(f"   P(|Z|>2): {p_z_gt_2:.2f}%")
                print(f"   P(|Z|>3): {p_z_gt_3:.2f}%")
                
                # 目标检查
                print(f"🎯 目标达成情况:")
                print(f"   median(|Z|) ≤ 1.0: {'✅' if median_z <= 1.0 else '❌'}")
                print(f"   P(|Z|>2) ≤ 8%: {'✅' if p_z_gt_2 <= 8.0 else '❌'}")
                print(f"   P(|Z|>3) ≤ 3%: {'✅' if p_z_gt_3 <= 3.0 else '❌'}")
                
                # 新增监控：空窗后前3笔Z分布
                if 'meta' in df.columns:
                    try:
                        # 解析meta字段获取warmup信息
                        warmup_mask = df['meta'].apply(lambda x: x.get('warmup', False) if isinstance(x, dict) else False)
                        if warmup_mask.any():
                            warmup_data = z_data[warmup_mask]
                            if len(warmup_data) > 0:
                                warmup_median = abs(warmup_data).median()
                                warmup_p_gt_2 = (abs(warmup_data) > 2).mean() * 100
                                print(f"🔍 空窗后前3笔Z分布:")
                                print(f"   median(|Z|): {warmup_median:.4f}")
                                print(f"   P(|Z|>2): {warmup_p_gt_2:.2f}%")
                    except Exception as e:
                        print(f"⚠️ 解析warmup数据时出错: {e}")
        
        # 检查测试时长
        if 'timestamp' in df.columns:
            time_span = (df['timestamp'].max() - df['timestamp'].min()) / 1000
            print(f"⏱️ 测试时长: {time_span:.1f} 秒 ({time_span/60:.1f} 分钟)")
            
            if time_span >= 1200:  # 20分钟
                print("✅ 测试时长达标 (≥20分钟)")
            else:
                remaining = 1200 - time_span
                print(f"⏳ 还需 {remaining:.1f} 秒 ({remaining/60:.1f} 分钟)")
        
        # 新增监控：scale_t（分母）统计
        if 'meta' in df.columns:
            try:
                # 尝试提取scale信息
                scale_values = []
                for meta in df['meta']:
                    if isinstance(meta, dict) and 'scale' in meta:
                        scale_values.append(meta['scale'])
                
                if scale_values:
                    scale_array = np.array(scale_values)
                    scale_p5 = np.percentile(scale_array, 5)
                    scale_p50 = np.percentile(scale_array, 50)
                    scale_p95 = np.percentile(scale_array, 95)
                    
                    print(f"📊 Scale（分母）统计:")
                    print(f"   P5: {scale_p5:.6f}")
                    print(f"   P50: {scale_p50:.6f}")
                    print(f"   P95: {scale_p95:.6f}")
            except Exception as e:
                print(f"⚠️ 解析scale数据时出错: {e}")
        
    except Exception as e:
        print(f"⚠️ 读取数据文件时出错: {e}")
    
    # 检查是否有报告文件
    report_files = list(output_dir.glob("*.json"))
    if report_files:
        latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
        print(f"📋 最新报告: {latest_report.name}")
        
        try:
            with open(latest_report, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            if 'final_metrics' in report_data:
                metrics = report_data['final_metrics']
                print(f"📊 运行指标:")
                print(f"   总消息数: {metrics.get('total_messages', 'N/A'):,}")
                print(f"   解析错误: {metrics.get('parse_errors', 'N/A')}")
                print(f"   队列丢弃: {metrics.get('queue_dropped', 'N/A')}")
                print(f"   重连次数: {metrics.get('reconnect_count', 'N/A')}")
                
                if 'agg_dup_count' in metrics:
                    print(f"   重复ID: {metrics['agg_dup_count']}")
                    print(f"   倒序ID: {metrics.get('agg_backward_count', 'N/A')}")
                    print(f"   延迟事件: {metrics.get('late_event_dropped', 'N/A')}")
                    print(f"   缓冲P95: {metrics.get('buffer_size_p95', 'N/A')}")
                    print(f"   缓冲最大: {metrics.get('buffer_size_max', 'N/A')}")
                
                # 新增监控：常规健康项
                print(f"🔍 常规健康项:")
                print(f"   Z冻结次数: {metrics.get('z_freeze_count', 'N/A')}")
                print(f"   10秒以上缺口: {metrics.get('gaps_over_10s', 'N/A')}")
                print(f"   P99到达间隔: {metrics.get('p99_interarrival', 'N/A')}ms")
        
        except Exception as e:
            print(f"⚠️ 读取报告文件时出错: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    import sys
    test_group = sys.argv[1] if len(sys.argv) > 1 else "A1"
    check_conservative_tune_progress(test_group)
