#!/usr/bin/env python3
"""
BTCUSDT压力测试监控脚本
监控高频交易对下的P0-B修复效果
"""

import sys
import os
import time
import json
from pathlib import Path
import pandas as pd
import io

# Windows Unicode支持
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

def check_btcusdt_progress():
    """检查BTCUSDT压力测试进度"""
    data_dir = Path("v13_ofi_ai_system/data/cvd_p0b_btcusdt_pressure")
    
    print("🔍 BTCUSDT压力测试监控")
    print("=" * 50)
    
    # 检查数据文件
    parquet_files = list(data_dir.glob("*.parquet"))
    json_files = list(data_dir.glob("*.json"))
    
    if not parquet_files:
        print("⏳ 测试进行中，数据文件尚未生成...")
        return
    
    # 读取最新数据
    latest_parquet = max(parquet_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 数据文件: {latest_parquet.name}")
    
    try:
        df = pd.read_parquet(latest_parquet)
        print(f"📈 当前数据点数: {len(df):,}")
        
        if len(df) > 0:
            # 时间跨度
            time_span = (df['timestamp'].max() - df['timestamp'].min()) / 60
            print(f"⏱️  时间跨度: {time_span:.1f} 分钟")
            
            # 消息频率
            if time_span > 0:
                freq = len(df) / time_span
                print(f"📊 消息频率: {freq:.1f} 条/分钟")
            
            # 关键指标
            print("\n🎯 关键指标:")
            print(f"  - agg_dup_count: {df.get('agg_dup_count', [0]).iloc[-1] if 'agg_dup_count' in df.columns else 'N/A'}")
            print(f"  - agg_backward_count: {df.get('agg_backward_count', [0]).iloc[-1] if 'agg_backward_count' in df.columns else 'N/A'}")
            print(f"  - late_event_dropped: {df.get('late_event_dropped', [0]).iloc[-1] if 'late_event_dropped' in df.columns else 'N/A'}")
            
            # 延迟统计
            if 'latency_ms' in df.columns:
                p95_latency = df['latency_ms'].quantile(0.95)
                print(f"  - 延迟P95: {p95_latency:.1f}ms")
            
            # 水位线健康
            if 'buffer_size_p95' in df.columns:
                buffer_p95 = df['buffer_size_p95'].iloc[-1]
                buffer_max = df['buffer_size_max'].iloc[-1]
                print(f"  - 水位线P95: {buffer_p95}")
                print(f"  - 水位线Max: {buffer_max}")
            
            # CVD连续性检查（抽样）
            if len(df) > 100:
                sample_df = df.sample(min(1000, len(df)))
                cvd_diffs = sample_df['cvd'].diff().dropna()
                continuity_errors = (abs(cvd_diffs) > 1e-6).sum()
                print(f"  - 连续性错误(抽样): {continuity_errors}/1000")
        
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
    
    # 检查JSON报告
    if json_files:
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        try:
            with open(latest_json, 'r') as f:
                report = json.load(f)
            
            print(f"\n📋 运行报告: {latest_json.name}")
            print(f"  - 总消息数: {report.get('total_messages', 'N/A')}")
            print(f"  - 解析错误: {report.get('parse_errors', 'N/A')}")
            print(f"  - 重连次数: {report.get('reconnect_count', 'N/A')}")
            print(f"  - 队列丢弃率: {report.get('queue_dropped_rate', 'N/A')}")
            
        except Exception as e:
            print(f"❌ 读取报告失败: {e}")

if __name__ == "__main__":
    check_btcusdt_progress()
