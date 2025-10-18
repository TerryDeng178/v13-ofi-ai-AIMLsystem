#!/usr/bin/env python3
"""
P1.2 Delta-Z微调测试进度监控脚本
监控ETHUSDT 20分钟测试的实时进度
"""

import os
import sys
import time
import json
import io
from pathlib import Path

# Windows Unicode兼容性
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_p1_2_progress():
    """检查P1.2测试进度"""
    print("🔍 P1.2 Delta-Z微调测试进度监控")
    print("=" * 50)
    
    # 检查数据目录
    data_dir = Path("../data/cvd_p1_2_ethusdt")
    if not data_dir.exists():
        print("❌ 数据目录不存在，测试可能未开始")
        return
    
    # 查找最新的parquet文件
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        print("⏳ 等待数据文件生成...")
        return
    
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    file_size = latest_file.stat().st_size
    file_time = time.ctime(latest_file.stat().st_mtime)
    
    print(f"📁 最新数据文件: {latest_file.name}")
    print(f"📊 文件大小: {file_size:,} bytes")
    print(f"⏰ 更新时间: {file_time}")
    
    # 检查JSON报告
    json_files = list(data_dir.glob("*.json"))
    if json_files:
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        try:
            with open(latest_json, 'r') as f:
                report = json.load(f)
            
            print(f"\n📈 测试报告: {latest_json.name}")
            print(f"⏱️  运行时长: {report.get('elapsed_seconds', 0):.1f} 秒")
            print(f"📊 记录数量: {report.get('records_collected', 0):,}")
            
            # 显示关键指标
            metrics = report.get('final_metrics', {})
            print(f"\n🎯 关键指标:")
            print(f"  - 重连次数: {metrics.get('reconnect_count', 0)}")
            print(f"  - 队列丢弃: {metrics.get('queue_dropped', 0)}")
            print(f"  - 解析错误: {metrics.get('parse_errors', 0)}")
            print(f"  - 丢弃率: {metrics.get('queue_dropped_rate', 0):.2%}")
            
            # 显示P1.2微调参数
            print(f"\n🔧 P1.2微调参数:")
            print(f"  - HALF_LIFE_TRADES: 200 (从300降至200)")
            print(f"  - WINSOR_LIMIT: 6.0 (从8.0降至6.0)")
            print(f"  - STALE_THRESHOLD_MS: 3000 (从5000降至3000)")
            print(f"  - FREEZE_MIN: 60 (从50升至60)")
            
        except Exception as e:
            print(f"❌ 读取报告失败: {e}")
    
    # 检查是否完成
    if file_size > 0:
        print(f"\n✅ 测试进行中...")
        print(f"🎯 目标: P(|Z|>3) ≤ 2%")
        print(f"📊 当前状态: 数据收集中")

if __name__ == "__main__":
    check_p1_2_progress()
