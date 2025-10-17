# -*- coding: utf-8 -*-
import sys
import io
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Windows UTF-8 fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Find latest parquet file
data_dir = Path("v13_ofi_ai_system/data/DEMO-USD")
parquet_files = sorted(data_dir.glob("*.parquet"), key=lambda x: x.stat().st_mtime, reverse=True)

if not parquet_files:
    print("❌ 未找到数据文件")
    sys.exit(1)

latest_file = parquet_files[0]
df = pd.read_parquet(latest_file)

elapsed_sec = (df['ts'].iloc[-1] - df['ts'].iloc[0]) / 1000
rate = len(df) / elapsed_sec
elapsed_min = elapsed_sec / 60

# 计算距离目标的进度
target_duration_sec = 2 * 3600  # 2小时
target_points = 300000
progress_pct = (elapsed_sec / target_duration_sec) * 100
points_progress_pct = (len(df) / target_points) * 100

# 预估最终结果
estimated_final_points = int(rate * target_duration_sec)
estimated_final_size_mb = (latest_file.stat().st_size / (1024**2)) * (target_duration_sec / elapsed_sec)

# 剩余时间
remaining_sec = target_duration_sec - elapsed_sec
remaining_min = remaining_sec / 60
eta = datetime.now() + timedelta(seconds=remaining_sec)

print("=" * 70)
print("📊 Task 1.2.5 - 2小时正式测试 进度报告")
print("=" * 70)
print(f"📁 数据文件: {latest_file.name}")
print(f"📏 文件大小: {latest_file.stat().st_size / (1024**2):.2f} MB")
print()
print("🎯 当前进度:")
print(f"  ⏱️  运行时间: {elapsed_min:.1f} 分钟 / 120 分钟 ({progress_pct:.1f}%)")
print(f"  📈 数据点数: {len(df):,} / {target_points:,} ({points_progress_pct:.1f}%)")
print(f"  🚀 采集速率: {rate:.1f} 点/秒 (目标: 50 点/秒)")
print()
print("📉 性能指标:")
print(f"  ⚡ 延迟 p50: {df['latency_ms'].quantile(0.5):.3f} ms")
print(f"  ⚡ 延迟 p95: {df['latency_ms'].quantile(0.95):.3f} ms")
print(f"  ⚡ 延迟 p99: {df['latency_ms'].quantile(0.99):.3f} ms")
print(f"  🔄 队列丢弃: {df['queue_dropped'].iloc[-1]:.0f}")
print(f"  🔌 重连次数: {df['reconnect_count'].iloc[-1]:.0f}")
print()
print("🔮 预估结果:")
print(f"  📊 预估总点数: {estimated_final_points:,} 点")
print(f"  📏 预估文件大小: {estimated_final_size_mb:.2f} MB")
print(f"  ✅ 达标状态: {'✅ 预计达标' if estimated_final_points >= target_points else '❌ 预计不达标'}")
print(f"  📈 超出目标: {(estimated_final_points - target_points) / target_points * 100:+.1f}%")
print()
print("⏳ 剩余时间:")
print(f"  ⏰ 剩余: {remaining_min:.1f} 分钟 ({remaining_sec/3600:.1f} 小时)")
print(f"  🕐 预计完成: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("📌 最新数据样本 (最后5行):")
print(df.tail(5)[['ts', 'ofi', 'z_ofi', 'ema_ofi', 'latency_ms', 'queue_dropped']].to_string())
print("=" * 70)

# 判断速率稳定性
if rate >= 48.0:
    print("✅ 速率稳定，预计顺利完成2小时测试！")
elif rate >= 42.0:
    print("⚠️  速率略低，但仍可达标（300k点）")
else:
    print("❌ 速率过低，可能无法达标，请检查系统负载")
print("=" * 70)

