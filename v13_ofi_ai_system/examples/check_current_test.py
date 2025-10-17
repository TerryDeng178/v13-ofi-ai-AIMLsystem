# -*- coding: utf-8 -*-
import sys
import io
import pandas as pd
from pathlib import Path
from datetime import datetime

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

elapsed = (df['ts'].iloc[-1] - df['ts'].iloc[0]) / 1000
rate = len(df) / elapsed

print(f"📊 最新数据文件: {latest_file.name}")
print(f"📈 数据点数: {len(df):,}")
print(f"⏱️  运行时间: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
print(f"🚀 采集速率: {rate:.1f} 点/秒")
print(f"")
print(f"前3行:")
print(df.head(3)[['ts', 'ofi', 'z_ofi', 'ema_ofi', 'latency_ms']].to_string())
print(f"")
print(f"最后3行:")
print(df.tail(3)[['ts', 'ofi', 'z_ofi', 'ema_ofi', 'latency_ms']].to_string())

