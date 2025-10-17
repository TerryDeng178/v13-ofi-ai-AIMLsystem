#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

df = pd.read_parquet('v13_ofi_ai_system/data/DEMO-USD/20251017_1756.parquet')
time_span_s = (df['ts'].max() - df['ts'].min()) / 1000
rate = len(df) / time_span_s
time_span_h = time_span_s / 3600

print(f'数据点数: {len(df):,}')
print(f'时间跨度: {time_span_h:.3f} 小时 ({time_span_s/60:.1f} 分钟)')
print(f'采集速率: {rate:.2f} 点/秒')
print(f'进度: {(time_span_h / 2) * 100:.1f}%')
print(f'预估最终: {int(rate * 7200):,} 点')
print()

if rate >= 45:
    print('✅ 速率已恢复正常 (≥45点/秒)')
    print('   预估可满足300k点验收标准')
elif rate >= 40:
    print('🟡 速率接近正常 (40-45点/秒)')
    print('   预估可达到288k-324k点')
elif rate >= 35:
    print('⚠️  速率略低 (35-40点/秒)')
    print('   预估252k-288k点，可能无法满足300k标准')
else:
    print('❌ 速率偏低 (<35点/秒)')
    print('   预估<252k点，无法满足300k标准')
    print('   建议延长测试时间或重启测试')

