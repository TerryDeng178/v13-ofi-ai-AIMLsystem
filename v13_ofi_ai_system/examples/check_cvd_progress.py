# -*- coding: utf-8 -*-
"""快速检查CVD测试进度"""
import sys
import io
from datetime import datetime, timedelta

# Windows控制台UTF-8支持
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 基于终端日志推算的启动时间
start_time = datetime(2025, 10, 18, 0, 32, 33)
target_duration = 7205  # 秒（120分5秒）

now = datetime.now()
elapsed = (now - start_time).total_seconds()
progress = min(elapsed / target_duration * 100, 100)
remaining = max(target_duration - elapsed, 0)

# 基于Task 1.2.9的速率（3.9笔/秒）估算
estimated_records = int(elapsed * 3.9)
total_estimated = int(target_duration * 3.9)

print("=" * 60)
print("🔍 CVD Gold级别测试 - 进度报告")
print("=" * 60)
print()
print(f"📅 启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏰ 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
print()
print(f"⏱️  已运行: {int(elapsed//60)}分{int(elapsed%60)}秒 ({elapsed:.0f}秒)")
print(f"⏳ 剩余时间: {int(remaining//60)}分{int(remaining%60)}秒 ({remaining:.0f}秒)")
print(f"📊 完成度: {progress:.1f}%")
print()
print("=" * 60)
print("📈 数据估算（基于3.9笔/秒速率）")
print("=" * 60)
print(f"预计已采集: ~{estimated_records:,}笔")
print(f"目标总量: ~{total_estimated:,}笔")
print()

# 里程碑检查
milestones = [
    (1800, "30分钟", "Bronze级别"),
    (3600, "60分钟", "Silver级别"),
    (7205, "120分钟", "Gold级别（目标）")
]

print("=" * 60)
print("🎯 里程碑进度")
print("=" * 60)
for duration, label, level in milestones:
    if elapsed >= duration:
        status = "✅ 已达成"
    elif elapsed >= duration * 0.9:
        status = f"🔜 即将达成（剩余{int(duration-elapsed)}秒）"
    else:
        status = f"⏳ 未达成（剩余{int(duration-elapsed)}秒）"
    print(f"{label:12} | {level:20} | {status}")

print("=" * 60)
print()

# 预计完成时间
if remaining > 0:
    eta = now + timedelta(seconds=remaining)
    print(f"🎉 预计完成时间: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   （约{int(remaining//60)}分钟后）")
else:
    print("✅ 测试应已完成！")
print()

