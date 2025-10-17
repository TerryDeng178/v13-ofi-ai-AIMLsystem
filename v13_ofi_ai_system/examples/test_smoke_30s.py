#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30秒快速冒烟测试
验证 run_realtime_ofi.py 是否能正常工作
"""
import sys
import os
import io
import subprocess
import time
from pathlib import Path

# Windows UTF-8 fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    print("=" * 60)
    print("🚀 30秒快速冒烟测试")
    print("=" * 60)
    print()
    
    # 设置环境变量
    env = os.environ.copy()
    env['ENABLE_DATA_COLLECTION'] = '1'
    env['LOG_LEVEL'] = 'INFO'
    
    # 启动进程
    print("▶️  启动 run_realtime_ofi.py --demo ...")
    cmd = [sys.executable, "run_realtime_ofi.py", "--demo"]
    
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )
    
    print(f"✅ 进程已启动 (PID: {proc.pid})")
    print(f"⏰ 运行30秒后自动停止...")
    print()
    
    # 实时输出前10行
    line_count = 0
    start_time = time.time()
    
    try:
        for line in proc.stdout:
            if line_count < 15:
                print(line.rstrip())
                line_count += 1
            elif line_count == 15:
                print("\n... (后续输出省略) ...\n")
                line_count += 1
            
            # 30秒后停止
            if time.time() - start_time > 30:
                print(f"\n⏱️  已运行30秒，停止进程...")
                proc.terminate()
                break
        
        # 等待进程结束
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  进程未响应，强制终止...")
            proc.kill()
            proc.wait()
        
        print(f"✅ 进程已停止 (退出码: {proc.returncode})")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断，停止进程...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    
    print()
    print("=" * 60)
    print("📊 检查生成的文件...")
    print("=" * 60)
    
    # 检查数据文件
    data_dir = Path("../data/DEMO-USD")
    if data_dir.exists():
        parquet_files = list(data_dir.glob("*.parquet"))
        if parquet_files:
            print(f"✅ 找到 {len(parquet_files)} 个数据文件:")
            for f in parquet_files:
                size_kb = f.stat().st_size / 1024
                print(f"   - {f.name} ({size_kb:.1f} KB)")
        else:
            print("❌ 未找到 .parquet 文件")
    else:
        print("❌ 数据目录不存在")
    
    # 检查日志文件
    log_dir = Path("../logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("ws_*.log"))
        if log_files:
            print(f"✅ 找到 {len(log_files)} 个日志文件:")
            for f in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                size_kb = f.stat().st_size / 1024
                print(f"   - {f.name} ({size_kb:.1f} KB)")
        else:
            print("⚠️  未找到日志文件")
    else:
        print("⚠️  日志目录不存在")
    
    print()
    print("=" * 60)
    print("✅ 冒烟测试完成！")
    print("=" * 60)
    print()
    print("📝 下一步:")
    print("   1. 检查上述输出，确认无错误")
    print("   2. 如果有数据文件，可以运行分析:")
    print("      python analysis.py --data ../data/DEMO-USD --out figs --report TASK_1_2_5_REPORT.md")
    print()

if __name__ == "__main__":
    main()

