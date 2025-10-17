#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5分钟DEMO冒烟测试
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
    print("=" * 70)
    print("🚀 Task 1.2.5 - 5分钟DEMO冒烟测试")
    print("=" * 70)
    print()
    print("📊 预期采集: 约15,000点 (50 Hz × 300秒)")
    print("⏰ 运行时间: 5分钟")
    print("📁 数据目录: v13_ofi_ai_system/data/DEMO-USD/")
    print()
    
    # 设置环境变量
    env = os.environ.copy()
    env['ENABLE_DATA_COLLECTION'] = '1'
    env['LOG_LEVEL'] = 'INFO'
    env['DATA_OUTPUT_DIR'] = 'v13_ofi_ai_system/data'
    
    print("🔧 环境变量:")
    print(f"   ENABLE_DATA_COLLECTION = {env['ENABLE_DATA_COLLECTION']}")
    print(f"   LOG_LEVEL = {env['LOG_LEVEL']}")
    print(f"   DATA_OUTPUT_DIR = {env['DATA_OUTPUT_DIR']}")
    print()
    
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
        bufsize=1,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    print(f"✅ 进程已启动 (PID: {proc.pid})")
    print()
    
    # 实时输出前20行
    line_count = 0
    start_time = time.time()
    last_summary_time = 0
    
    print("📝 实时输出（前20行）:")
    print("-" * 70)
    
    try:
        for line in proc.stdout:
            if line_count < 20:
                print(line.rstrip())
                line_count += 1
            elif line_count == 20:
                print("... (后续输出省略) ...\n")
                line_count += 1
            
            # 每30秒打印一个进度
            elapsed = time.time() - start_time
            if elapsed - last_summary_time >= 30:
                print(f"⏱️  已运行 {int(elapsed)}秒 / 300秒...")
                last_summary_time = elapsed
            
            # 5分钟后停止
            if elapsed > 300:
                print()
                print("=" * 70)
                print(f"⏱️  已运行5分钟 ({int(elapsed)}秒)，停止进程...")
                print("=" * 70)
                proc.terminate()
                break
        
        # 等待进程结束
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("⚠️  进程未响应，强制终止...")
            proc.kill()
            proc.wait()
        
        print(f"✅ 进程已停止 (退出码: {proc.returncode})")
        
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print(f"⚠️  用户中断 (已运行 {int(time.time() - start_time)}秒)，停止进程...")
        print("=" * 70)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    
    print()
    print("=" * 70)
    print("📊 检查生成的文件...")
    print("=" * 70)
    print()
    
    # 检查数据文件（相对于脚本所在目录）
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data" / "DEMO-USD"
    
    print(f"📁 数据目录: {data_dir}")
    
    if data_dir.exists():
        parquet_files = list(data_dir.glob("*.parquet"))
        if parquet_files:
            print(f"✅ 找到 {len(parquet_files)} 个数据文件:")
            total_size = 0
            for f in sorted(parquet_files, key=lambda x: x.stat().st_mtime, reverse=True):
                size_kb = f.stat().st_size / 1024
                total_size += size_kb
                print(f"   - {f.name} ({size_kb:.1f} KB)")
            print(f"   总大小: {total_size:.1f} KB")
            
            # 尝试读取数据点数
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_files[0])
                print(f"   数据点数: {len(df)}")
                print(f"   时间跨度: {(df['ts'].max() - df['ts'].min()) / 1000 / 60:.1f} 分钟")
            except Exception as e:
                print(f"   ⚠️  无法读取数据: {e}")
        else:
            print("❌ 未找到 .parquet 文件")
    else:
        print("❌ 数据目录不存在")
    
    # 检查日志文件
    log_dir = script_dir.parent / "logs"
    print()
    print(f"📁 日志目录: {log_dir}")
    
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
    print("=" * 70)
    print("✅ 冒烟测试完成！")
    print("=" * 70)
    print()
    
    if data_dir.exists() and list(data_dir.glob("*.parquet")):
        print("📝 下一步：运行分析脚本")
        print()
        print("   cd v13_ofi_ai_system\\examples")
        print("   python analysis.py ^")
        print("       --data ..\\data\\DEMO-USD ^")
        print("       --out figs ^")
        print("       --report TASK_1_2_5_REPORT.md")
        print()
    else:
        print("⚠️  未生成数据文件，请检查:")
        print("   1. ENABLE_DATA_COLLECTION 环境变量是否设置")
        print("   2. 进程是否正常运行")
        print("   3. 是否有错误日志")
        print()

if __name__ == "__main__":
    main()

