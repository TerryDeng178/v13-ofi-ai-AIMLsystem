#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1.2.5 正式2小时测试
从项目根目录运行，确保数据路径正确
"""
import sys
import os
import io
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta

# Windows UTF-8 fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    print("=" * 80)
    print("🎯 Task 1.2.5 - 正式2小时测试")
    print("=" * 80)
    print()
    print("📊 测试参数:")
    print("   - 运行时长: 2小时 (7200秒)")
    print("   - 数据频率: 50 Hz (50 msgs/s)")
    print("   - 预期数据点: ≈360,000点")
    print("   - 采集模式: DEMO模式 (合成订单簿数据)")
    print()
    print("📁 数据目录: v13_ofi_ai_system/data/DEMO-USD/")
    print("   (从项目根目录运行，路径正确)")
    print()
    
    # 检查当前目录
    cwd = Path.cwd()
    if not (cwd / "v13_ofi_ai_system").exists():
        print("❌ 错误: 请从项目根目录运行此脚本")
        print(f"   当前目录: {cwd}")
        print(f"   应在目录: ofi_cvd_framework/ofi_cvd_framework/")
        return 1
    
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
    
    # 计算预期完成时间
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=2)
    print(f"⏰ 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏰ 预计完成: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 启动进程
    print("▶️  启动 run_realtime_ofi.py --demo ...")
    cmd = [
        sys.executable,
        "v13_ofi_ai_system/examples/run_realtime_ofi.py",
        "--demo"
    ]
    
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
        cwd=cwd
    )
    
    print(f"✅ 进程已启动 (PID: {proc.pid})")
    print()
    print("📝 实时输出（前30行 + 每10分钟打印一次进度）:")
    print("-" * 80)
    
    # 实时输出
    line_count = 0
    last_progress_time = 0
    progress_interval = 600  # 10分钟
    
    try:
        for line in proc.stdout:
            # 显示前30行
            if line_count < 30:
                print(line.rstrip())
                line_count += 1
            elif line_count == 30:
                print()
                print("... (输出继续，每10分钟打印进度) ...")
                print()
                line_count += 1
            
            # 每10分钟打印一次进度
            elapsed = time.time() - start_time.timestamp()
            if elapsed - last_progress_time >= progress_interval:
                elapsed_min = int(elapsed / 60)
                remaining_min = 120 - elapsed_min
                progress_pct = (elapsed / 7200) * 100
                print(f"⏱️  进度: {elapsed_min}/120分钟 ({progress_pct:.1f}%) | 剩余: {remaining_min}分钟")
                last_progress_time = elapsed
            
            # 2小时后停止
            if elapsed > 7200:
                print()
                print("=" * 80)
                print(f"⏱️  已运行2小时 ({int(elapsed)}秒)，停止进程...")
                print("=" * 80)
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
        elapsed = time.time() - start_time.timestamp()
        elapsed_min = int(elapsed / 60)
        print()
        print("=" * 80)
        print(f"⚠️  用户中断 (已运行 {elapsed_min}分钟)，停止进程...")
        print("=" * 80)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        
        if elapsed_min < 120:
            print()
            print("⚠️  测试未满2小时，可能无法满足所有验收标准")
            print(f"   实际运行: {elapsed_min}分钟 / 120分钟")
    
    print()
    print("=" * 80)
    print("📊 检查生成的文件...")
    print("=" * 80)
    print()
    
    # 检查数据文件（正确路径）
    data_dir = cwd / "v13_ofi_ai_system" / "data" / "DEMO-USD"
    
    print(f"📁 数据目录: {data_dir}")
    
    if data_dir.exists():
        parquet_files = list(data_dir.glob("*.parquet"))
        if parquet_files:
            print(f"✅ 找到 {len(parquet_files)} 个数据文件:")
            total_size = 0
            for f in sorted(parquet_files, key=lambda x: x.stat().st_mtime, reverse=True):
                size_mb = f.stat().st_size / 1024 / 1024
                total_size += size_mb
                print(f"   - {f.name} ({size_mb:.2f} MB)")
            print(f"   总大小: {total_size:.2f} MB")
            
            # 尝试读取数据点数
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_files[0])
                print(f"   数据点数: {len(df):,}")
                time_span_hours = (df['ts'].max() - df['ts'].min()) / 1000 / 3600
                print(f"   时间跨度: {time_span_hours:.2f} 小时")
                
                # 验收标准检查
                print()
                print("📋 快速验收检查:")
                if len(df) >= 300000:
                    print(f"   ✅ 采样点数: {len(df):,} ≥ 300,000")
                else:
                    print(f"   ❌ 采样点数: {len(df):,} < 300,000")
                
                if time_span_hours >= 2.0:
                    print(f"   ✅ 时间跨度: {time_span_hours:.2f}小时 ≥ 2小时")
                else:
                    print(f"   ❌ 时间跨度: {time_span_hours:.2f}小时 < 2小时")
                
            except Exception as e:
                print(f"   ⚠️  无法读取数据: {e}")
        else:
            print("❌ 未找到 .parquet 文件")
    else:
        print("❌ 数据目录不存在")
    
    print()
    print("=" * 80)
    print("✅ 数据采集完成！")
    print("=" * 80)
    print()
    print("📝 下一步：运行分析脚本")
    print()
    print("   cd v13_ofi_ai_system\\examples")
    print("   python analysis.py ^")
    print("       --data ..\\data\\DEMO-USD ^")
    print("       --out figs ^")
    print("       --report TASK_1_2_5_REPORT.md")
    print()
    print("或直接运行:")
    print("   python v13_ofi_ai_system\\examples\\analysis.py ^")
    print("       --data v13_ofi_ai_system\\data\\DEMO-USD ^")
    print("       --out v13_ofi_ai_system\\examples\\figs ^")
    print("       --report v13_ofi_ai_system\\examples\\TASK_1_2_5_REPORT.md")
    print()

if __name__ == "__main__":
    sys.exit(main() or 0)

