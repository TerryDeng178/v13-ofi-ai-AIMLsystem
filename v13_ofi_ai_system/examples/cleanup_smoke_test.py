#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理冒烟测试数据"""
import os
import sys
import io
from pathlib import Path

# Windows UTF-8 fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    # 冒烟测试数据路径
    smoke_data_dir = Path("v13_ofi_ai_system/examples/v13_ofi_ai_system/data/DEMO-USD")
    figs_dir = Path("v13_ofi_ai_system/examples/figs")
    report_file = Path("v13_ofi_ai_system/examples/TASK_1_2_5_REPORT.md")
    json_file = Path("v13_ofi_ai_system/examples/figs/analysis_results.json")
    
    deleted_count = 0
    
    # 删除冒烟测试数据文件
    if smoke_data_dir.exists():
        for f in smoke_data_dir.glob("*.parquet"):
            try:
                f.unlink()
                print(f"✓ 已删除: {f.name}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ 无法删除 {f.name}: {e}")
    
    # 删除图表
    if figs_dir.exists():
        for f in figs_dir.glob("*.png"):
            try:
                f.unlink()
                print(f"✓ 已删除: {f.name}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ 无法删除 {f.name}: {e}")
    
    # 删除报告
    if report_file.exists():
        try:
            report_file.unlink()
            print(f"✓ 已删除: {report_file.name}")
            deleted_count += 1
        except Exception as e:
            print(f"✗ 无法删除 {report_file.name}: {e}")
    
    # 删除JSON
    if json_file.exists():
        try:
            json_file.unlink()
            print(f"✓ 已删除: {json_file.name}")
            deleted_count += 1
        except Exception as e:
            print(f"✗ 无法删除 {json_file.name}: {e}")
    
    print()
    print(f"🧹 清理完成：共删除 {deleted_count} 个文件")

if __name__ == "__main__":
    main()

