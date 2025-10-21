#!/usr/bin/env python3
"""
监控BTC金测进度
"""

import os
import time
import sys
import io
from datetime import datetime, timedelta

# 修复Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_test_progress():
    """检查测试进度"""
    print("🔍 BTC金测进度监控")
    print("=" * 50)
    
    # 查找最新的测试目录
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("❌ data目录不存在")
        return
    
    # 查找BTC金测目录
    btc_dirs = [d for d in os.listdir(data_dir) if d.startswith("cvd_btc_gold_")]
    if not btc_dirs:
        print("❌ 未找到BTC金测目录")
        return
    
    latest_dir = max(btc_dirs)
    test_path = os.path.join(data_dir, latest_dir)
    
    print(f"📁 测试目录: {test_path}")
    
    # 检查文件数量
    if os.path.exists(test_path):
        files = os.listdir(test_path)
        parquet_files = [f for f in files if f.endswith('.parquet')]
        json_files = [f for f in files if f.endswith('.json')]
        
        print(f"📊 数据文件: {len(parquet_files)} 个Parquet文件")
        print(f"📊 报告文件: {len(json_files)} 个JSON文件")
        
        if parquet_files:
            # 检查最新文件的时间
            latest_file = max(parquet_files)
            file_path = os.path.join(test_path, latest_file)
            file_time = os.path.getmtime(file_path)
            file_datetime = datetime.fromtimestamp(file_time)
            
            print(f"⏰ 最新文件: {latest_file}")
            print(f"⏰ 文件时间: {file_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 计算运行时间
            now = datetime.now()
            runtime = now - file_datetime
            print(f"⏱️ 运行时间: {runtime}")
            
            # 估算剩余时间
            target_duration = timedelta(hours=2)  # 120分钟
            remaining = target_duration - runtime
            if remaining.total_seconds() > 0:
                print(f"⏳ 预计剩余: {remaining}")
            else:
                print("✅ 测试应该已完成")
    
    # 检查进程状态
    print(f"\n🔍 进程状态:")
    try:
        import subprocess
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        if 'python.exe' in result.stdout:
            print("✅ Python进程正在运行")
        else:
            print("❌ 未找到Python进程")
    except:
        print("⚠️ 无法检查进程状态")

def main():
    print("🚀 BTC金测监控工具")
    print("=" * 60)
    
    check_test_progress()
    
    print("\n" + "=" * 60)
    print("📋 监控说明:")
    print("1. 测试目标: 120分钟 (7200秒)")
    print("2. 测试符号: BTCUSDT")
    print("3. 预期数据量: 约10,000-15,000笔交易")
    print("4. 验收标准: 8/8指标全部通过")
    
    print("\n🎯 下一步:")
    print("1. 等待测试完成 (约2小时)")
    print("2. 运行分析脚本")
    print("3. 生成金测报告")

if __name__ == '__main__':
    main()
