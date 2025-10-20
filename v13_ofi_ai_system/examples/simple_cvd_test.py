#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_cvd_test.py - 简化的CVD参数测试脚本

功能：
- 测试单个参数组合
- 生成基础统计报告
- 支持快速验证

使用方法：
    python simple_cvd_test.py --config step_1_6_microtune.env --symbol BTCUSDT --duration 300
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def load_env_config(config_file: Path) -> dict:
    """加载.env配置文件"""
    config = {}
    with open(config_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                config[key] = value
    return config

def run_cvd_test(config_file: Path, symbol: str, duration: int, output_dir: Path) -> dict:
    """运行CVD测试"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量
    env = os.environ.copy()
    config = load_env_config(config_file)
    env.update(config)
    
    # 运行CVD测试
    cmd = [
        sys.executable, "examples/run_realtime_cvd.py",
        "--symbol", symbol,
        "--duration", str(duration),
        "--output-dir", str(output_dir)
    ]
    
    print(f"运行CVD测试: {config_file.name}")
    print(f"命令: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=duration+60)
        end_time = time.time()
        
        # 查找生成的报告文件
        report_files = list(output_dir.glob("report_*.json"))
        if report_files:
            with open(report_files[0], 'r') as f:
                report = json.load(f)
        else:
            report = {"error": "No report file found"}
        
        return {
            "config_file": str(config_file),
            "test_dir": str(output_dir),
            "duration_actual": end_time - start_time,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "report": report,
            "success": result.returncode == 0
        }
        
    except subprocess.TimeoutExpired:
        return {
            "config_file": str(config_file),
            "test_dir": str(output_dir),
            "error": "Test timeout",
            "return_code": -1,
            "success": False
        }
    except Exception as e:
        return {
            "config_file": str(config_file),
            "test_dir": str(output_dir),
            "error": str(e),
            "return_code": -1,
            "success": False
        }

def print_test_summary(result: dict):
    """打印测试摘要"""
    print("\n" + "="*60)
    print("CVD Test Results Summary")
    print("="*60)
    
    if result["success"]:
        print("SUCCESS: Test completed successfully")
        report = result.get("report", {})
        data_stats = report.get("data_stats", {})
        validation = report.get("validation", {})
        
        print(f"Data Statistics:")
        print(f"  - Total Records: {data_stats.get('total_records', 'N/A')}")
        print(f"  - Avg Rate: {data_stats.get('avg_rate_per_sec', 'N/A'):.2f} records/sec")
        print(f"  - CVD Range: {data_stats.get('cvd_range', 'N/A')}")
        
        print(f"Latency Statistics:")
        latency_stats = data_stats.get("latency_stats", {})
        if latency_stats:
            print(f"  - P50: {latency_stats.get('p50', 'N/A'):.1f}ms")
            print(f"  - P95: {latency_stats.get('p95', 'N/A'):.1f}ms")
            print(f"  - P99: {latency_stats.get('p99', 'N/A'):.1f}ms")
        
        print(f"Validation Results:")
        if validation:
            for key, value in validation.items():
                status = "PASS" if value else "FAIL"
                print(f"  - {key}: {status}")
        
        print(f"Output Directory: {result['test_dir']}")
        
    else:
        print("FAILED: Test failed")
        if "error" in result:
            print(f"Error: {result['error']}")
        if result.get("stderr"):
            print(f"Error Output: {result['stderr']}")

def main():
    parser = argparse.ArgumentParser(description="简化的CVD参数测试")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--symbol", default="BTCUSDT", help="交易对")
    parser.add_argument("--duration", type=int, default=300, help="测试时长（秒）")
    parser.add_argument("--output-dir", help="输出目录（可选）")
    
    args = parser.parse_args()
    
    # 配置文件路径
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return 1
    
    # 输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"data/cvd_test_{timestamp}")
    
    # 运行测试
    result = run_cvd_test(config_file, args.symbol, args.duration, output_dir)
    
    # 打印结果
    print_test_summary(result)
    
    # 保存结果
    result_file = output_dir / "test_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n📄 详细结果已保存到: {result_file}")
    
    return 0 if result["success"] else 1

if __name__ == "__main__":
    sys.exit(main())
