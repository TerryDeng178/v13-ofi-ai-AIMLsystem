#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_configs.py - CVD参数网格搜索与对比分析

功能：
- 自动生成参数组合的.env文件
- 串行运行CVD测试
- 汇总结果并生成排名表
- 输出对比图表和报告

使用方法：
    python compare_configs.py --symbol ETHUSDT --duration 1200 --grid "MAD_MULTIPLIER=[1.46,1.47,1.48];SCALE_FAST_WEIGHT=[0.30,0.32,0.35]"
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def parse_grid_spec(grid_spec: str) -> Dict[str, List[float]]:
    """解析网格规格字符串"""
    params = {}
    for param_spec in grid_spec.split(';'):
        if '=' in param_spec:
            param_name, values_str = param_spec.split('=', 1)
            # 解析 [1.46,1.47,1.48] 格式
            values_str = values_str.strip('[]')
            values = [float(x.strip()) for x in values_str.split(',')]
            params[param_name] = values
    return params

def generate_config_combinations(params: Dict[str, List[float]]) -> List[Dict[str, float]]:
    """生成所有参数组合"""
    import itertools
    
    param_names = list(params.keys())
    param_values = list(params.values())
    
    combinations = []
    for combo in itertools.product(*param_values):
        combination = dict(zip(param_names, combo))
        combinations.append(combination)
    
    return combinations

def create_env_file(config: Dict[str, float], output_dir: Path, index: int) -> Path:
    """创建.env配置文件"""
    env_file = output_dir / f"config_{index:02d}.env"
    
    # 基线配置
    base_config = {
        "CVD_Z_MODE": "delta",
        "HALF_LIFE_TRADES": 300,
        "WINSOR_LIMIT": 8.0,
        "STALE_THRESHOLD_MS": 5000,
        "FREEZE_MIN": 80,
        "SCALE_MODE": "hybrid",
        "EWMA_FAST_HL": 80,
        "MAD_WINDOW_TRADES": 300,
        "MAD_SCALE_FACTOR": 1.4826,
        "SCALE_SLOW_WEIGHT": 0.65  # 默认值
    }
    
    # 更新配置
    base_config.update(config)
    
    # 计算SCALE_SLOW_WEIGHT
    if "SCALE_FAST_WEIGHT" in config:
        base_config["SCALE_SLOW_WEIGHT"] = 1.0 - config["SCALE_FAST_WEIGHT"]
    
    # 写入.env文件
    with open(env_file, 'w') as f:
        for key, value in base_config.items():
            f.write(f"{key}={value}\n")
    
    return env_file

def run_cvd_test(env_file: Path, symbol: str, duration: int, output_dir: Path) -> Dict[str, Any]:
    """运行CVD测试"""
    test_output_dir = output_dir / f"test_{env_file.stem}"
    test_output_dir.mkdir(exist_ok=True)
    
    # 设置环境变量
    env = os.environ.copy()
    with open(env_file, 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                env[key] = value
    
    # 运行CVD测试
    cmd = [
        sys.executable, "examples/run_realtime_cvd.py",
        "--symbol", symbol,
        "--duration", str(duration),
        "--output-dir", str(test_output_dir)
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=duration+60)
        end_time = time.time()
        
        # 查找生成的报告文件
        report_files = list(test_output_dir.glob("report_*.json"))
        if report_files:
            with open(report_files[0], 'r') as f:
                report = json.load(f)
        else:
            report = {"error": "No report file found"}
        
        return {
            "config_file": str(env_file),
            "test_dir": str(test_output_dir),
            "duration_actual": end_time - start_time,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "report": report
        }
        
    except subprocess.TimeoutExpired:
        return {
            "config_file": str(env_file),
            "test_dir": str(test_output_dir),
            "error": "Test timeout",
            "return_code": -1
        }
    except Exception as e:
        return {
            "config_file": str(env_file),
            "test_dir": str(test_output_dir),
            "error": str(e),
            "return_code": -1
        }

def analyze_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """分析测试结果并生成排名表"""
    data = []
    
    for i, result in enumerate(results):
        if "error" in result:
            continue
            
        report = result.get("report", {})
        data_stats = report.get("data_stats", {})
        validation = report.get("validation", {})
        
        # 提取关键指标
        row = {
            "config_id": i,
            "config_file": Path(result["config_file"]).stem,
            "total_records": data_stats.get("total_records", 0),
            "avg_rate_per_sec": data_stats.get("avg_rate_per_sec", 0),
            "duration_actual": result.get("duration_actual", 0),
            "return_code": result.get("return_code", -1),
            "validation_passed": sum(validation.values()) if validation else 0,
            "validation_total": len(validation) if validation else 0
        }
        
        # 添加Z-score统计（如果有）
        z_stats = data_stats.get("z_cvd_stats", {})
        if z_stats:
            row.update({
                "z_p50": z_stats.get("p50"),
                "z_p95": z_stats.get("p95"),
                "z_p99": z_stats.get("p99")
            })
        
        # 添加延迟统计
        latency_stats = data_stats.get("latency_stats", {})
        if latency_stats:
            row.update({
                "latency_p50": latency_stats.get("p50"),
                "latency_p95": latency_stats.get("p95"),
                "latency_p99": latency_stats.get("p99")
            })
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 计算排名分数
    if not df.empty:
        # 基于验证通过率和数据量计算分数
        df["score"] = (
            df["validation_passed"] / df["validation_total"] * 0.4 +
            (df["total_records"] / df["total_records"].max()) * 0.3 +
            (1 - df["latency_p95"] / df["latency_p95"].max()) * 0.3
        )
        df = df.sort_values("score", ascending=False)
    
    return df

def generate_report(results: List[Dict[str, Any]], df: pd.DataFrame, output_dir: Path, report_file: Path):
    """生成分析报告"""
    report_content = f"""# CVD参数敏感性分析报告

## 测试概览

- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **测试数量**: {len(results)}
- **成功数量**: {len([r for r in results if r.get('return_code') == 0])}
- **失败数量**: {len([r for r in results if r.get('return_code') != 0])}

## 参数组合排名

| 排名 | 配置ID | 验证通过率 | 记录数 | 延迟P95 | 分数 |
|------|--------|------------|--------|---------|------|
"""
    
    for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
        report_content += f"| {i} | {row['config_id']} | {row['validation_passed']}/{row['validation_total']} | {row['total_records']} | {row.get('latency_p95', 'N/A'):.1f}ms | {row['score']:.3f} |\n"
    
    report_content += f"""

## 详细结果

### Top 3 配置详情

"""
    
    for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
        config_id = row['config_id']
        result = results[config_id] if config_id < len(results) else {}
        
        report_content += f"""
#### 配置 {i}: {row['config_file']}

- **验证通过率**: {row['validation_passed']}/{row['validation_total']}
- **总记录数**: {row['total_records']}
- **平均速率**: {row['avg_rate_per_sec']:.2f} 条/秒
- **延迟P95**: {row.get('latency_p95', 'N/A'):.1f}ms
- **分数**: {row['score']:.3f}

"""
    
    # 保存报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"报告已保存到: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="CVD参数网格搜索与对比分析")
    parser.add_argument("--symbol", default="ETHUSDT", help="交易对")
    parser.add_argument("--duration", type=int, default=1200, help="测试时长（秒）")
    parser.add_argument("--warmup-sec", type=int, default=300, help="预热时长（秒）")
    parser.add_argument("--grid", required=True, help="参数网格规格")
    parser.add_argument("--out", default="./data/cvd_grid_test", help="输出目录")
    parser.add_argument("--report", default="./docs/reports/PARAMETER_SENSITIVITY_ANALYSIS.md", help="报告文件")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析参数网格
    params = parse_grid_spec(args.grid)
    print(f"参数网格: {params}")
    
    # 生成参数组合
    combinations = generate_config_combinations(params)
    print(f"生成 {len(combinations)} 个参数组合")
    
    # 创建配置目录
    config_dir = output_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    
    # 生成配置文件
    config_files = []
    for i, combo in enumerate(combinations):
        env_file = create_env_file(combo, config_dir, i)
        config_files.append(env_file)
        print(f"创建配置 {i+1}/{len(combinations)}: {env_file.name}")
    
    # 运行测试
    print(f"\n开始运行 {len(combinations)} 个测试...")
    results = []
    
    for i, env_file in enumerate(config_files):
        print(f"运行测试 {i+1}/{len(config_files)}: {env_file.name}")
        result = run_cvd_test(env_file, args.symbol, args.duration, output_dir)
        results.append(result)
        
        # 保存中间结果
        results_file = output_dir / "grid_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # 分析结果
    print("\n分析结果...")
    df = analyze_results(results)
    
    # 保存排名表
    rank_file = output_dir / "grid_rank_table.csv"
    df.to_csv(rank_file, index=False)
    print(f"排名表已保存到: {rank_file}")
    
    # 生成报告
    report_file = Path(args.report)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    generate_report(results, df, output_dir, report_file)
    
    print(f"\n✅ 网格搜索完成！")
    print(f"📊 结果目录: {output_dir}")
    print(f"📋 报告文件: {report_file}")
    print(f"🏆 Top 3 配置:")
    for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
        print(f"  {i}. 配置{row['config_id']}: 分数={row['score']:.3f}, 验证={row['validation_passed']}/{row['validation_total']}")

if __name__ == "__main__":
    main()
