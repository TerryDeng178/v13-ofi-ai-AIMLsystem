#!/usr/bin/env python3
"""
CVD Z-score微调优化 - 扩展网格搜索
基于初步结果，扩大参数搜索空间
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from itertools import product

# 确保项目根目录在sys.path中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def create_test_config(mad_multiplier, scale_fast_weight, half_life_trades, winsor_limit, output_dir):
    """创建测试配置文件"""
    config_content = f"""# CVD Z-score微调测试配置 - 扩展搜索
# 基于初步结果优化参数范围

# CVD计算模式
CVD_Z_MODE=delta
HALF_LIFE_TRADES={half_life_trades}
WINSOR_LIMIT={winsor_limit}
STALE_THRESHOLD_MS=5000
FREEZE_MIN=80

# 软冻结配置（默认关闭）
SOFT_FREEZE_V2=0
SOFT_FREEZE_GAP_MS_MIN=4000
SOFT_FREEZE_GAP_MS_MAX=5000
SOFT_FREEZE_MIN_TRADES=1

# 缩放配置
SCALE_MODE=hybrid
EWMA_FAST_HL=80
SCALE_FAST_WEIGHT={scale_fast_weight}
SCALE_SLOW_WEIGHT={1-scale_fast_weight}

# MAD配置
MAD_WINDOW_TRADES=300
MAD_SCALE_FACTOR=1.4826
MAD_MULTIPLIER={mad_multiplier}

# 测试参数
SYMBOL=BTCUSDT
DURATION=600
WATERMARK_MS=500
PRINT_EVERY=1000
"""
    
    config_file = output_dir / f"ext_grid_{mad_multiplier}_{scale_fast_weight}_{half_life_trades}_{winsor_limit}.env"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    return config_file

def run_cvd_test(config_file, symbol, duration, output_dir):
    """运行CVD测试"""
    test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = output_dir / f"test_{test_id}"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        str(project_root / "examples" / "run_realtime_cvd.py"),
        "--symbol", symbol,
        "--duration", str(duration),
        "--output-dir", str(test_output_dir)
    ]
    
    # 准备环境变量
    env = os.environ.copy()
    
    # 从配置文件加载参数
    with open(config_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                env[key] = value
    
    print(f"运行测试: {config_file.name}")
    
    process = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    # 读取测试结果
    report_file = test_output_dir / f"report_{symbol.lower()}_{test_id}.json"
    report_data = {}
    
    if report_file.exists():
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
        except json.JSONDecodeError as e:
            report_data["error"] = f"JSON解析错误: {e}"
    else:
        report_data["error"] = "报告文件未找到"
    
    return {
        "config_file": str(config_file),
        "test_output_dir": str(test_output_dir),
        "stdout": process.stdout,
        "stderr": process.stderr,
        "return_code": process.returncode,
        "report": report_data,
        "success": process.returncode == 0 and "error" not in report_data
    }

def analyze_z_score_distribution(parquet_file):
    """分析Z-score分布"""
    try:
        import pandas as pd
        import numpy as np
        
        df = pd.read_parquet(parquet_file)
        z_valid = df[df['z_cvd'].notna()]['z_cvd']
        
        if len(z_valid) == 0:
            return {
                "p_z_gt_2": 0.0,
                "p_z_gt_3": 0.0,
                "median_z": 0.0,
                "p95_z": 0.0,
                "p99_z": 0.0,
                "valid_count": 0
            }
        
        return {
            "p_z_gt_2": float(np.mean(np.abs(z_valid) > 2)),
            "p_z_gt_3": float(np.mean(np.abs(z_valid) > 3)),
            "median_z": float(np.median(z_valid)),
            "p95_z": float(np.percentile(z_valid, 95)),
            "p99_z": float(np.percentile(z_valid, 99)),
            "valid_count": len(z_valid)
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="CVD Z-score微调优化 - 扩展网格搜索")
    parser.add_argument("--symbol", default="BTCUSDT", help="交易对")
    parser.add_argument("--duration", type=int, default=600, help="每个测试的时长（秒）")
    parser.add_argument("--output-dir", type=Path, default=Path("data/cvd_extended_grid_search"), help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = args.output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 扩展参数搜索空间
    mad_multipliers = [1.40, 1.42, 1.44, 1.46, 1.48, 1.50]  # 扩大MAD范围
    scale_fast_weights = [0.25, 0.30, 0.35, 0.40, 0.45]     # 扩大权重范围
    half_life_trades = [200, 250, 300, 350, 400]            # 扩大半衰期范围
    winsor_limits = [6.0, 8.0, 10.0]                        # 测试不同截断值
    
    print("="*80)
    print("CVD Z-score微调优化 - 扩展网格搜索")
    print("="*80)
    print(f"交易对: {args.symbol}")
    print(f"测试时长: {args.duration}秒")
    print(f"输出目录: {args.output_dir}")
    print()
    
    # 生成所有参数组合（限制数量避免过多测试）
    combinations = []
    for mad in mad_multipliers:
        for scale in scale_fast_weights:
            for half_life in half_life_trades:
                for winsor in winsor_limits:
                    combinations.append((mad, scale, half_life, winsor))
    
    # 只测试前20个组合（避免过多测试）
    combinations = combinations[:20]
    
    print(f"测试参数组合数: {len(combinations)}")
    print("参数范围:")
    print(f"  MAD_MULTIPLIER: {mad_multipliers}")
    print(f"  SCALE_FAST_WEIGHT: {scale_fast_weights}")
    print(f"  HALF_LIFE_TRADES: {half_life_trades}")
    print(f"  WINSOR_LIMIT: {winsor_limits}")
    print()
    
    # 运行测试
    results = []
    start_time = time.time()
    
    for i, (mad_mult, scale_fast, half_life, winsor) in enumerate(combinations):
        print(f"进度: {i+1}/{len(combinations)} - MAD={mad_mult}, SCALE_FAST={scale_fast}, HALF_LIFE={half_life}, WINSOR={winsor}")
        
        # 创建配置文件
        config_file = create_test_config(mad_mult, scale_fast, half_life, winsor, config_dir)
        
        # 运行测试
        result = run_cvd_test(config_file, args.symbol, args.duration, args.output_dir)
        
        # 分析Z-score分布
        if result["success"]:
            test_output_dir = Path(result["test_output_dir"])
            parquet_files = list(test_output_dir.glob("*.parquet"))
            if parquet_files:
                z_analysis = analyze_z_score_distribution(parquet_files[0])
                result["z_analysis"] = z_analysis
        
        results.append(result)
        
        # 小延迟
        time.sleep(3)
    
    total_time = time.time() - start_time
    print(f"\n所有测试完成，总耗时: {total_time/60:.1f}分钟")
    
    # 分析结果
    print("\n" + "="*80)
    print("扩展网格搜索结果分析")
    print("="*80)
    
    valid_results = [r for r in results if r["success"] and "z_analysis" in r and "error" not in r["z_analysis"]]
    
    if not valid_results:
        print("❌ 没有成功的测试结果")
        return
    
    # 按P(|Z|>3)排序
    valid_results.sort(key=lambda x: x["z_analysis"]["p_z_gt_3"])
    
    print(f"成功测试数量: {len(valid_results)}/{len(results)}")
    print("\nTop 5 结果:")
    print("-" * 80)
    
    for i, result in enumerate(valid_results[:5]):
        config_name = Path(result["config_file"]).name
        z_analysis = result["z_analysis"]
        report = result["report"]
        data_stats = report.get("data_stats", {})
        
        print(f"第{i+1}名: {config_name}")
        print(f"  总记录数: {data_stats.get('total_records', 0)}")
        print(f"  接收速率: {data_stats.get('avg_rate_per_sec', 0):.2f} 笔/秒")
        print(f"  P(|Z|>2): {z_analysis['p_z_gt_2']:.1%}")
        print(f"  P(|Z|>3): {z_analysis['p_z_gt_3']:.1%}")
        print(f"  Median(|Z|): {z_analysis['median_z']:.3f}")
        print(f"  P95(|Z|): {z_analysis['p95_z']:.3f}")
        print(f"  有效Z-score数: {z_analysis['valid_count']}")
        print()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = args.output_dir / f"extended_grid_search_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_info": {
                "symbol": args.symbol,
                "duration": args.duration,
                "total_combinations": len(combinations),
                "timestamp": timestamp
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {results_file}")
    
    # 检查是否有达标的配置
    best_result = valid_results[0] if valid_results else None
    if best_result and best_result["z_analysis"]["p_z_gt_3"] <= 0.02:  # 2%
        print(f"\n🎉 找到达标配置: {Path(best_result['config_file']).name}")
        print("P(|Z|>3) ≤ 2.0% 目标达成！")
    else:
        print(f"\n⚠️ 最佳配置P(|Z|>3) = {best_result['z_analysis']['p_z_gt_3']:.1%}，仍需优化")

if __name__ == "__main__":
    main()
