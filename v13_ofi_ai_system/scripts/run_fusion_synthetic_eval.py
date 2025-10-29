"""
OFI_CVD_Fusion 合成数据评测脚本

生成合成数据并评估融合组件的性能和信号质量

Author: V13 QA Team
Date: 2025-10-28
"""

import sys
import os
import importlib.util
from pathlib import Path
import time
import csv

# 动态导入依赖
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[ERROR] numpy 未安装，请安装: pip install numpy")
    sys.exit(1)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[WARNING] pandas 未安装，将跳过部分功能")


def load_fusion_module():
    """动态加载 ofi_cvd_fusion 模块"""
    project_root = Path(__file__).parent.parent
    fusion_path = project_root / "src" / "ofi_cvd_fusion.py"
    
    if not fusion_path.exists():
        raise FileNotFoundError(f"找不到 ofi_cvd_fusion.py: {fusion_path}")
    
    spec = importlib.util.spec_from_file_location("ofi_cvd_fusion", str(fusion_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module


def generate_scenario_s1():
    """场景 S1: 同向强信号 + 小抖动"""
    rng = np.random.default_rng(42)
    n_samples = 18000  # 3分钟 * 100Hz = 18000
    dt = 0.01  # 100Hz
    
    base_ofi = 2.8
    base_cvd = 3.2
    noise_std = 0.2
    
    ts_list = []
    z_ofi_list = []
    z_cvd_list = []
    lag_list = []
    
    for i in range(n_samples):
        ts_list.append(1000.0 + i * dt)
        z_ofi_list.append(base_ofi + rng.normal(0, noise_std))
        z_cvd_list.append(base_cvd + rng.normal(0, noise_std))
        lag_list.append(0.0)  # 无滞后
    
    return ts_list, z_ofi_list, z_cvd_list, lag_list


def generate_scenario_s2():
    """场景 S2: 交替滞后 + 超时降级"""
    rng = np.random.default_rng(42)
    n_samples = 18000
    dt = 0.01
    
    ts_list = []
    z_ofi_list = []
    z_cvd_list = []
    lag_list = []
    
    for i in range(n_samples):
        ts = 1000.0 + i * dt
        
        # 每隔2s注入滞后
        if (i % 200) == 0:  # 每200个样本（2s）
            lag = 0.5  # max_lag * 1.5 (假设 max_lag=0.25)
        else:
            lag = 0.0
        
        ts_list.append(ts)
        z_ofi_list.append(rng.normal(3.0, 0.5))
        z_cvd_list.append(rng.normal(2.0, 0.5))
        lag_list.append(lag)
    
    return ts_list, z_ofi_list, z_cvd_list, lag_list


def generate_scenario_s3():
    """场景 S3: 对冲/反向"""
    rng = np.random.default_rng(42)
    n_samples = 18000
    dt = 0.01
    
    ts_list = []
    z_ofi_list = []
    z_cvd_list = []
    lag_list = []
    
    for i in range(n_samples):
        ts = 1000.0 + i * dt
        
        ts_list.append(ts)
        z_ofi_list.append(rng.normal(2.0, 0.5))  # 正
        z_cvd_list.append(-rng.normal(2.0, 0.5))  # 负（反向）
        lag_list.append(0.0)
    
    return ts_list, z_ofi_list, z_cvd_list, lag_list


def run_evaluation(fusion, ts_list, z_ofi_list, z_cvd_list, lag_list, scenario_name):
    """运行评估并收集指标"""
    results = []
    update_costs = []
    
    for i, (ts, z_ofi, z_cvd, lag_sec) in enumerate(zip(ts_list, z_ofi_list, z_cvd_list, lag_list)):
        # 性能测试
        t_start = time.perf_counter()
        result = fusion.update(z_ofi=z_ofi, z_cvd=z_cvd, ts=ts, lag_sec=lag_sec)
        t_end = time.perf_counter()
        
        update_cost_ms = (t_end - t_start) * 1000
        update_costs.append(update_cost_ms)
        
        results.append({
            'index': i,
            'ts': ts,
            'signal': result.get('signal', 'neutral'),
            'fusion_score': result.get('fusion_score', 0.0),
            'consistency': result.get('consistency', 0.0),
            'reason_codes': ','.join(result.get('reason_codes', [])),
            'update_cost_ms': update_cost_ms
        })
        
        if (i + 1) % 1000 == 0:
            print(f"  [进度] {i+1}/{len(ts_list)} ({100*(i+1)/len(ts_list):.1f}%)")
    
    # 计算统计指标
    stats = fusion.get_stats()
    non_neutral_count = sum(1 for r in results if r['signal'] != 'neutral')
    downgrade_count = stats.get('downgrades', 0)
    cooldown_count = sum(1 for r in results if 'cooldown' in r['reason_codes'])
    min_duration_count = sum(1 for r in results if 'min_duration' in r['reason_codes'])
    consistency_boost_count = sum(1 for r in results if 'consistency_boost' in r['reason_codes'])
    
    # 计算分位数
    update_costs_sorted = sorted(update_costs)
    n = len(update_costs)
    p50 = update_costs_sorted[n//2] if n > 0 else 0.0
    p95 = update_costs_sorted[int(n*0.95)] if n > 0 else 0.0
    p99 = update_costs_sorted[int(n*0.99)] if n > 0 else 0.0
    
    metrics = {
        'scenario': scenario_name,
        'updates': len(results),
        'non_neutral_count': non_neutral_count,
        'non_neutral_rate': non_neutral_count / len(results) if len(results) > 0 else 0.0,
        'downgrades': downgrade_count,
        'downgrade_rate': downgrade_count / len(results) if len(results) > 0 else 0.0,
        'cooldown_blocks': cooldown_count,
        'cooldown_rate': cooldown_count / len(results) if len(results) > 0 else 0.0,
        'min_duration_blocks': min_duration_count,
        'min_duration_rate': min_duration_count / len(results) if len(results) > 0 else 0.0,
        'consistency_boost': consistency_boost_count,
        'consistency_boost_rate': consistency_boost_count / len(results) if len(results) > 0 else 0.0,
        'update_cost_p50_ms': p50,
        'update_cost_p95_ms': p95,
        'update_cost_p99_ms': p99
    }
    
    return metrics


def save_summary_csv(metrics_list, output_path):
    """保存指标汇总到 CSV"""
    if not metrics_list:
        return
    
    fieldnames = list(metrics_list[0].keys())
    
    # 检查文件是否存在（追加模式）
    file_exists = output_path.exists()
    
    with open(output_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for metrics in metrics_list:
            writer.writerow(metrics)
    
    print(f"[OK] 指标已保存到: {output_path}")


def generate_report_section(metrics_list):
    """生成报告章节"""
    section = "\n## 合成数据评测\n\n"
    section += "| 场景 | 更新次数 | 非中性率 | 降级率 | 冷却率 | 最小持续率 | 一致性提升率 | P50(ms) | P95(ms) | P99(ms) |\n"
    section += "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    
    for m in metrics_list:
        section += f"{m['scenario']} | "
        section += f"{m['updates']} | "
        section += f"{m['non_neutral_rate']:.2%} | "
        section += f"{m['downgrade_rate']:.2%} | "
        section += f"{m['cooldown_rate']:.2%} | "
        section += f"{m['min_duration_rate']:.2%} | "
        section += f"{m['consistency_boost_rate']:.2%} | "
        section += f"{m['update_cost_p50_ms']:.3f} | "
        section += f"{m['update_cost_p95_ms']:.3f} | "
        section += f"{m['update_cost_p99_ms']:.3f} |\n"
    
    section += "\n### 结论\n\n"
    
    # 检查验收标准
    for m in metrics_list:
        section += f"#### {m['scenario']}\n\n"
        
        checks = []
        if m['update_cost_p99_ms'] < 3.0:
            checks.append("[OK] P99(update_cost) < 3ms")
        else:
            checks.append(f"[WARN] P99(update_cost) = {m['update_cost_p99_ms']:.3f}ms (>= 3ms)")
        
        if m['scenario'] == 'S1':
            if 0.05 <= m['non_neutral_rate'] <= 0.20:
                checks.append(f"[OK] non_neutral_rate = {m['non_neutral_rate']:.2%} (在 5%-20% 范围内)")
            else:
                checks.append(f"[WARN] non_neutral_rate = {m['non_neutral_rate']:.2%} (不在 5%-20% 范围内)")
            
            if m['consistency_boost_rate'] > 0:
                checks.append(f"[OK] consistency_boost_rate > 0")
            else:
                checks.append(f"[WARN] consistency_boost_rate = 0")
        
        elif m['scenario'] == 'S2':
            if m['downgrade_rate'] > 0:
                checks.append(f"[OK] downgrade_rate > 0")
            else:
                checks.append(f"[WARN] downgrade_rate = 0")
        
        for check in checks:
            section += f"- {check}\n"
        section += "\n"
    
    return section


def main():
    """主函数"""
    print("=" * 60)
    print("OFI_CVD_Fusion 合成数据评测")
    print("=" * 60)
    
    # 加载融合模块
    try:
        fusion_mod = load_fusion_module()
        OFI_CVD_Fusion = fusion_mod.OFI_CVD_Fusion
        OFICVDFusionConfig = fusion_mod.OFICVDFusionConfig
    except Exception as e:
        print(f"[ERROR] 无法加载融合模块: {e}")
        sys.exit(1)
    
    # 创建输出目录
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    summary_csv = results_dir / "fusion_metrics_summary.csv"
    
    # 如果文件存在，先删除（确保每次运行是新文件）
    if summary_csv.exists():
        summary_csv.unlink()
    
    all_metrics = []
    
    # 场景列表
    scenarios = [
        ('S1', generate_scenario_s1, '同向强信号 + 小抖动'),
        ('S2', generate_scenario_s2, '交替滞后 + 超时降级'),
        ('S3', generate_scenario_s3, '对冲/反向')
    ]
    
    # 运行每个场景
    for scenario_key, scenario_func, scenario_desc in scenarios:
        print(f"\n[场景 {scenario_key}] {scenario_desc}")
        print("-" * 60)
        
        # 重置融合器
        fusion = OFI_CVD_Fusion(cfg=OFICVDFusionConfig())
        
        # 生成数据
        print("  [1/3] 生成合成数据...")
        ts_list, z_ofi_list, z_cvd_list, lag_list = scenario_func()
        print(f"  [OK] 已生成 {len(ts_list)} 个样本")
        
        # 运行评估
        print("  [2/3] 运行评估...")
        metrics = run_evaluation(fusion, ts_list, z_ofi_list, z_cvd_list, lag_list, scenario_key)
        print(f"  [OK] 评估完成")
        
        all_metrics.append(metrics)
        
        # 打印关键指标
        print(f"\n  关键指标:")
        print(f"    - 非中性率: {metrics['non_neutral_rate']:.2%}")
        print(f"    - P99(update_cost): {metrics['update_cost_p99_ms']:.3f}ms")
    
    # 保存汇总
    print(f"\n[保存] 保存指标汇总...")
    save_summary_csv(all_metrics, summary_csv)
    
    # 生成报告章节
    report_section = generate_report_section(all_metrics)
    
    # 保存到报告文件（如果是首次运行，创建文件）
    report_file = results_dir / "fusion_test_report.md"
    if not report_file.exists():
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# OFI_CVD_Fusion 测试报告\n\n")
            f.write("本报告包含融合组件的测试结果与指标分析。\n\n")
    
    # 追加合成数据章节
    with open(report_file, 'a', encoding='utf-8') as f:
        f.write(report_section)
    
    print(f"\n[SUCCESS] 所有场景评测完成！")
    print(f"  - 指标汇总: {summary_csv}")
    print(f"  - 测试报告: {report_file}")


if __name__ == "__main__":
    main()

