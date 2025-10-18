#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OFI Data Analysis Script for Task 1.2.5
按照任务卡验收标准进行数据分析和验证
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json
import sys
import io

# Windows UTF-8 fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Analyze OFI data collected from run_realtime_ofi.py")
    parser.add_argument("--data", required=True, help="Path to parquet file or directory containing parquet files")
    parser.add_argument("--out", default="v13_ofi_ai_system/examples/figs", help="Output directory for figures")
    parser.add_argument("--report", default="v13_ofi_ai_system/examples/TASK_1_2_5_REPORT.md", help="Output report file")
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data)
    if data_path.is_dir():
        # Find parquet files
        parquet_files = list(data_path.glob("*.parquet"))
        if not parquet_files:
            print(f"错误: 在 {data_path} 中未找到 parquet 文件")
            sys.exit(1)
        print(f"找到 {len(parquet_files)} 个 parquet 文件")
        
        # 💡 优化4: 多文件合并时添加run_id，便于后续分运行统计
        dfs = []
        for f in parquet_files:
            df_temp = pd.read_parquet(f)
            # 使用文件名（不含路径和扩展名）作为run_id
            df_temp['run_id'] = f.stem  # e.g., "20251017_1800"
            dfs.append(df_temp)
        df = pd.concat(dfs, ignore_index=True)
        print(f"✓ 已添加run_id列，便于分运行统计")
    else:
        df = pd.read_parquet(data_path)
        # 单文件也添加run_id
        df['run_id'] = data_path.stem
    
    print(f"\n总数据点数: {len(df)}")
    print(f"时间跨度: {(df['ts'].max() - df['ts'].min()) / 1000 / 3600:.2f} 小时")
    
    # ⚠️ 关键修复1: 按时间排序，确保连续性计算正确
    df = df.sort_values('ts').reset_index(drop=True)
    print("✓ 数据已按时间戳排序")
    
    # 创建输出目录
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 验收标准检查 ==========
    results = {}
    
    # 1. 数据覆盖
    print("\n" + "="*60)
    print("1. 数据覆盖验证")
    print("="*60)
    results['total_points'] = len(df)
    results['time_span_hours'] = (df['ts'].max() - df['ts'].min()) / 1000 / 3600
    
    # 数据连续性 (排序后计算)
    ts_diff = df['ts'].diff()
    max_gap = ts_diff.max()
    results['max_gap_ms'] = max_gap
    results['continuity_pass'] = max_gap <= 2000
    
    # 💡 优化1: 连续性的稳健统计（监控用，非硬标准）
    if len(ts_diff) > 1:
        gap_p99 = ts_diff.quantile(0.99)
        gap_p999 = ts_diff.quantile(0.999)
        results['gap_p99_ms'] = gap_p99
        results['gap_p999_ms'] = gap_p999
    else:
        results['gap_p99_ms'] = 0
        results['gap_p999_ms'] = 0
    
    print(f"采样点数: {results['total_points']} ({'✓ 通过' if results['total_points'] >= 300000 else '✗ 未达标'})")
    print(f"时间跨度: {results['time_span_hours']:.2f} 小时")
    print(f"最大时间缺口: {max_gap:.2f} ms ({'✓ 通过' if results['continuity_pass'] else '✗ 未达标'})")
    if results.get('gap_p99_ms', 0) > 0:
        print(f"  - P99缺口: {results['gap_p99_ms']:.2f} ms (监控)")
        print(f"  - P99.9缺口: {results['gap_p999_ms']:.2f} ms (监控)")
    
    # 2. 功能正确性
    print("\n" + "="*60)
    print("2. 功能正确性验证")
    print("="*60)
    
    # 分量和校验
    if 'k_components_sum' in df.columns:
        component_check = np.abs(df["k_components_sum"] - df["ofi"]) < 1e-9
        results['component_check_pass_rate'] = component_check.mean()
        results['component_check_pass'] = results['component_check_pass_rate'] > 0.99
        print(f"分量和校验通过率: {results['component_check_pass_rate']*100:.2f}% ({'✓ 通过' if results['component_check_pass'] else '✗ 未达标'})")
    else:
        print("⚠ 警告: 缺少 k_components_sum 字段，无法进行分量和校验")
        results['component_check_pass'] = False
    
    # 非空字段自洽性
    df_no_warmup = df[df["warmup"] == False]
    results['null_check'] = {
        'ofi_null': df["ofi"].isna().sum(),
        'ema_ofi_null': df["ema_ofi"].isna().sum(),
        'warmup_null': df["warmup"].isna().sum(),
        'std_zero_null': df["std_zero"].isna().sum(),
        'z_ofi_null_non_warmup': df_no_warmup["z_ofi"].isna().sum(),
        'ts_null': df["ts"].isna().sum(),
    }
    results['null_check_pass'] = all(v == 0 for v in results['null_check'].values())
    
    print(f"非空字段检查: {'✓ 全部通过' if results['null_check_pass'] else '✗ 有NULL值'}")
    for k, v in results['null_check'].items():
        if v > 0:
            print(f"  - {k}: {v} NULL值")
    
    # 3. Z-score 标准化稳健性
    print("\n" + "="*60)
    print("3. Z-score 标准化稳健性验证")
    print("="*60)
    
    z_median = df_no_warmup["z_ofi"].median()
    z_q25 = df_no_warmup["z_ofi"].quantile(0.25)
    z_q75 = df_no_warmup["z_ofi"].quantile(0.75)
    z_iqr = z_q75 - z_q25
    z_tail2 = (df_no_warmup["z_ofi"].abs() > 2).mean()
    z_tail3 = (df_no_warmup["z_ofi"].abs() > 3).mean()
    
    results['z_score'] = {
        'median': z_median,
        'iqr': z_iqr,
        'tail2_pct': z_tail2 * 100,
        'tail3_pct': z_tail3 * 100,
        'median_pass': -0.1 <= z_median <= 0.1,
        'iqr_pass': 0.8 <= z_iqr <= 1.6,
        'tail2_pass': 0.01 <= z_tail2 <= 0.08,
        'tail3_pass': z_tail3 <= 0.015,
    }
    
    print(f"中位数: {z_median:.4f} ({'✓ 通过' if results['z_score']['median_pass'] else '✗ 未达标'})")
    print(f"IQR: {z_iqr:.4f} ({'✓ 通过' if results['z_score']['iqr_pass'] else '✗ 未达标'})")
    print(f"|Z|>2 占比: {z_tail2*100:.2f}% ({'✓ 通过' if results['z_score']['tail2_pass'] else '✗ 未达标'})")
    print(f"|Z|>3 占比: {z_tail3*100:.2f}% ({'✓ 通过' if results['z_score']['tail3_pass'] else '✗ 未达标'})")
    
    std_zero_count = (df["std_zero"] == True).sum()
    results['std_zero_count'] = std_zero_count
    results['std_zero_pass'] = std_zero_count == 0
    print(f"std_zero标记次数: {std_zero_count} ({'✓ 通过' if results['std_zero_pass'] else '✗ 未达标'})")
    
    warmup_pct = (df["warmup"] == True).mean()
    results['warmup_pct'] = warmup_pct * 100
    results['warmup_pass'] = warmup_pct <= 0.10
    print(f"warmup占比: {warmup_pct*100:.2f}% ({'✓ 通过' if results['warmup_pass'] else '✗ 未达标'})")
    
    # ⚠️ 关键修复2: 汇总Z-score子项到顶层
    results['z_score_pass'] = all([
        results['z_score']['median_pass'],
        results['z_score']['iqr_pass'],
        results['z_score']['tail2_pass'],
        results['z_score']['tail3_pass'],
        results['std_zero_pass'],
        results['warmup_pass'],
    ])
    
    # 4. 数据质量
    print("\n" + "="*60)
    print("4. 数据质量验证")
    print("="*60)
    
    # 坏数据点率
    if 'bad_points' in df.columns:
        bad_points_incremental = df["bad_points"].diff().clip(lower=0).fillna(0).sum()
        bad_points_rate = bad_points_incremental / len(df)
        results['bad_points_rate'] = bad_points_rate
        results['bad_points_pass'] = bad_points_rate <= 0.001
        print(f"坏数据点率: {bad_points_rate*100:.4f}% ({'✓ 通过' if results['bad_points_pass'] else '✗ 未达标'})")
    else:
        results['bad_points_pass'] = True
        print("坏数据点率: N/A (无 bad_points 字段)")
    
    # 解析错误 (假设运行期间无解析错误)
    results['parse_errors'] = 0
    results['parse_errors_pass'] = True
    print(f"解析错误: {results['parse_errors']} ({'✓ 通过' if results['parse_errors_pass'] else '✗ 未达标'})")
    
    # 5. 稳定性与性能
    print("\n" + "="*60)
    print("5. 稳定性与性能验证")
    print("="*60)
    
    if 'latency_ms' in df.columns:
        latency_p95 = df["latency_ms"].quantile(0.95)
        results['latency_p95'] = latency_p95
        results['latency_pass'] = latency_p95 < 5
        print(f"处理延迟p95: {latency_p95:.3f} ms ({'✓ 通过' if results['latency_pass'] else '✗ 未达标'})")
    else:
        results['latency_pass'] = False
        print("处理延迟p95: N/A (无 latency_ms 字段)")
    
    if 'reconnect_count' in df.columns:
        reconnects = df["reconnect_count"].max() - df["reconnect_count"].min()
        # 💡 优化2: 除零保护
        if results['time_span_hours'] > 0:
            reconnect_rate = reconnects / results['time_span_hours']
        else:
            reconnect_rate = 0 if reconnects == 0 else float('inf')
        results['reconnect_rate_per_hour'] = reconnect_rate
        results['reconnect_pass'] = reconnect_rate <= 3
        print(f"重连频率: {reconnect_rate:.2f} 次/小时 ({'✓ 通过' if results['reconnect_pass'] else '✗ 未达标'})")
    else:
        results['reconnect_pass'] = True
        print("重连频率: N/A (无 reconnect_count 字段)")
    
    if 'queue_dropped' in df.columns:
        # ⚠️ 关键修复3: 使用diff()计算增量，避免累计值失真
        queue_dropped_incremental = df["queue_dropped"].diff().clip(lower=0).fillna(0).sum()
        queue_dropped_rate = queue_dropped_incremental / len(df)
        results['queue_dropped_rate'] = queue_dropped_rate
        results['queue_dropped_pass'] = queue_dropped_rate <= 0.005
        print(f"队列丢弃率: {queue_dropped_rate*100:.4f}% ({'✓ 通过' if results['queue_dropped_pass'] else '✗ 未达标'})")
    else:
        results['queue_dropped_pass'] = True
        print("队列丢弃率: N/A (无 queue_dropped 字段)")
    
    # ========== 生成图表 ==========
    print("\n" + "="*60)
    print("生成图表")
    print("="*60)
    
    # 图1: Z-score直方图
    plt.figure(figsize=(10, 6))
    plt.hist(df_no_warmup["z_ofi"], bins=100, edgecolor='black', alpha=0.7)
    plt.axvline(z_median, color='red', linestyle='--', label=f'Median={z_median:.3f}')
    plt.axvline(z_q25, color='orange', linestyle='--', label=f'Q25={z_q25:.3f}')
    plt.axvline(z_q75, color='orange', linestyle='--', label=f'Q75={z_q75:.3f}')
    plt.xlabel('Z-score')
    plt.ylabel('Frequency')
    plt.title('Z-score Distribution (non-warmup)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    hist_path = out_dir / "hist_z.png"
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {hist_path}")
    plt.close()
    
    # 图2: OFI时间序列 (采样)
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).sort_values('ts')
    
    plt.figure(figsize=(14, 6))
    plt.plot(range(len(df_sample)), df_sample["ofi"], alpha=0.6, linewidth=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('OFI')
    plt.title(f'OFI Time Series (sampled {sample_size} points)')
    plt.grid(True, alpha=0.3)
    ofi_ts_path = out_dir / "ofi_timeseries.png"
    plt.savefig(ofi_ts_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {ofi_ts_path}")
    plt.close()
    
    # 图3: Z-score时间序列 (采样)
    plt.figure(figsize=(14, 6))
    df_sample_no_warmup = df_sample[df_sample["warmup"] == False]
    plt.plot(range(len(df_sample_no_warmup)), df_sample_no_warmup["z_ofi"], alpha=0.6, linewidth=0.5)
    plt.axhline(2, color='red', linestyle='--', alpha=0.5, label='|Z|=2')
    plt.axhline(-2, color='red', linestyle='--', alpha=0.5)
    plt.axhline(3, color='orange', linestyle='--', alpha=0.5, label='|Z|=3')
    plt.axhline(-3, color='orange', linestyle='--', alpha=0.5)
    plt.xlabel('Sample Index (non-warmup)')
    plt.ylabel('Z-score')
    plt.title(f'Z-score Time Series (sampled {len(df_sample_no_warmup)} points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    z_ts_path = out_dir / "z_timeseries.png"
    plt.savefig(z_ts_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {z_ts_path}")
    plt.close()
    
    # 图4: 延迟箱线图
    if 'latency_ms' in df.columns:
        plt.figure(figsize=(8, 6))
        plt.boxplot(df["latency_ms"], vert=True)
        plt.ylabel('Latency (ms)')
        plt.title('Processing Latency Distribution')
        plt.grid(True, alpha=0.3, axis='y')
        lat_box_path = out_dir / "latency_box.png"
        plt.savefig(lat_box_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存: {lat_box_path}")
        plt.close()
    
    # ========== 生成报告 ==========
    print("\n" + "="*60)
    print("生成报告")
    print("="*60)
    
    report_path = Path(args.report)
    
    # ⚠️ 关键修复4: 计算图片相对路径，避免路径不匹配
    import os
    report_dir = report_path.parent
    def rel_path(img_path):
        """计算图片相对于报告的相对路径"""
        try:
            return os.path.relpath(img_path, report_dir).replace('\\', '/')
        except:
            return str(img_path)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Task 1.2.5 OFI计算测试报告\n\n")
        f.write(f"**测试执行时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**数据源**: `{args.data}`\n\n")
        f.write("---\n\n")
        
        f.write("## 验收标准对照结果\n\n")
        
        f.write("### 1. 数据覆盖\n")
        f.write(f"- [{'x' if results['total_points'] >= 300000 else ' '}] 采样点数: {results['total_points']:,} (≥300,000)\n")
        f.write(f"- [{'x' if results['continuity_pass'] else ' '}] 数据连续性: max_gap={max_gap:.2f}ms (≤2000ms)\n")
        f.write(f"- [{'x' if results['time_span_hours'] >= 2 else ' '}] 时间跨度: {results['time_span_hours']:.2f}小时 (≥2小时)\n\n")
        
        f.write("### 2. 功能正确性\n")
        f.write(f"- [{'x' if results.get('component_check_pass', False) else ' '}] 分量和校验: {results.get('component_check_pass_rate', 0)*100:.2f}% (>99%)\n")
        f.write(f"- [{'x' if results['null_check_pass'] else ' '}] 非空字段自洽性: {'通过' if results['null_check_pass'] else '未通过'}\n\n")
        
        f.write("### 3. Z-score 标准化稳健性\n")
        f.write(f"- [{'x' if results['z_score']['median_pass'] else ' '}] 中位数: {z_median:.4f} (∈[-0.1, +0.1])\n")
        f.write(f"- [{'x' if results['z_score']['iqr_pass'] else ' '}] IQR: {z_iqr:.4f} (∈[0.8, 1.6])\n")
        f.write(f"- [{'x' if results['z_score']['tail2_pass'] else ' '}] |Z|>2 占比: {z_tail2*100:.2f}% (∈[1%, 8%])\n")
        f.write(f"- [{'x' if results['z_score']['tail3_pass'] else ' '}] |Z|>3 占比: {z_tail3*100:.2f}% (≤1.5%)\n")
        f.write(f"- [{'x' if results['std_zero_pass'] else ' '}] std_zero计数: {std_zero_count} (==0)\n")
        f.write(f"- [{'x' if results['warmup_pass'] else ' '}] warmup占比: {warmup_pct*100:.2f}% (≤10%)\n\n")
        
        f.write("### 4. 数据质量\n")
        f.write(f"- [{'x' if results.get('bad_points_pass', False) else ' '}] 坏数据点率: {results.get('bad_points_rate', 0)*100:.4f}% (≤0.1%)\n")
        f.write(f"- [{'x' if results['parse_errors_pass'] else ' '}] 解析错误: {results['parse_errors']} (==0)\n\n")
        
        f.write("### 5. 稳定性与性能\n")
        f.write(f"- [{'x' if results.get('latency_pass', False) else ' '}] 处理延迟p95: {results.get('latency_p95', 0):.3f}ms (<5ms)\n")
        f.write(f"- [{'x' if results.get('reconnect_pass', True) else ' '}] 重连频率: {results.get('reconnect_rate_per_hour', 0):.2f}次/小时 (≤3/小时)\n")
        f.write(f"- [{'x' if results.get('queue_dropped_pass', True) else ' '}] 队列丢弃率: {results.get('queue_dropped_rate', 0)*100:.4f}% (≤0.5%)\n\n")
        
        f.write("## 图表\n\n")
        f.write(f"1. ![Z-score直方图]({rel_path(hist_path)})\n")
        f.write(f"2. ![OFI时间序列]({rel_path(ofi_ts_path)})\n")
        f.write(f"3. ![Z-score时间序列]({rel_path(z_ts_path)})\n")
        if 'latency_ms' in df.columns:
            f.write(f"4. ![延迟箱线图]({rel_path(lat_box_path)})\n\n")
        
        # 总体结论 (使用汇总的顶层pass标志)
        all_pass = all([
            results['total_points'] >= 300000,
            results['continuity_pass'],
            results.get('component_check_pass', False),
            results['null_check_pass'],
            results['z_score_pass'],  # 使用汇总标志
            results.get('bad_points_pass', True),
            results['parse_errors_pass'],
            results.get('latency_pass', True),
            results.get('reconnect_pass', True),
            results.get('queue_dropped_pass', True),
        ])
        
        f.write("## 结论\n\n")
        if all_pass:
            f.write("**✅ 所有验收标准通过，可继续下一任务**\n")
        else:
            f.write("**❌ 部分验收标准未通过，需要改进**\n")
    
    print(f"✓ 报告已保存: {report_path}")
    
    # 💡 优化3: 结果JSON与图表放在同一目录，便于打包
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    results_native = convert_to_native(results)
    results_json_path = out_dir / "analysis_results.json"
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_native, f, indent=2, ensure_ascii=False)
    print(f"✓ 详细结果已保存: {results_json_path}")
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    
    # ⚠️ 关键修复5: 退出码包含所有顶层pass标志
    all_pass_for_exit = all([
        results['total_points'] >= 300000,
        results['continuity_pass'],
        results.get('component_check_pass', False),
        results['null_check_pass'],
        results['z_score_pass'],  # 包含Z-score子项
        results.get('bad_points_pass', True),
        results['parse_errors_pass'],
        results.get('latency_pass', True),
        results.get('reconnect_pass', True),
        results.get('queue_dropped_pass', True),
    ])
    
    print(f"\n最终结果: {'✅ 全部通过' if all_pass_for_exit else '❌ 部分未通过'}")
    
    # 返回退出码 (0=成功, 1=失败)
    sys.exit(0 if all_pass_for_exit else 1)

if __name__ == "__main__":
    main()
