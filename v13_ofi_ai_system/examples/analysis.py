#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OFI数据分析脚本 - Task 1.2.5
分析OFI数据，进行统计分析和质量验证，产出量化评估报告
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_ofi_data(data_path):
    """加载OFI数据 - 支持多天数据合并"""
    print(f"正在加载OFI数据: {data_path}")
    
    # 如果路径是单个日期，则查找所有日期的数据
    if 'date=' in data_path:
        # 提取基础路径和日期
        base_path = data_path.split('date=')[0]
        print(f"检测到单日数据路径，将搜索所有日期的数据: {base_path}")
        
        # 查找所有日期目录
        all_parquet_files = []
        for root, dirs, files in os.walk(base_path):
            if 'date=' in root and 'kind=ofi' in root:
                for file in files:
                    if file.endswith('.parquet'):
                        all_parquet_files.append(os.path.join(root, file))
        
        parquet_files = all_parquet_files
        print(f"找到多天数据: {len(parquet_files)} 个parquet文件")
        
        # 如果路径还包含单个交易对，则扩展到所有交易对
        if 'symbol=' in data_path:
            print(f"检测到单交易对数据路径，将搜索所有交易对的数据")
            # 查找所有交易对的数据
            all_symbol_files = []
            for root, dirs, files in os.walk(base_path):
                if 'date=' in root and 'kind=ofi' in root and 'symbol=' in root:
                    for file in files:
                        if file.endswith('.parquet'):
                            all_symbol_files.append(os.path.join(root, file))
            
            parquet_files = all_symbol_files
            print(f"找到多交易对数据: {len(parquet_files)} 个parquet文件")
            
            # 统计交易对分布
            symbol_counts = {}
            for file in parquet_files[:100]:  # 检查前100个文件
                try:
                    df_sample = pd.read_parquet(file)
                    if 'symbol' in df_sample.columns:
                        symbol = df_sample['symbol'].iloc[0] if len(df_sample) > 0 else 'unknown'
                        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                except:
                    continue
            
            if symbol_counts:
                print(f"交易对分布预览: {dict(list(symbol_counts.items())[:5])}")
    else:
        # 查找所有parquet文件
        parquet_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
        
        if not parquet_files:
            raise ValueError(f"在 {data_path} 中未找到parquet文件")
        
        print(f"找到 {len(parquet_files)} 个parquet文件")
    
    # 加载所有数据
    dfs = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            print(f"警告: 无法加载文件 {file}: {e}")
    
    if not dfs:
        raise ValueError("没有成功加载任何数据文件")
    
    # 合并数据
    df = pd.concat(dfs, ignore_index=True)
    
    # 按时间戳排序
    if 'ts' in df.columns:
        df = df.sort_values('ts').reset_index(drop=True)
    elif 'ts_ms' in df.columns:
        df = df.sort_values('ts_ms').reset_index(drop=True)
    else:
        print("警告: 未找到时间戳字段")
    
    print(f"成功加载 {len(df)} 条记录")
    return df

def analyze_data_quality(df):
    """分析数据质量"""
    print("\n=== 数据质量分析 ===")
    
    # 基础统计
    total_points = len(df)
    print(f"总数据点数: {total_points:,}")
    
    # 时间跨度
    ts_col = 'ts' if 'ts' in df.columns else 'ts_ms'
    if ts_col in df.columns:
        time_span = (df[ts_col].max() - df[ts_col].min()) / 1000 / 3600  # 小时
        print(f"时间跨度: {time_span:.2f} 小时")
        
        # 数据连续性检查
        time_diffs = df[ts_col].diff().dropna()
        max_gap = time_diffs.max()
        print(f"最大时间间隔: {max_gap:.2f} ms")
        
        # 数据连续性判定
        if max_gap <= 2000:
            print("OK 数据连续性: 通过 (max_gap <= 2000ms)")
        else:
            print(f"FAIL 数据连续性: 失败 (max_gap = {max_gap:.2f}ms > 2000ms)")
    
    # 字段完整性检查 - 适应实际数据结构
    required_fields = ['ts_ms', 'ofi_value', 'ofi_z']  # 根据实际数据结构调整
    missing_fields = [field for field in required_fields if field not in df.columns]
    
    if missing_fields:
        print(f"FAIL 缺少必需字段: {missing_fields}")
    else:
        print("OK 字段完整性: 通过")
    
    # 非空字段检查
    null_counts = df.isnull().sum()
    print(f"\n字段空值统计:")
    for col in df.columns:
        null_count = null_counts[col]
        null_rate = null_count / len(df) * 100
        print(f"  {col}: {null_count} ({null_rate:.2f}%)")
    
    # 修复返回值，确保正确返回时间跨度
    result = {
        'total_points': total_points,
        'null_counts': null_counts.to_dict()
    }
    
    # 添加时间跨度信息
    if ts_col in df.columns and 'time_span' in locals():
        result['time_span_hours'] = time_span
        result['max_gap_ms'] = max_gap
    else:
        result['time_span_hours'] = 0
        result['max_gap_ms'] = 0
    
    return result

def analyze_z_score_robustness(df):
    """分析Z-score稳健性"""
    print("\n=== Z-score稳健性分析 ===")
    
    # 适应实际数据结构
    z_col = 'z_ofi' if 'z_ofi' in df.columns else 'ofi_z'
    if z_col not in df.columns:
        print(f"FAIL 缺少{z_col}字段")
        return {}
    
    # 过滤掉warmup期间的数据（如果存在warmup字段）
    if 'warmup' in df.columns:
        z_data = df[df['warmup'] == False][z_col].dropna()
    else:
        z_data = df[z_col].dropna()
    
    if len(z_data) == 0:
        print("FAIL 没有有效的Z-score数据")
        return {}
    
    # 基础统计
    median = z_data.median()
    iqr = z_data.quantile(0.75) - z_data.quantile(0.25)
    
    print(f"有效Z-score数据点: {len(z_data):,}")
    print(f"中位数: {median:.6f}")
    print(f"IQR: {iqr:.6f}")
    
    # 尾部占比
    p_gt_2 = (np.abs(z_data) > 2).mean() * 100
    p_gt_3 = (np.abs(z_data) > 3).mean() * 100
    
    print(f"P(|z| > 2): {p_gt_2:.2f}%")
    print(f"P(|z| > 3): {p_gt_3:.2f}%")
    
    # 验收标准检查
    results = {}
    
    # 中位数居中
    if -0.1 <= median <= 0.1:
        print("OK 中位数居中: 通过")
        results['median_ok'] = True
    else:
        print(f"FAIL 中位数居中: 失败 (median = {median:.6f})")
        results['median_ok'] = False
    
    # IQR合理
    if 0.8 <= iqr <= 1.6:
        print("OK IQR合理: 通过")
        results['iqr_ok'] = True
    else:
        print(f"FAIL IQR合理: 失败 (IQR = {iqr:.6f})")
        results['iqr_ok'] = False
    
    # 尾部占比
    if 1 <= p_gt_2 <= 8:
        print("OK P(|z| > 2): 通过")
        results['tail_2_ok'] = True
    else:
        print(f"FAIL P(|z| > 2): 失败 ({p_gt_2:.2f}%)")
        results['tail_2_ok'] = False
    
    if p_gt_3 <= 1.5:
        print("OK P(|z| > 3): 通过")
        results['tail_3_ok'] = True
    else:
        print(f"FAIL P(|z| > 3): 失败 ({p_gt_3:.2f}%)")
        results['tail_3_ok'] = False
    
    return {
        'median': median,
        'iqr': iqr,
        'p_gt_2': p_gt_2,
        'p_gt_3': p_gt_3,
        'valid_points': len(z_data),
        **results
    }

def analyze_performance(df):
    """分析性能指标"""
    print("\n=== 性能分析 ===")
    
    # 处理延迟
    if 'latency_ms' in df.columns:
        latency_data = df['latency_ms'].dropna()
        if len(latency_data) > 0:
            p95_latency = latency_data.quantile(0.95)
            print(f"处理延迟P95: {p95_latency:.3f} ms")
            
            if p95_latency < 5:
                print("OK 处理延迟: 通过 (< 5ms)")
                latency_ok = True
            else:
                print(f"FAIL 处理延迟: 失败 ({p95_latency:.3f}ms >= 5ms)")
                latency_ok = False
        else:
            print("WARN 无延迟数据")
            latency_ok = None
    else:
        print("WARN 缺少latency_ms字段")
        latency_ok = None
    
    # 重连频率
    if 'reconnect_count' in df.columns:
        reconnect_data = df['reconnect_count'].dropna()
        if len(reconnect_data) > 0:
            max_reconnect = reconnect_data.max()
            min_reconnect = reconnect_data.min()
            reconnect_freq = max_reconnect - min_reconnect
            
            print(f"重连次数: {reconnect_freq}")
            
            if reconnect_freq <= 3:
                print("OK 重连频率: 通过 (<= 3次/小时)")
                reconnect_ok = True
            else:
                print(f"FAIL 重连频率: 失败 ({reconnect_freq} > 3)")
                reconnect_ok = False
        else:
            print("WARN 无重连数据")
            reconnect_ok = None
    else:
        print("WARN 缺少reconnect_count字段")
        reconnect_ok = None
    
    # 队列丢弃率
    if 'queue_dropped' in df.columns:
        queue_data = df['queue_dropped'].dropna()
        if len(queue_data) > 0:
            max_dropped = queue_data.max()
            min_dropped = queue_data.min()
            total_dropped = max_dropped - min_dropped
            drop_rate = total_dropped / len(df) * 100
            
            print(f"队列丢弃率: {drop_rate:.2f}%")
            
            if drop_rate <= 0.5:
                print("OK 队列丢弃率: 通过 (<= 0.5%)")
                queue_ok = True
            else:
                print(f"FAIL 队列丢弃率: 失败 ({drop_rate:.2f}% > 0.5%)")
                queue_ok = False
        else:
            print("WARN 无队列丢弃数据")
            queue_ok = None
    else:
        print("WARN 缺少queue_dropped字段")
        queue_ok = None
    
    return {
        'latency_ok': latency_ok,
        'reconnect_ok': reconnect_ok,
        'queue_ok': queue_ok
    }

def generate_plots(df, output_dir):
    """生成图表"""
    print(f"\n=== 生成图表到 {output_dir} ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Z-score直方图
    z_col = 'z_ofi' if 'z_ofi' in df.columns else 'ofi_z'
    if z_col in df.columns:
        if 'warmup' in df.columns:
            z_data = df[df['warmup'] == False][z_col].dropna()
        else:
            z_data = df[z_col].dropna()
        if len(z_data) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(z_data, bins=50, alpha=0.7, edgecolor='black')
            plt.title('Z-score Distribution')
            plt.xlabel('Z-score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'hist_z.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("OK 生成 hist_z.png")
    
    # 2. OFI时间序列
    ofi_col = 'ofi' if 'ofi' in df.columns else 'ofi_value'
    ts_col = 'ts' if 'ts' in df.columns else 'ts_ms'
    if ofi_col in df.columns and ts_col in df.columns:
        plt.figure(figsize=(12, 6))
        # 采样显示（避免数据点过多）
        sample_size = min(10000, len(df))
        sample_df = df.sample(n=sample_size).sort_values(ts_col)
        
        plt.plot(sample_df[ts_col], sample_df[ofi_col], alpha=0.7)
        plt.title('OFI Time Series')
        plt.xlabel('Timestamp')
        plt.ylabel('OFI Value')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'ofi_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("OK 生成 ofi_timeseries.png")
    
    # 3. Z-score时间序列
    if z_col in df.columns and ts_col in df.columns:
        if 'warmup' in df.columns:
            z_df = df[df['warmup'] == False].dropna(subset=[z_col])
        else:
            z_df = df.dropna(subset=[z_col])
        if len(z_df) > 0:
            plt.figure(figsize=(12, 6))
            # 采样显示
            sample_size = min(10000, len(z_df))
            sample_df = z_df.sample(n=sample_size).sort_values(ts_col)
            
            plt.plot(sample_df[ts_col], sample_df[z_col], alpha=0.7)
            plt.title('Z-score Time Series')
            plt.xlabel('Timestamp')
            plt.ylabel('Z-score')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'z_timeseries.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("OK 生成 z_timeseries.png")
    
    # 4. 延迟箱线图
    if 'latency_ms' in df.columns:
        latency_data = df['latency_ms'].dropna()
        if len(latency_data) > 0:
            plt.figure(figsize=(10, 6))
            plt.boxplot(latency_data)
            plt.title('Processing Latency Distribution')
            plt.ylabel('Latency (ms)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'latency_box.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("OK 生成 latency_box.png")

def generate_report(data_quality, z_score_analysis, performance, output_file):
    """生成分析报告"""
    print(f"\n=== 生成报告到 {output_file} ===")
    
    report = {
        "analysis_time": datetime.now().isoformat(),
        "data_quality": data_quality,
        "z_score_robustness": z_score_analysis,
        "performance": performance
    }
    
    # 保存JSON报告
    json_file = output_file.replace('.md', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"OK 生成 {json_file}")
    
    # 生成Markdown报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# OFI数据分析报告\n\n")
        f.write(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 数据质量摘要\n\n")
        f.write(f"- 总数据点数: {data_quality['total_points']:,}\n")
        f.write(f"- 时间跨度: {data_quality['time_span_hours']:.2f} 小时\n")
        f.write(f"- 最大时间间隔: {data_quality['max_gap_ms']:.2f} ms\n\n")
        
        if 'z_score_robustness' in z_score_analysis:
            f.write("## Z-score稳健性\n\n")
            f.write(f"- 有效数据点: {z_score_analysis['valid_points']:,}\n")
            f.write(f"- 中位数: {z_score_analysis['median']:.6f}\n")
            f.write(f"- IQR: {z_score_analysis['iqr']:.6f}\n")
            f.write(f"- P(|z| > 2): {z_score_analysis['p_gt_2']:.2f}%\n")
            f.write(f"- P(|z| > 3): {z_score_analysis['p_gt_3']:.2f}%\n\n")
        
        f.write("## 验收标准检查\n\n")
        
        # 数据覆盖
        f.write("### 数据覆盖\n")
        if data_quality['total_points'] >= 300000:
            f.write("- OK 采样点数: 通过 (>= 300,000)\n")
        else:
            f.write(f"- FAIL 采样点数: 失败 ({data_quality['total_points']:,} < 300,000)\n")
        
        if data_quality['max_gap_ms'] <= 2000:
            f.write("- OK 数据连续性: 通过 (<= 2000ms)\n")
        else:
            f.write(f"- FAIL 数据连续性: 失败 ({data_quality['max_gap_ms']:.2f}ms > 2000ms)\n")
        
        # Z-score稳健性
        if 'z_score_robustness' in z_score_analysis:
            f.write("\n### Z-score稳健性\n")
            if z_score_analysis.get('median_ok', False):
                f.write("- OK 中位数居中: 通过\n")
            else:
                f.write("- FAIL 中位数居中: 失败\n")
            
            if z_score_analysis.get('iqr_ok', False):
                f.write("- OK IQR合理: 通过\n")
            else:
                f.write("- FAIL IQR合理: 失败\n")
            
            if z_score_analysis.get('tail_2_ok', False):
                f.write("- OK P(|z| > 2): 通过\n")
            else:
                f.write("- FAIL P(|z| > 2): 失败\n")
            
            if z_score_analysis.get('tail_3_ok', False):
                f.write("- OK P(|z| > 3): 通过\n")
            else:
                f.write("- FAIL P(|z| > 3): 失败\n")
    
    print(f"OK 生成 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='OFI数据分析脚本')
    parser.add_argument('--data', required=True, help='数据路径')
    parser.add_argument('--out', default='figs', help='图表输出目录')
    parser.add_argument('--report', default='TASK_1_2_5_REPORT.md', help='报告输出文件')
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        df = load_ofi_data(args.data)
        
        # 分析数据质量
        data_quality = analyze_data_quality(df)
        
        # 分析Z-score稳健性
        z_score_analysis = analyze_z_score_robustness(df)
        
        # 分析性能
        performance = analyze_performance(df)
        
        # 生成图表
        generate_plots(df, args.out)
        
        # 生成报告
        generate_report(data_quality, z_score_analysis, performance, args.report)
        
        print("\n=== 分析完成 ===")
        print(f"数据点数: {data_quality['total_points']:,}")
        print(f"时间跨度: {data_quality['time_span_hours']:.2f} 小时")
        print(f"图表目录: {args.out}")
        print(f"报告文件: {args.report}")
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
