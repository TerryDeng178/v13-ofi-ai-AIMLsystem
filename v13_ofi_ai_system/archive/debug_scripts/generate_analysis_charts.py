#!/usr/bin/env python3
"""
生成Task_1.2.13分析图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_test_results():
    """加载所有测试结果"""
    results = []
    
    # 加载网格搜索结果
    grid_dir = Path('data/cvd_grid_search')
    for test_folder in grid_dir.glob('test_*'):
        parquet_files = list(test_folder.glob('*.parquet'))
        if parquet_files:
            try:
                df = pd.read_parquet(parquet_files[0])
                z_valid = df[df['z_cvd'].notna()]['z_cvd']
                if len(z_valid) > 0:
                    results.append({
                        'phase': 'Grid Search',
                        'test': test_folder.name,
                        'p_z_gt_2': np.mean(np.abs(z_valid) > 2),
                        'p_z_gt_3': np.mean(np.abs(z_valid) > 3),
                        'median_z': np.median(z_valid),
                        'p95_z': np.percentile(z_valid, 95),
                        'count': len(z_valid)
                    })
            except:
                pass
    
    # 加载优化搜索结果
    opt_dir = Path('data/cvd_optimized_search')
    for test_folder in opt_dir.glob('test_*'):
        parquet_files = list(test_folder.glob('*.parquet'))
        if parquet_files:
            try:
                df = pd.read_parquet(parquet_files[0])
                z_valid = df[df['z_cvd'].notna()]['z_cvd']
                if len(z_valid) > 0:
                    results.append({
                        'phase': 'Optimized Search',
                        'test': test_folder.name,
                        'p_z_gt_2': np.mean(np.abs(z_valid) > 2),
                        'p_z_gt_3': np.mean(np.abs(z_valid) > 3),
                        'median_z': np.median(z_valid),
                        'p95_z': np.percentile(z_valid, 95),
                        'count': len(z_valid)
                    })
            except:
                pass
    
    # 加载超精细搜索结果
    ultra_dir = Path('data/cvd_ultra_fine_search')
    for test_folder in ultra_dir.glob('test_*'):
        parquet_files = list(test_folder.glob('*.parquet'))
        if parquet_files:
            try:
                df = pd.read_parquet(parquet_files[0])
                z_valid = df[df['z_cvd'].notna()]['z_cvd']
                if len(z_valid) > 0:
                    results.append({
                        'phase': 'Ultra Fine Search',
                        'test': test_folder.name,
                        'p_z_gt_2': np.mean(np.abs(z_valid) > 2),
                        'p_z_gt_3': np.mean(np.abs(z_valid) > 3),
                        'median_z': np.median(z_valid),
                        'p95_z': np.percentile(z_valid, 95),
                        'count': len(z_valid)
                    })
            except:
                pass
    
    return pd.DataFrame(results)

def create_analysis_charts():
    """创建分析图表"""
    # 加载数据
    df = load_test_results()
    
    if df.empty:
        print("没有找到测试数据")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Task_1.2.13 CVD Z-score微调优化分析', fontsize=16, fontweight='bold')
    
    # 1. P(|Z|>3) 优化历程
    ax1 = axes[0, 0]
    phase_data = df.groupby('phase')['p_z_gt_3'].agg(['mean', 'min', 'max']).reset_index()
    phases = phase_data['phase']
    means = phase_data['mean']
    mins = phase_data['min']
    maxs = phase_data['max']
    
    x = np.arange(len(phases))
    ax1.bar(x, means, alpha=0.7, label='平均值', color='skyblue')
    ax1.errorbar(x, means, yerr=[means - mins, maxs - means], 
                fmt='o', color='red', capsize=5, label='范围')
    ax1.axhline(y=0.02, color='green', linestyle='--', label='目标线 (2%)')
    ax1.set_xlabel('优化阶段')
    ax1.set_ylabel('P(|Z|>3)')
    ax1.set_title('P(|Z|>3) 优化历程')
    ax1.set_xticks(x)
    ax1.set_xticklabels(phases, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. P(|Z|>2) vs P(|Z|>3) 散点图
    ax2 = axes[0, 1]
    colors = {'Grid Search': 'red', 'Optimized Search': 'orange', 'Ultra Fine Search': 'green'}
    for phase in df['phase'].unique():
        phase_df = df[df['phase'] == phase]
        ax2.scatter(phase_df['p_z_gt_2'], phase_df['p_z_gt_3'], 
                   c=colors[phase], label=phase, alpha=0.7, s=60)
    
    ax2.axhline(y=0.02, color='green', linestyle='--', alpha=0.7, label='P(|Z|>3) 目标')
    ax2.axvline(x=0.08, color='blue', linestyle='--', alpha=0.7, label='P(|Z|>2) 目标')
    ax2.set_xlabel('P(|Z|>2)')
    ax2.set_ylabel('P(|Z|>3)')
    ax2.set_title('P(|Z|>2) vs P(|Z|>3) 分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 数据量变化
    ax3 = axes[1, 0]
    phase_counts = df.groupby('phase')['count'].agg(['mean', 'std']).reset_index()
    x = np.arange(len(phase_counts))
    ax3.bar(x, phase_counts['mean'], yerr=phase_counts['std'], 
           alpha=0.7, color='lightgreen', capsize=5)
    ax3.set_xlabel('优化阶段')
    ax3.set_ylabel('平均数据量')
    ax3.set_title('测试数据量变化')
    ax3.set_xticks(x)
    ax3.set_xticklabels(phase_counts['phase'], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. 最佳结果对比
    ax4 = axes[1, 1]
    best_results = df.groupby('phase').apply(lambda x: x.loc[x['p_z_gt_3'].idxmin()]).reset_index(drop=True)
    
    metrics = ['p_z_gt_2', 'p_z_gt_3', 'median_z']
    metric_labels = ['P(|Z|>2)', 'P(|Z|>3)', 'Median(|Z|)']
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, phase in enumerate(best_results['phase']):
        values = [best_results.iloc[i][metric] for metric in metrics]
        ax4.bar(x + i*width, values, width, label=phase, alpha=0.7)
    
    ax4.set_xlabel('指标')
    ax4.set_ylabel('值')
    ax4.set_title('各阶段最佳结果对比')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(metric_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('docs/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'task_1_2_13_analysis_charts.png', dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_dir / 'task_1_2_13_analysis_charts.png'}")
    
    # 显示图表
    plt.show()

def create_parameter_analysis():
    """创建参数分析图表"""
    # 这里可以添加更详细的参数分析
    pass

if __name__ == "__main__":
    create_analysis_charts()
