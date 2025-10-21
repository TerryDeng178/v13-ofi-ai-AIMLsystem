#!/usr/bin/env python3
"""
分析当前6个关键改进的验证结果
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_results():
    """分析当前结果"""
    print("=== 6个关键改进验证结果分析 ===")
    print()
    
    # 读取结果
    results_file = Path("artifacts/analysis/ofi_cvd/summary/metrics_overview.csv")
    if not results_file.exists():
        print("结果文件不存在")
        return
    
    df = pd.read_csv(results_file)
    
    print("关键指标汇总:")
    print("-" * 50)
    
    # 按信号类型和窗口分组分析
    for signal_type in ['ofi', 'cvd', 'fusion']:
        print(f"\n{signal_type.upper()} 信号分析:")
        signal_data = df[df['signal_type'] == signal_type]
        
        if signal_data.empty:
            continue
            
        for window in ['60s', '180s', '300s']:
            window_data = signal_data[signal_data['window'] == window]
            if window_data.empty:
                continue
                
            print(f"  {window}:")
            for _, row in window_data.iterrows():
                symbol = row['symbol']
                auc = row['AUC']
                pr_auc = row['PR_AUC']
                ic = row['IC']
                direction = row.get('direction_suggestion', 'N/A')
                
                print(f"    {symbol}: AUC={auc:.3f}, PR-AUC={pr_auc:.3f}, IC={ic:.3f}, 方向={direction}")
    
    print("\n" + "="*60)
    print("DoD vNext 门槛检查:")
    print("="*60)
    
    # 检查Fusion AUC
    fusion_data = df[df['signal_type'] == 'fusion']
    max_fusion_auc = fusion_data['AUC'].max()
    print(f"Fusion最大AUC: {max_fusion_auc:.3f} (门槛: ≥0.58)")
    
    if max_fusion_auc >= 0.58:
        print("Fusion AUC达标")
    else:
        print("Fusion AUC未达标")
    
    # 检查方向翻转建议
    flip_suggestions = df[df['direction_suggestion'] == 'flip']
    print(f"\n方向翻转建议: {len(flip_suggestions)}个信号建议翻转")
    
    # 检查校准指标
    if 'Brier' in df.columns and 'ECE' in df.columns:
        min_ece = df['ECE'].min()
        print(f"最小ECE: {min_ece:.3f} (门槛: ≤0.10)")
        if min_ece <= 0.10:
            print("校准指标达标")
        else:
            print("校准指标未达标")
    
    print("\n" + "="*60)
    print("诊断建议:")
    print("="*60)
    
    # 分析问题
    issues = []
    
    if max_fusion_auc < 0.58:
        issues.append("Fusion AUC过低，需要优化信号组合")
    
    if len(flip_suggestions) > 0:
        issues.append(f"{len(flip_suggestions)}个信号建议翻转，说明方向可能有问题")
    
    # 检查OFI和CVD的AUC
    ofi_aucs = df[df['signal_type'] == 'ofi']['AUC'].values
    cvd_aucs = df[df['signal_type'] == 'cvd']['AUC'].values
    
    if np.mean(ofi_aucs) < 0.52:
        issues.append("OFI信号质量不佳，可能需要L1价跃迁优化")
    
    if np.mean(cvd_aucs) < 0.52:
        issues.append("CVD信号质量不佳，可能需要参数调优")
    
    if issues:
        print("发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("未发现明显问题")
    
    print("\n" + "="*60)
    print("下一步建议:")
    print("="*60)
    
    if max_fusion_auc < 0.58:
        print("1. 运行参数调优网格搜索")
        print("2. 检查切片分析，寻找有优势的时间段")
        print("3. 考虑调整Fusion权重或添加门控")
    
    if len(flip_suggestions) > 0:
        print("4. 实施信号翻转，重新评估")
    
    print("5. 检查L1 OFI的价跃迁事件统计")
    print("6. 验证中间价标签的构造质量")

if __name__ == "__main__":
    analyze_results()
