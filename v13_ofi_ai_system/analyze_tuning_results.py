#!/usr/bin/env python3
"""
分析参数调优结果
"""

import pandas as pd
import numpy as np

def analyze_tuning_results():
    """分析参数调优结果"""
    print("=" * 80)
    print("参数调优结果分析")
    print("=" * 80)
    
    # 读取参数调优结果
    df = pd.read_csv('artifacts/analysis/param_tuning_results.csv')
    
    # 分析Fusion信号的最佳参数
    fusion_df = df[df['signal_type'] == 'fusion']
    if not fusion_df.empty:
        best_fusion = fusion_df.loc[fusion_df['AUC'].idxmax()]
        print(f"最佳Fusion AUC: {best_fusion['AUC']:.4f}")
        print("最佳参数组合:")
        print(f"  half_life_sec: {best_fusion['half_life_sec']}")
        print(f"  mad_multiplier: {best_fusion['mad_multiplier']}")
        print(f"  winsor_limit: {best_fusion['winsor_limit']}")
        print(f"  fast_weight: {best_fusion['fast_weight']}")
        print(f"  方向建议: {best_fusion['direction_suggestion']}")
        print(f"  winsor命中率: {best_fusion['winsor_hit_rate']:.2%}")
        print(f"  sigma_floor命中率: {best_fusion['sigma_floor_hit_rate']:.2%}")
    
    print()
    print("方向问题分析:")
    direction_issues = df[df['direction_suggestion'] == 'flip']
    if not direction_issues.empty:
        print(f"发现 {len(direction_issues)} 个信号需要翻转")
        for signal_type in direction_issues['signal_type'].unique():
            signal_direction = direction_issues[direction_issues['signal_type'] == signal_type]
            print(f"  {signal_type}: {len(signal_direction)} 个窗口建议翻转")
    
    print()
    print("尺度钳制分析:")
    high_winsor = df[df['winsor_hit_rate'] > 0.5]
    high_floor = df[df['sigma_floor_hit_rate'] > 0.4]
    print(f"winsor命中率>50%的组合: {len(high_winsor)}")
    print(f"sigma_floor命中率>40%的组合: {len(high_floor)}")
    
    # 分析AUC提升情况
    print()
    print("AUC提升分析:")
    for signal_type in ['ofi', 'cvd', 'fusion']:
        signal_data = df[df['signal_type'] == signal_type]
        if not signal_data.empty:
            max_auc = signal_data['AUC'].max()
            min_auc = signal_data['AUC'].min()
            avg_auc = signal_data['AUC'].mean()
            print(f"  {signal_type}: 最大AUC={max_auc:.4f}, 最小AUC={min_auc:.4f}, 平均AUC={avg_auc:.4f}")
    
    # 分析最佳参数组合
    print()
    print("最佳参数组合分析:")
    best_combinations = []
    for signal_type in ['ofi', 'cvd', 'fusion']:
        signal_data = df[df['signal_type'] == signal_type]
        if not signal_data.empty:
            best_idx = signal_data['AUC'].idxmax()
            best_combo = signal_data.loc[best_idx]
            best_combinations.append({
                'signal_type': signal_type,
                'AUC': best_combo['AUC'],
                'half_life_sec': best_combo['half_life_sec'],
                'mad_multiplier': best_combo['mad_multiplier'],
                'winsor_limit': best_combo['winsor_limit'],
                'fast_weight': best_combo['fast_weight']
            })
    
    for combo in best_combinations:
        print(f"  {combo['signal_type']}: AUC={combo['AUC']:.4f}, "
              f"hl={combo['half_life_sec']}, mad={combo['mad_multiplier']}, "
              f"w={combo['winsor_limit']}, fw={combo['fast_weight']}")

if __name__ == "__main__":
    analyze_tuning_results()
