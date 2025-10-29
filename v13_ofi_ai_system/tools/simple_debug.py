#!/usr/bin/env python3
"""
简单调试脚本，分析为什么只有TL场景被优化
"""
import pandas as pd
import numpy as np

def simple_debug():
    """简单调试分析"""
    print("=== 简单调试分析 ===")
    
    # 读取最新的结果文件
    results_file = "outputs/grid_search/20251022_195228/results_btcusdt.csv"
    
    if not os.path.exists(results_file):
        print(f"结果文件不存在: {results_file}")
        return
    
    # 读取结果
    df = pd.read_csv(results_file)
    print(f"总结果数: {len(df)}")
    print(f"场景分布: {df['regime'].value_counts().to_dict()}")
    
    # 检查每个场景的样本数
    print("\n=== 场景样本数检查 ===")
    for regime in ['TL', 'TH', 'WL', 'WH']:
        regime_df = df[df['regime'] == regime]
        count = len(regime_df)
        print(f"{regime}: {count} 个参数组合")
        
        if count > 0:
            # 显示该场景的最佳参数
            # 过滤掉NaN值
            valid_df = regime_df.dropna(subset=['score_reg'])
            if len(valid_df) > 0:
                best_idx = valid_df['score_reg'].idxmax()
                best_row = valid_df.loc[best_idx]
                print(f"  -> 最佳score: {best_row['score_reg']:.4f}")
                print(f"  -> 最佳IR: {best_row['IR_after_cost']:.4f}")
                print(f"  -> 最佳参数: ewm_span={best_row['ewm_span']}, z_window={best_row['z_window']}, w_cvd={best_row['w_cvd']}")
            else:
                print(f"  -> 所有score_reg都是NaN")
    
    # 检查原始数据中的场景分布
    print("\n=== 检查原始数据场景分布 ===")
    # 从之前的输出中可以看到场景分布
    print("从运行输出中看到的场景分布:")
    print("BTCUSDT: {'TL': 32132, 'TH': 11552}")
    print("ETHUSDT: {'TL': 4346, 'TH': 1623}")
    print("ADAUSDT: {'TL': 45005, 'TH': 16445}")
    print("BNBUSDT: {'TL': 44621, 'TH': 16832}")
    print("SOLUSDT: {'TL': 1827, 'TH': 592}")
    print("XRPUSDT: {'TL': 32235, 'TH': 11476}")
    
    print("\n=== 分析问题 ===")
    print("问题分析:")
    print("1. 原始数据中有TL和TH两个场景")
    print("2. 但网格搜索结果中只有TL场景")
    print("3. 可能的原因:")
    print("   - TH场景的样本数虽然>10，但可能<500")
    print("   - 或者在数据合并过程中TH场景被过滤掉了")
    print("   - 或者在网格搜索过程中TH场景没有满足其他条件")
    
    # 检查网格搜索的阈值
    print("\n=== 检查网格搜索阈值 ===")
    print("当前min_samples=10，但可能还有其他过滤条件")
    
    return df

if __name__ == '__main__':
    import os
    simple_debug()
