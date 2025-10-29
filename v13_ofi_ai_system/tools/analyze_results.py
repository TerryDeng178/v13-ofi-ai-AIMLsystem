#!/usr/bin/env python3
"""
分析2x2场景化参数优化结果
"""
import pandas as pd
import yaml
import os

def analyze_results():
    """分析优化结果"""
    results_dir = "outputs/grid_search/20251022_193741"
    
    print("=== 2x2场景化参数优化结果分析 ===")
    
    # 读取最优参数
    with open("outputs/best_params.yaml", 'r', encoding='utf-8') as f:
        best_params = yaml.safe_load(f)
    
    print(f"生成时间: {best_params['metadata']['generated_at']}")
    print(f"总交易对数: {best_params['metadata']['total_regimes']}")
    print(f"可用交易对: {best_params['metadata']['available_regimes']}")
    
    print("\n=== 各交易对最优参数 ===")
    for symbol in best_params['metadata']['available_regimes']:
        symbol_params = best_params['cvd']['regimes'][symbol]
        print(f"\n{symbol}:")
        for regime, params in symbol_params.items():
            print(f"  {regime}: {params}")
    
    # 分析每个交易对的详细结果
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
    
    print("\n=== 详细结果分析 ===")
    for symbol in symbols:
        results_file = os.path.join(results_dir, f"results_{symbol.lower()}.csv")
        if os.path.exists(results_file):
            try:
                df = pd.read_csv(results_file)
                print(f"\n{symbol}:")
                print(f"  总结果数: {len(df)}")
                print("  场景分布:")
                regime_counts = df['regime'].value_counts()
                for regime, count in regime_counts.items():
                    print(f"    {regime}: {count} 个参数组合")
                
                # 显示每个场景的最佳参数
                print("  各场景最优参数:")
                for regime in df['regime'].unique():
                    regime_df = df[df['regime'] == regime]
                    best_idx = regime_df['score_reg'].idxmax()
                    best_row = regime_df.loc[best_idx]
                    print(f"    {regime}: score={best_row['score_reg']:.4f}, "
                          f"IR={best_row['IR_after_cost']:.4f}, "
                          f"ewm_span={best_row['ewm_span']}, "
                          f"z_window={best_row['z_window']}, "
                          f"w_cvd={best_row['w_cvd']}")
                    
            except Exception as e:
                print(f"{symbol}: 读取失败 - {e}")
        else:
            print(f"{symbol}: 结果文件不存在")

if __name__ == '__main__':
    analyze_results()




