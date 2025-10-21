#!/usr/bin/env python3
"""
参数调优脚本 - 实施排障→提效方案
"""

import sys
sys.path.append('.')

from analysis.ofi_cvd_signal_eval import OFICVDSignalEvaluator
import pandas as pd
import numpy as np
from datetime import datetime

def run_parameter_grid():
    """运行参数网格搜索"""
    print("=" * 80)
    print("参数调优 - 排障→提效方案")
    print("=" * 80)
    
    # 参数网格（按您的建议）
    param_grid = {
        'half_life_sec': [120, 300, 600, 1200],
        'mad_multiplier': [1.2, 1.5, 1.8, 2.2],
        'winsor_limit': [4, 6, 8],
        'fast_weight': [0.10, 0.20, 0.40]
    }
    
    results = []
    total_combinations = len(param_grid['half_life_sec']) * len(param_grid['mad_multiplier']) * len(param_grid['winsor_limit']) * len(param_grid['fast_weight'])
    current = 0
    
    print(f"总共需要测试 {total_combinations} 种参数组合...")
    print()
    
    for half_life in param_grid['half_life_sec']:
        for mad_mult in param_grid['mad_multiplier']:
            for winsor in param_grid['winsor_limit']:
                for fast_weight in param_grid['fast_weight']:
                    current += 1
                    print(f"[{current}/{total_combinations}] 测试参数组合:")
                    print(f"  half_life_sec: {half_life}")
                    print(f"  mad_multiplier: {mad_mult}")
                    print(f"  winsor_limit: {winsor}")
                    print(f"  fast_weight: {fast_weight}")
                    
                    try:
                        # 创建评估器
                        evaluator = OFICVDSignalEvaluator(
                            data_root='data/ofi_cvd',
                            symbols=['ETHUSDT'],
                            date_from='2025-10-21',
                            date_to='2025-10-21',
                            horizons=[60, 180, 300],
                            fusion_weights={'w_ofi': 0.5, 'w_cvd': 0.5},  # 先等权重
                            slices={'regime': ['Active', 'Quiet']},
                            output_dir=f'artifacts/analysis/param_tuning/hl{half_life}_mad{mad_mult}_w{winsor}_fw{fast_weight}',
                            run_tag=f'param_tuning_hl{half_life}_mad{mad_mult}_w{winsor}_fw{fast_weight}'
                        )
                        
                        # 运行分析
                        result = evaluator.run_analysis()
                        
                        # 提取关键指标
                        if hasattr(evaluator, 'metrics') and evaluator.metrics:
                            for symbol, symbol_metrics in evaluator.metrics.items():
                                for signal_type, windows in symbol_metrics.items():
                                    for window, window_metrics in windows.items():
                                        if 'AUC' in window_metrics:
                                            results.append({
                                                'half_life_sec': half_life,
                                                'mad_multiplier': mad_mult,
                                                'winsor_limit': winsor,
                                                'fast_weight': fast_weight,
                                                'symbol': symbol,
                                                'signal_type': signal_type,
                                                'window': window,
                                                'AUC': window_metrics.get('AUC', np.nan),
                                                'PR_AUC': window_metrics.get('PR_AUC', np.nan),
                                                'IC': window_metrics.get('IC', np.nan),
                                                'AUC_direction_delta': window_metrics.get('AUC_direction_delta', np.nan),
                                                'IC_direction_delta': window_metrics.get('IC_direction_delta', np.nan),
                                                'direction_suggestion': window_metrics.get('direction_suggestion', 'unknown'),
                                                'winsor_hit_rate': window_metrics.get('winsor_hit_rate', np.nan),
                                                'sigma_floor_hit_rate': window_metrics.get('sigma_floor_hit_rate', np.nan),
                                                'top_bottom_5pct_delta': window_metrics.get('top_bottom_5pct_delta', np.nan)
                                            })
                        
                        print(f"  [OK] 完成")
                        
                    except Exception as e:
                        print(f"  [ERROR] 参数组合失败: {e}")
                        continue
                    
                    print()
    
    # 保存结果
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('artifacts/analysis/param_tuning_results.csv', index=False)
        
        print("=" * 80)
        print("参数调优结果分析")
        print("=" * 80)
        
        # 找出最佳参数组合
        fusion_results = results_df[results_df['signal_type'] == 'fusion']
        if not fusion_results.empty:
            best_fusion = fusion_results.loc[fusion_results['AUC'].idxmax()]
            print(f"最佳Fusion AUC: {best_fusion['AUC']:.4f}")
            print(f"最佳参数组合:")
            print(f"  half_life_sec: {best_fusion['half_life_sec']}")
            print(f"  mad_multiplier: {best_fusion['mad_multiplier']}")
            print(f"  winsor_limit: {best_fusion['winsor_limit']}")
            print(f"  fast_weight: {best_fusion['fast_weight']}")
            print(f"  方向建议: {best_fusion['direction_suggestion']}")
            print(f"  winsor命中率: {best_fusion['winsor_hit_rate']:.2%}")
            print(f"  sigma_floor命中率: {best_fusion['sigma_floor_hit_rate']:.2%}")
        
        # 分析方向问题
        print("\n方向问题分析:")
        direction_issues = results_df[results_df['direction_suggestion'] == 'flip']
        if not direction_issues.empty:
            print(f"发现 {len(direction_issues)} 个信号需要翻转")
            for signal_type in direction_issues['signal_type'].unique():
                signal_direction = direction_issues[direction_issues['signal_type'] == signal_type]
                print(f"  {signal_type}: {len(signal_direction)} 个窗口建议翻转")
        
        # 分析尺度钳制问题
        print("\n尺度钳制分析:")
        high_winsor = results_df[results_df['winsor_hit_rate'] > 0.5]
        high_floor = results_df[results_df['sigma_floor_hit_rate'] > 0.4]
        print(f"winsor命中率>50%的组合: {len(high_winsor)}")
        print(f"sigma_floor命中率>40%的组合: {len(high_floor)}")
        
        print(f"\n结果已保存到: artifacts/analysis/param_tuning_results.csv")
        
    else:
        print("没有成功的结果")

def run_direction_verification():
    """快速验证方向/符号假设"""
    print("=" * 80)
    print("方向验证 - 30分钟内见分晓")
    print("=" * 80)
    
    # 使用建议的临时配置
    evaluator = OFICVDSignalEvaluator(
        data_root='data/ofi_cvd',
        symbols=['ETHUSDT'],
        date_from='2025-10-21',
        date_to='2025-10-21',
        horizons=[60, 180, 300],
        fusion_weights={'w_ofi': 0.5, 'w_cvd': 0.5},
        slices={'regime': ['Active', 'Quiet']},
        output_dir='artifacts/analysis/direction_verification',
        run_tag='direction_verification_20251021'
    )
    
    print("运行方向验证分析...")
    result = evaluator.run_analysis()
    
    print("\n方向验证结果:")
    if hasattr(evaluator, 'metrics') and evaluator.metrics:
        for symbol, symbol_metrics in evaluator.metrics.items():
            print(f"\n{symbol} 信号方向分析:")
            for signal_type, windows in symbol_metrics.items():
                print(f"  {signal_type}:")
                for window, window_metrics in windows.items():
                    auc_delta = window_metrics.get('AUC_direction_delta', 0)
                    ic_delta = window_metrics.get('IC_direction_delta', 0)
                    suggestion = window_metrics.get('direction_suggestion', 'unknown')
                    
                    print(f"    {window}: AUC差值={auc_delta:.4f}, IC差值={ic_delta:.4f}, 建议={suggestion}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='参数调优脚本')
    parser.add_argument('--mode', choices=['direction', 'grid'], default='direction', 
                       help='运行模式: direction=方向验证, grid=参数网格搜索')
    
    args = parser.parse_args()
    
    if args.mode == 'direction':
        run_direction_verification()
    else:
        run_parameter_grid()
