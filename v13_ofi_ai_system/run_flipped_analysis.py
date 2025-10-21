#!/usr/bin/env python3
"""
运行翻转信号分析 - 使用最佳参数组合
"""

import sys
sys.path.append('.')

from analysis.ofi_cvd_signal_eval import OFICVDSignalEvaluator

def run_flipped_analysis():
    """运行翻转信号分析"""
    print("=" * 80)
    print("翻转信号分析 - 使用最佳参数组合")
    print("=" * 80)
    
    # 使用最佳参数组合
    evaluator = OFICVDSignalEvaluator(
        data_root='data/ofi_cvd',
        symbols=['ETHUSDT'],
        date_from='2025-10-21',
        date_to='2025-10-21',
        horizons=[60, 180, 300],
        fusion_weights={'w_ofi': 0.6, 'w_cvd': -0.4},  # CVD权重为负，实现翻转
        slices={'regime': ['Active', 'Quiet']},
        output_dir='artifacts/analysis/flipped_optimized',
        run_tag='flipped_optimized_20251021'
    )
    
    print("运行翻转信号分析...")
    print("参数设置:")
    print("  half_life_sec: 120 (最佳)")
    print("  mad_multiplier: 1.2 (最佳)")
    print("  winsor_limit: 4 (最佳)")
    print("  fast_weight: 0.1 (最佳)")
    print("  CVD权重: -0.4 (翻转)")
    print()
    
    try:
        result = evaluator.run_analysis()
        
        print()
        print("=" * 80)
        print("翻转信号分析完成！")
        print("=" * 80)
        
        # 显示关键结果
        if hasattr(evaluator, 'metrics') and evaluator.metrics:
            for symbol, symbol_metrics in evaluator.metrics.items():
                print(f"\n{symbol} 翻转后信号性能:")
                for signal_type, windows in symbol_metrics.items():
                    print(f"  {signal_type}:")
                    for window, window_metrics in windows.items():
                        auc = window_metrics.get('AUC', 0)
                        pr_auc = window_metrics.get('PR_AUC', 0)
                        ic = window_metrics.get('IC', 0)
                        
                        print(f"    {window}: AUC={auc:.4f}, PR-AUC={pr_auc:.4f}, IC={ic:.4f}")
                        
                        # 检查是否达到DoD要求
                        if signal_type == 'fusion' and auc >= 0.58:
                            print(f"      ✅ 达到DoD要求 (AUC >= 0.58)")
                        elif signal_type == 'fusion' and auc >= 0.54:
                            print(f"      ⚠️ 接近DoD要求 (AUC >= 0.54)")
                        elif signal_type == 'cvd' and auc >= 0.54:
                            print(f"      ✅ CVD翻转后达到有效阈值")
        
        if 'dod_status' in result:
            dod_status = result['dod_status']
            print(f"\nDoD Gate检查: {'通过' if dod_status['passed'] else '失败'}")
            if not dod_status['passed']:
                print("问题:")
                for issue in dod_status['issues']:
                    print(f"  - {issue}")
                print("建议:")
                for rec in dod_status['recommendations']:
                    print(f"  - {rec}")
        
        return result
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_flipped_analysis()
