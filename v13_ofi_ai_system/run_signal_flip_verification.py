#!/usr/bin/env python3
"""
信号翻转验证 - 验证方向修复效果
"""

import sys
sys.path.append('.')

from analysis.ofi_cvd_signal_eval import OFICVDSignalEvaluator
import pandas as pd
import numpy as np

def run_flip_verification():
    """验证信号翻转效果"""
    print("=" * 80)
    print("信号翻转验证 - 验证方向修复效果")
    print("=" * 80)
    
    # 创建评估器（使用翻转后的信号）
    evaluator = OFICVDSignalEvaluator(
        data_root='data/ofi_cvd',
        symbols=['ETHUSDT'],
        date_from='2025-10-21',
        date_to='2025-10-21',
        horizons=[60, 180, 300],
        fusion_weights={'w_ofi': 0.6, 'w_cvd': 0.4},  # 保持原权重
        slices={'regime': ['Active', 'Quiet']},
        output_dir='artifacts/analysis/flip_verification',
        run_tag='flip_verification_20251021'
    )
    
    print("运行翻转验证分析...")
    result = evaluator.run_analysis()
    
    # 手动翻转CVD和Fusion信号进行验证
    print("\n手动翻转验证结果:")
    print("=" * 60)
    
    if hasattr(evaluator, 'metrics') and evaluator.metrics:
        for symbol, symbol_metrics in evaluator.metrics.items():
            print(f"\n{symbol} 翻转后信号性能:")
            for signal_type, windows in symbol_metrics.items():
                print(f"  {signal_type}:")
                for window, window_metrics in windows.items():
                    original_auc = window_metrics.get('AUC', 0)
                    flipped_auc = window_metrics.get('AUC_flipped', 0)
                    improvement = flipped_auc - original_auc
                    
                    print(f"    {window}: 原始AUC={original_auc:.4f}, 翻转AUC={flipped_auc:.4f}, 提升={improvement:.4f}")
                    
                    # 如果翻转后AUC > 0.54，说明翻转有效
                    if flipped_auc > 0.54:
                        print(f"      [OK] 翻转后达到有效阈值 (>{0.54})")
                    elif improvement > 0.05:
                        print(f"      [WARN] 翻转有明显改善 (>{0.05})")
                    else:
                        print(f"      [FAIL] 翻转改善有限")

def create_flipped_analysis():
    """创建翻转信号的分析版本"""
    print("\n" + "=" * 80)
    print("创建翻转信号分析版本")
    print("=" * 80)
    
    # 这里我们需要修改信号提取逻辑，在实际应用中翻转CVD和Fusion信号
    # 由于当前架构限制，我们通过修改权重来实现类似效果
    
    # 对于CVD信号翻转，我们可以调整融合权重
    evaluator = OFICVDSignalEvaluator(
        data_root='data/ofi_cvd',
        symbols=['ETHUSDT'],
        date_from='2025-10-21',
        date_to='2025-10-21',
        horizons=[60, 180, 300],
        fusion_weights={'w_ofi': 0.6, 'w_cvd': -0.4},  # CVD权重为负，实现翻转
        slices={'regime': ['Active', 'Quiet']},
        output_dir='artifacts/analysis/flipped_signals',
        run_tag='flipped_signals_20251021'
    )
    
    print("运行翻转信号分析...")
    result = evaluator.run_analysis()
    
    print("\n翻转信号分析完成！")
    print("结果已保存到: artifacts/analysis/flipped_signals")
    
    return result

if __name__ == "__main__":
    # 先运行翻转验证
    run_flip_verification()
    
    # 再创建翻转信号分析
    create_flipped_analysis()
