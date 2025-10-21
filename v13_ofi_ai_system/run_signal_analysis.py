#!/usr/bin/env python3
"""
运行OFI+CVD信号分析
"""

import sys
sys.path.append('.')

from analysis.ofi_cvd_signal_eval import OFICVDSignalEvaluator

def main():
    print("=" * 60)
    print("OFI+CVD信号分析工具 - 基于修复后的组件")
    print("=" * 60)
    
    # 创建评估器
    evaluator = OFICVDSignalEvaluator(
        data_root='data/ofi_cvd',
        symbols=['ETHUSDT'],  # 使用ETHUSDT，因为BTCUSDT的OFI数据质量较差
        date_from='2025-10-21',
        date_to='2025-10-21',
        horizons=[60, 180, 300],  # 1分钟、3分钟、5分钟前瞻窗口
        fusion_weights={'w_ofi': 0.6, 'w_cvd': 0.4},
        slices={'regime': ['Active', 'Quiet']},
        output_dir='artifacts/analysis/ofi_cvd',
        run_tag='eth_analysis_fixed_20251021'
    )
    
    print("开始OFI+CVD信号分析...")
    print(f"数据源: {evaluator.data_root}")
    print(f"交易对: {evaluator.symbols}")
    print(f"时间窗口: {evaluator.horizons}")
    print(f"输出目录: {evaluator.output_dir}")
    print()
    
    try:
        # 运行完整分析流程
        results = evaluator.run_analysis()
        
        print()
        print("=" * 60)
        print("分析完成！")
        print("=" * 60)
        print(f"结果已保存到: {evaluator.output_dir}")
        print(f"运行标签: {evaluator.run_tag}")
        
        # 显示关键结果
        if 'dod_status' in results:
            dod_status = results['dod_status']
            print(f"\nDoD Gate检查: {'通过' if dod_status['passed'] else '失败'}")
            if not dod_status['passed']:
                print("问题:")
                for issue in dod_status['issues']:
                    print(f"  - {issue}")
                print("建议:")
                for rec in dod_status['recommendations']:
                    print(f"  - {rec}")
        
        return results
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
