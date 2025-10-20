#!/usr/bin/env python3
"""
分析超精细搜索结果
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_ultra_fine_results():
    """分析超精细搜索结果"""
    test_dir = Path('data/cvd_ultra_fine_search')
    results = []
    
    print("超精细搜索结果分析:")
    print("="*60)
    
    for test_folder in sorted(test_dir.glob('test_*')):
        parquet_files = list(test_folder.glob('*.parquet'))
        if parquet_files:
            try:
                df = pd.read_parquet(parquet_files[0])
                z_valid = df[df['z_cvd'].notna()]['z_cvd']
                
                if len(z_valid) > 0:
                    p_z_gt_2 = np.mean(np.abs(z_valid) > 2)
                    p_z_gt_3 = np.mean(np.abs(z_valid) > 3)
                    median_z = np.median(z_valid)
                    p95_z = np.percentile(z_valid, 95)
                    p99_z = np.percentile(z_valid, 99)
                    
                    # 读取配置文件信息
                    config_name = "unknown"
                    try:
                        # 尝试从测试文件夹名推断配置
                        test_name = test_folder.name
                        # 这里可以根据实际情况调整
                        config_name = test_name
                    except:
                        pass
                    
                    results.append({
                        'test': test_folder.name,
                        'config': config_name,
                        'p_z_gt_2': p_z_gt_2,
                        'p_z_gt_3': p_z_gt_3,
                        'median_z': median_z,
                        'p95_z': p95_z,
                        'p99_z': p99_z,
                        'count': len(z_valid)
                    })
            except Exception as e:
                print(f"Error processing {test_folder}: {e}")
    
    # 按P(|Z|>3)排序
    results.sort(key=lambda x: x['p_z_gt_3'])
    
    print(f"成功分析 {len(results)} 个测试结果")
    print()
    
    # 显示所有结果
    print("所有结果 (按P(|Z|>3)排序):")
    print("-" * 60)
    
    for i, r in enumerate(results):
        print(f"第{i+1}名: {r['test']}")
        print(f"  P(|Z|>2): {r['p_z_gt_2']:.1%}")
        print(f"  P(|Z|>3): {r['p_z_gt_3']:.1%}")
        print(f"  Median(|Z|): {r['median_z']:.3f}")
        print(f"  P95(|Z|): {r['p95_z']:.3f}")
        print(f"  P99(|Z|): {r['p99_z']:.3f}")
        print(f"  有效数据: {r['count']}")
        print()
    
    # 检查达标情况
    if results:
        best = results[0]
        print("最佳结果达标检查:")
        print("-" * 30)
        
        p_z_gt_3_ok = best['p_z_gt_3'] <= 0.015  # 1.5%
        p_z_gt_2_ok = best['p_z_gt_2'] <= 0.05   # 5%
        median_ok = best['median_z'] <= 0.5
        p95_ok = best['p95_z'] <= 2.0
        
        print(f"P(|Z|>3) <= 1.5%: {'PASS' if p_z_gt_3_ok else 'FAIL'} ({best['p_z_gt_3']:.1%})")
        print(f"P(|Z|>2) <= 5.0%: {'PASS' if p_z_gt_2_ok else 'FAIL'} ({best['p_z_gt_2']:.1%})")
        print(f"Median(|Z|) <= 0.5: {'PASS' if median_ok else 'FAIL'} ({best['median_z']:.3f})")
        print(f"P95(|Z|) <= 2.0: {'PASS' if p95_ok else 'FAIL'} ({best['p95_z']:.3f})")
        
        if p_z_gt_3_ok and p_z_gt_2_ok and median_ok and p95_ok:
            print("\n所有指标均达标！")
        else:
            print("\n仍有指标未达标，需要进一步优化")
    
    return results

if __name__ == "__main__":
    analyze_ultra_fine_results()
