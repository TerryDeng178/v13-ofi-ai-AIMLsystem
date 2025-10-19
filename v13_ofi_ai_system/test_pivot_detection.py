"""
测试枢轴检测器
"""

import sys
import os
import io
import numpy as np

# 修复Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ofi_cvd_divergence import PivotDetector


def test_pivot_detection():
    """测试枢轴检测器"""
    print("=== 测试枢轴检测器 ===\n")
    
    # 创建枢轴检测器
    detector = PivotDetector(window_size=3)  # 减小窗口大小
    
    # 生成测试数据 - 创建明显的峰值和谷值
    n_samples = 30  # 减少数据点，但确保足够检测枢轴
    timestamps = np.arange(n_samples)
    prices = np.array([100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 104, 103, 102, 101])
    indicators = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])
    
    print(f"数据点数: {n_samples}")
    print(f"价格范围: {prices.min()} - {prices.max()}")
    print(f"指标范围: {indicators.min()} - {indicators.max()}")
    print()
    
    # 添加数据点
    for i in range(n_samples):
        detector.add_point(timestamps[i], prices[i], indicators[i])
        print(f"添加点 {i}: price={prices[i]}, indicator={indicators[i]}")
    
    print()
    
    # 检测枢轴
    pivots = detector.find_pivots()
    print(f"检测到枢轴数: {len(pivots)}")
    
    for i, pivot in enumerate(pivots):
        print(f"枢轴 {i+1}:")
        print(f"  索引: {pivot['index']}")
        print(f"  时间: {pivot['ts']}")
        print(f"  价格: {pivot['price']}")
        print(f"  指标: {pivot['indicator']}")
        print(f"  价格高点: {pivot['is_price_high']}")
        print(f"  价格低点: {pivot['is_price_low']}")
        print(f"  指标高点: {pivot['is_indicator_high']}")
        print(f"  指标低点: {pivot['is_indicator_low']}")
        print()
    
    return len(pivots) > 0


if __name__ == '__main__':
    success = test_pivot_detection()
    if success:
        print("枢轴检测器工作正常！")
    else:
        print("枢轴检测器有问题！")
