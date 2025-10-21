"""
调试枢轴检测器
"""

import sys
import os
import io

# 修复Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ofi_cvd_divergence import PivotDetector


def debug_pivot_detection():
    """调试枢轴检测器"""
    print("=== 调试枢轴检测器 ===\n")
    
    # 创建枢轴检测器
    detector = PivotDetector(window_size=2)  # 最小窗口
    
    # 创建简单的测试数据：明显的峰值和谷值
    test_data = [
        (0, 100, 1),   # 0
        (1, 101, 2),   # 1
        (2, 102, 3),   # 2
        (3, 103, 4),   # 3
        (4, 104, 5),   # 4
        (5, 105, 4),   # 5 - 价格高点
        (6, 104, 3),   # 6
        (7, 103, 2),   # 7
        (8, 102, 1),   # 8
        (9, 101, 0),   # 9
        (10, 100, -1), # 10
        (11, 99, -2),  # 11
        (12, 98, -3),  # 12
        (13, 97, -4),  # 13
        (14, 96, -5),  # 14 - 价格低点
        (15, 97, -4),  # 15
        (16, 98, -3),  # 16
        (17, 99, -2),  # 17
        (18, 100, -1), # 18
        (19, 101, 0),  # 19
    ]
    
    print(f"数据点数: {len(test_data)}")
    print(f"窗口大小: 2")
    print(f"需要最少数据点: {2 * 2 + 1} = 5")
    print()
    
    # 添加数据点
    for i, (ts, price, indicator) in enumerate(test_data):
        detector.add_point(ts, price, indicator)
        print(f"点 {i}: ts={ts}, price={price}, indicator={indicator}")
        
        # 检查是否可以开始检测枢轴
        if len(detector.price_buffer) >= 5:
            print(f"  -> 可以开始检测枢轴 (缓冲区大小: {len(detector.price_buffer)})")
            
            # 手动检查枢轴条件
            prices = list(detector.price_buffer)
            indicators = list(detector.indicator_buffer)
            
            print(f"  -> 价格序列: {prices}")
            print(f"  -> 指标序列: {indicators}")
            
            # 检查每个可能的枢轴点
            for j in range(2, len(prices) - 2):
                left_prices = prices[j-2:j]
                right_prices = prices[j+1:j+3]
                current_price = prices[j]
                
                is_high = (current_price >= max(left_prices) and current_price >= max(right_prices))
                is_low = (current_price <= min(left_prices) and current_price <= min(right_prices))
                
                print(f"  -> 点 {j}: price={current_price}, left={left_prices}, right={right_prices}")
                print(f"     is_high={is_high}, is_low={is_low}")
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
    success = debug_pivot_detection()
    if success:
        print("枢轴检测器工作正常！")
    else:
        print("枢轴检测器有问题！")
