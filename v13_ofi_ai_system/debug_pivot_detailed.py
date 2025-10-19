"""
详细调试枢轴检测器
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


def debug_pivot_detailed():
    """详细调试枢轴检测器"""
    print("=== 详细调试枢轴检测器 ===\n")
    
    # 创建枢轴检测器
    detector = PivotDetector(window_size=2)
    
    # 创建测试数据
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
    
    # 添加所有数据点
    for ts, price, indicator in test_data:
        detector.add_point(ts, price, indicator)
    
    print(f"数据点数: {len(test_data)}")
    print(f"窗口大小: 2")
    print(f"缓冲区大小: {len(detector.price_buffer)}")
    print(f"需要最少数据点: {2 * 2 + 1} = 5")
    print()
    
    # 手动实现枢轴检测逻辑
    prices = list(detector.price_buffer)
    indicators = list(detector.indicator_buffer)
    timestamps = list(detector.timestamp_buffer)
    
    print(f"价格序列: {prices}")
    print(f"指标序列: {indicators}")
    print(f"时间序列: {timestamps}")
    print()
    
    print("枢轴检测范围:", f"range({2}, {len(prices) - 2}) = range(2, {len(prices) - 2})")
    print()
    
    pivots = []
    for i in range(2, len(prices) - 2):
        print(f"检查索引 {i}:")
        
        # 价格检查
        left_prices = prices[i - 2:i]
        right_prices = prices[i + 1:i + 3]
        current_price = prices[i]
        
        is_price_high = (len(left_prices) > 0 and len(right_prices) > 0 and 
                        current_price >= max(left_prices) and current_price >= max(right_prices))
        is_price_low = (len(left_prices) > 0 and len(right_prices) > 0 and 
                       current_price <= min(left_prices) and current_price <= min(right_prices))
        
        print(f"  价格: {current_price}, 左侧: {left_prices}, 右侧: {right_prices}")
        print(f"  价格高点: {is_price_high}, 价格低点: {is_price_low}")
        
        # 指标检查
        left_indicators = indicators[i - 2:i]
        right_indicators = indicators[i + 1:i + 3]
        current_indicator = indicators[i]
        
        is_indicator_high = (len(left_indicators) > 0 and len(right_indicators) > 0 and 
                           current_indicator >= max(left_indicators) and current_indicator >= max(right_indicators))
        is_indicator_low = (len(left_indicators) > 0 and len(right_indicators) > 0 and 
                          current_indicator <= min(left_indicators) and current_indicator <= min(right_indicators))
        
        print(f"  指标: {current_indicator}, 左侧: {left_indicators}, 右侧: {right_indicators}")
        print(f"  指标高点: {is_indicator_high}, 指标低点: {is_indicator_low}")
        
        if is_price_high or is_price_low:
            pivot = {
                'index': i,
                'ts': timestamps[i],
                'price': prices[i],
                'indicator': indicators[i],
                'is_price_high': is_price_high,
                'is_price_low': is_price_low,
                'is_indicator_high': is_indicator_high,
                'is_indicator_low': is_indicator_low
            }
            pivots.append(pivot)
            print(f"  -> 发现枢轴: {pivot}")
        else:
            print(f"  -> 不是枢轴")
        print()
    
    print(f"手动检测到枢轴数: {len(pivots)}")
    
    # 使用原始方法检测枢轴
    original_pivots = detector.find_pivots()
    print(f"原始方法检测到枢轴数: {len(original_pivots)}")
    
    if len(original_pivots) != len(pivots):
        print("警告：手动检测和原始方法结果不一致！")
        print(f"手动结果: {pivots}")
        print(f"原始结果: {original_pivots}")
    
    return len(pivots) > 0


if __name__ == '__main__':
    success = debug_pivot_detailed()
    if success:
        print("枢轴检测器工作正常！")
    else:
        print("枢轴检测器有问题！")
