"""
测试实时枢轴检测
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


def test_realtime_pivot():
    """测试实时枢轴检测"""
    print("=== 测试实时枢轴检测 ===\n")
    
    # 创建枢轴检测器
    detector = PivotDetector(window_size=3)
    
    # 创建测试数据 - 明显的峰值和谷值
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
    print(f"窗口大小: 3")
    print(f"需要最少数据点: {3 * 2 + 1} = 7")
    print()
    
    all_pivots = []
    
    # 实时添加数据点并检测枢轴
    for i, (ts, price, indicator) in enumerate(test_data):
        new_pivots_count = detector.add_point_and_detect(ts, price, indicator)
        
        print(f"点 {i}: ts={ts}, price={price}, indicator={indicator}")
        
        if new_pivots_count > 0:
            print(f"  -> 检测到 {new_pivots_count} 个新枢轴")
            # 获取所有枢轴
            all_current_pivots = detector.get_all_pivots()
            if len(all_current_pivots) > 0:
                latest_pivot = all_current_pivots[-1]
                print(f"     最新枢轴: 索引={latest_pivot['index']}, 价格={latest_pivot['price']}, 指标={latest_pivot['indicator']}")
                print(f"               价格高点={latest_pivot['is_price_high']}, 价格低点={latest_pivot['is_price_low']}")
                print(f"               指标高点={latest_pivot['is_indicator_high']}, 指标低点={latest_pivot['is_indicator_low']}")
                all_pivots.append(latest_pivot)
        else:
            print(f"  -> 无新枢轴")
        print()
    
    print(f"总共检测到枢轴数: {len(all_pivots)}")
    
    # 显示所有枢轴
    for i, pivot in enumerate(all_pivots):
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
    
    return len(all_pivots) > 0


if __name__ == '__main__':
    success = test_realtime_pivot()
    if success:
        print("实时枢轴检测工作正常！")
    else:
        print("实时枢轴检测有问题！")
