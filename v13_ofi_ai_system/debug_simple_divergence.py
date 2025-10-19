"""
简单背离检测调试
使用明显的数据模式测试枢轴检测
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

from ofi_cvd_divergence import DivergenceDetector, DivergenceConfig


def create_obvious_data():
    """创建明显的数据模式"""
    # 创建明显的峰值和谷值
    n_samples = 50
    timestamps = np.arange(n_samples) + 1.0  # 从1.0开始，避免ts=0
    
    # 价格：明显的峰值和谷值
    prices = np.array([
        100, 101, 102, 103, 104, 105, 104, 103, 102, 101,  # 上升然后下降
        100, 99, 98, 97, 96, 95, 96, 97, 98, 99,           # 下降然后上升
        100, 101, 102, 103, 104, 105, 104, 103, 102, 101,  # 重复模式
        100, 99, 98, 97, 96, 95, 96, 97, 98, 99,           # 重复模式
        100, 101, 102, 103, 104, 105, 104, 103, 102, 101   # 重复模式
    ])
    
    # OFI：与价格反向（背离）
    z_ofi = np.array([
        -1, -2, -3, -4, -5, -4, -3, -2, -1, 0,             # 与价格反向
        1, 2, 3, 4, 5, 4, 3, 2, 1, 0,                      # 与价格反向
        -1, -2, -3, -4, -5, -4, -3, -2, -1, 0,             # 重复模式
        1, 2, 3, 4, 5, 4, 3, 2, 1, 0,                      # 重复模式
        -1, -2, -3, -4, -5, -4, -3, -2, -1, 0              # 重复模式
    ])
    
    # CVD：与价格反向（背离）
    z_cvd = z_ofi.copy()  # 与OFI相同
    
    return timestamps, prices, z_ofi, z_cvd


def test_simple_divergence():
    """测试简单背离检测"""
    print("=== 简单背离检测调试 ===\n")
    
    # 创建配置
    config = DivergenceConfig(
        swing_L=2,  # 最小窗口
        min_separation=1,  # 最小分离
        cooldown_secs=0.1,  # 短冷却
        warmup_min=3,  # 进一步减小预热
        z_hi=1.0,  # 降低阈值
        z_mid=0.5,
        weak_threshold=20.0,
        use_fusion=False  # 先不用融合
    )
    
    detector = DivergenceDetector(config)
    
    # 创建测试数据
    timestamps, prices, z_ofi, z_cvd = create_obvious_data()
    
    print(f"数据点数: {len(timestamps)}")
    print(f"价格范围: {prices.min()} - {prices.max()}")
    print(f"OFI Z范围: {z_ofi.min()} - {z_ofi.max()}")
    print(f"枢轴检测窗口: {config.swing_L}")
    print(f"需要最少数据点: {config.swing_L * 2 + 1}")
    print()
    
    # 逐步添加数据点并检测
    events = []
    pivot_counts = []
    
    for i in range(len(timestamps)):
        ts = timestamps[i]
        price = prices[i]
        ofi = z_ofi[i]
        cvd = z_cvd[i]
        
        # 更新背离检测器
        event = detector.update(ts, price, ofi, cvd, lag_sec=0.1)
        
        # 统计枢轴数
        total_pivots = (len(detector.price_ofi_detector.price_buffer) >= config.swing_L * 2 + 1 and 
                       len(detector.price_ofi_detector.find_pivots()) +
                       len(detector.price_cvd_detector.find_pivots()))
        
        pivot_counts.append(total_pivots)
        
        if event:
            events.append(event)
            print(f"点 {i}: 检测到事件 - {event['type']}")
            print(f"  时间: {event['ts']}")
            print(f"  价格: {event['pivots']['price']}")
            print(f"  分数: {event['score']}")
            print(f"  原因: {event['reason_codes']}")
            print()
        
        # 每10个点报告一次
        if (i + 1) % 10 == 0:
            print(f"样本 {i+1}: 枢轴数={total_pivots}, 事件数={len(events)}")
    
    print(f"\n=== 最终结果 ===")
    print(f"总事件数: {len(events)}")
    print(f"枢轴检测统计: {detector._stats}")
    
    # 分析事件类型
    event_types = {}
    for event in events:
        event_type = event['type']
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    print(f"事件类型分布:")
    for event_type, count in event_types.items():
        print(f"  {event_type}: {count}")
    
    # 检查枢轴检测
    ofi_pivots = detector.price_ofi_detector.find_pivots()
    cvd_pivots = detector.price_cvd_detector.find_pivots()
    
    print(f"\n枢轴检测结果:")
    print(f"OFI枢轴数: {len(ofi_pivots)}")
    print(f"CVD枢轴数: {len(cvd_pivots)}")
    
    if len(ofi_pivots) > 0:
        print(f"OFI枢轴示例: {ofi_pivots[0]}")
    if len(cvd_pivots) > 0:
        print(f"CVD枢轴示例: {cvd_pivots[0]}")
    
    return len(events) > 0


if __name__ == '__main__':
    success = test_simple_divergence()
    if success:
        print("\n✅ 简单背离检测测试成功！")
    else:
        print("\n❌ 简单背离检测测试失败！")
