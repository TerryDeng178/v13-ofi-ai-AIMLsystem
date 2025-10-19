"""
最终背离检测测试 - 确保有足够的同型枢轴
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


def test_divergence_final():
    """最终背离检测测试"""
    print("=== 最终背离检测测试 ===\n")
    
    # 创建配置 - 使用建议的参数
    config = DivergenceConfig(
        swing_L=5,  # 建议参数
        min_separation=3,
        cooldown_secs=1.0,
        warmup_min=10,
        z_hi=1.5,  # 建议参数
        z_mid=0.7,
        weak_threshold=35.0,
        use_fusion=False
    )
    
    detector = DivergenceDetector(config)
    
    # 创建明显的背离数据 - 确保有多个同型枢轴
    n_samples = 200
    timestamps = np.arange(n_samples) + 1.0
    
    # 价格：创建多个明显的峰值和谷值
    prices = []
    z_ofi = []
    z_cvd = []
    
    for i in range(n_samples):
        # 创建多个周期，确保有多个高点和低点
        cycle = i // 40  # 每40个点一个周期
        phase = (i % 40) / 40.0 * 2 * np.pi  # 0到2π
        
        if cycle < 4:  # 前4个周期
            # 价格：正弦波模式，确保有多个高点和低点
            price = 100 + 10 * np.sin(phase) + cycle * 2
            # OFI：与价格反向，确保背离
            ofi = -2 * np.sin(phase) + cycle * 0.5
            cvd = -2 * np.sin(phase) + cycle * 0.5
        else:
            # 最后周期：更复杂的模式
            price = 100 + 8 * np.sin(phase) + 5 * np.sin(phase * 2) + cycle * 2
            ofi = -1.5 * np.sin(phase) - 0.5 * np.sin(phase * 2) + cycle * 0.5
            cvd = -1.5 * np.sin(phase) - 0.5 * np.sin(phase * 2) + cycle * 0.5
        
        prices.append(price)
        z_ofi.append(ofi)
        z_cvd.append(cvd)
    
    print(f"数据点数: {n_samples}")
    print(f"价格范围: {min(prices):.2f} - {max(prices):.2f}")
    print(f"OFI Z范围: {min(z_ofi):.2f} - {max(z_ofi):.2f}")
    print(f"枢轴检测窗口: {config.swing_L}")
    print(f"需要最少数据点: {config.swing_L * 2 + 1}")
    print()
    
    # 逐步添加数据点并检测
    events = []
    pivot_counts = []
    
    for i in range(n_samples):
        ts = timestamps[i]
        price = prices[i]
        ofi = z_ofi[i]
        cvd = z_cvd[i]
        
        # 更新背离检测器
        event = detector.update(ts, price, ofi, cvd, lag_sec=0.1)
        
        # 统计枢轴数
        ofi_pivots = detector.price_ofi_detector.get_all_pivots()
        cvd_pivots = detector.price_cvd_detector.get_all_pivots()
        total_pivots = len(ofi_pivots) + len(cvd_pivots)
        
        pivot_counts.append(total_pivots)
        
        if event and event['type'] is not None:
            events.append(event)
            print(f"点 {i}: 检测到事件 - {event['type']}")
            print(f"  时间: {event['ts']}")
            print(f"  分数: {event['score']}")
            print(f"  原因: {event['reason_codes']}")
            print()
        
        # 每50个点报告一次
        if (i + 1) % 50 == 0:
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
    ofi_pivots = detector.price_ofi_detector.get_all_pivots()
    cvd_pivots = detector.price_cvd_detector.get_all_pivots()
    
    print(f"\n枢轴检测结果:")
    print(f"OFI枢轴数: {len(ofi_pivots)}")
    print(f"CVD枢轴数: {len(cvd_pivots)}")
    
    # 分析同型枢轴
    ofi_lows = [p for p in ofi_pivots if p.get('is_price_low')]
    ofi_highs = [p for p in ofi_pivots if p.get('is_price_high')]
    cvd_lows = [p for p in cvd_pivots if p.get('is_price_low')]
    cvd_highs = [p for p in cvd_pivots if p.get('is_price_high')]
    
    print(f"OFI价格低点数: {len(ofi_lows)}")
    print(f"OFI价格高点数: {len(ofi_highs)}")
    print(f"CVD价格低点数: {len(cvd_lows)}")
    print(f"CVD价格高点数: {len(cvd_highs)}")
    
    if len(ofi_pivots) > 0:
        print(f"OFI枢轴示例: 索引={ofi_pivots[0]['index']}, 价格={ofi_pivots[0]['price']:.2f}")
    if len(cvd_pivots) > 0:
        print(f"CVD枢轴示例: 索引={cvd_pivots[0]['index']}, 价格={cvd_pivots[0]['price']:.2f}")
    
    return len(events) > 0


if __name__ == '__main__':
    success = test_divergence_final()
    if success:
        print("\n✅ 最终背离检测测试成功！")
    else:
        print("\n❌ 最终背离检测测试失败！")
