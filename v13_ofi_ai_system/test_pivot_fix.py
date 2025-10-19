"""
测试枢轴修复效果
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


def test_pivot_fix():
    """测试枢轴修复效果"""
    print("=== 测试枢轴修复效果 ===\n")
    
    # 创建配置 - 使用建议的参数
    config = DivergenceConfig(
        swing_L=6,  # 建议参数
        min_separation=3,
        cooldown_secs=1.0,
        warmup_min=10,
        z_hi=1.8,  # 建议参数
        z_mid=0.8,
        weak_threshold=35.0,
        use_fusion=False
    )
    
    detector = DivergenceDetector(config)
    
    # 创建明显的背离数据
    n_samples = 100
    timestamps = np.arange(n_samples) + 1.0
    
    # 价格：明显的V型模式
    prices = []
    z_ofi = []
    z_cvd = []
    
    for i in range(n_samples):
        if i < 30:
            # 上升阶段
            price = 100 + i * 0.5
            ofi = -2.0 + i * 0.1  # OFI与价格反向
            cvd = -2.0 + i * 0.1  # CVD与价格反向
        elif i < 70:
            # 下降阶段
            price = 115 - (i - 30) * 0.5
            ofi = 1.0 - (i - 30) * 0.05  # OFI与价格反向
            cvd = 1.0 - (i - 30) * 0.05  # CVD与价格反向
        else:
            # 再次上升阶段
            price = 95 + (i - 70) * 0.3
            ofi = -1.0 + (i - 70) * 0.03  # OFI与价格反向
            cvd = -1.0 + (i - 70) * 0.03  # CVD与价格反向
        
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
        
        # 每20个点报告一次
        if (i + 1) % 20 == 0:
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
    
    if len(ofi_pivots) > 0:
        print(f"OFI枢轴示例: 索引={ofi_pivots[0]['index']}, 价格={ofi_pivots[0]['price']:.2f}")
    if len(cvd_pivots) > 0:
        print(f"CVD枢轴示例: 索引={cvd_pivots[0]['index']}, 价格={cvd_pivots[0]['price']:.2f}")
    
    return len(events) > 0


if __name__ == '__main__':
    success = test_pivot_fix()
    if success:
        print("\n✅ 枢轴修复测试成功！")
    else:
        print("\n❌ 枢轴修复测试失败！")
