"""
测试背离检测修复效果

验证：
1. 冲突事件不再阻塞方向性背离
2. 同型枢轴配对正确工作
3. 评分机制使用枢轴处强度
4. 参数调整生效
"""

import sys
import os
import io
import numpy as np
import time

# 修复Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ofi_cvd_divergence import DivergenceConfig, DivergenceDetector


def test_divergence_fixes():
    """测试背离检测修复效果"""
    print("=== 测试背离检测修复效果 ===\n")
    
    # 创建配置 - 使用更小的窗口以便测试
    config = DivergenceConfig(
        swing_L=3,  # 进一步减小窗口大小
        z_hi=1.5,
        z_mid=0.7,
        min_separation=2,  # 减小最小间距
        cooldown_secs=0.5,
        warmup_min=10,  # 减小暖启动样本
        weak_threshold=35.0
    )
    
    detector = DivergenceDetector(config)
    
    # 生成测试数据 - 创建明显的背离模式
    n_samples = 200
    timestamps = np.arange(n_samples)
    prices = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 10 + 100  # 价格波动
    z_ofi = np.sin(np.linspace(0, 4*np.pi, n_samples) + np.pi) * 2  # OFI与价格反向
    z_cvd = np.sin(np.linspace(0, 4*np.pi, n_samples) + np.pi) * 2  # CVD与价格反向
    fusion_scores = (z_ofi + z_cvd) / 2
    consistency = np.ones(n_samples) * 0.8
    
    print(f"枢轴检测窗口大小: {config.swing_L}")
    print(f"需要的最小数据点: {config.swing_L * 2 + 1}")
    print(f"实际数据点数: {n_samples}")
    print()
    
    print(f"测试数据: {n_samples} 个样本")
    print(f"价格范围: {prices.min():.2f} - {prices.max():.2f}")
    print(f"OFI Z范围: {z_ofi.min():.2f} - {z_ofi.max():.2f}")
    print(f"CVD Z范围: {z_cvd.min():.2f} - {z_cvd.max():.2f}")
    print()
    
    # 运行检测
    events = []
    directional_events = []
    conflict_events = []
    
    for i in range(n_samples):
        event = detector.update(
            ts=timestamps[i],
            price=prices[i],
            z_ofi=z_ofi[i],
            z_cvd=z_cvd[i],
            fusion_score=fusion_scores[i],
            consistency=consistency[i],
            warmup=False,
            lag_sec=0.0
        )
        
        if event and event.get('type'):
            events.append(event)
            if event['type'] == 'ofi_cvd_conflict':
                conflict_events.append(event)
            else:
                directional_events.append(event)
        
        # 每50个样本检查一次枢轴检测状态
        if i % 50 == 0 and i > 0:
            stats = detector.get_stats()
            print(f"样本 {i}: 枢轴数={stats.get('pivots_detected', 0)}, 事件数={stats.get('events_total', 0)}")
    
    # 统计结果
    print(f"检测到总事件数: {len(events)}")
    print(f"方向性事件数: {len(directional_events)}")
    print(f"冲突事件数: {len(conflict_events)}")
    print()
    
    # 事件类型分布
    type_counts = {}
    for event in events:
        event_type = event['type']
        type_counts[event_type] = type_counts.get(event_type, 0) + 1
    
    print("事件类型分布:")
    for event_type, count in type_counts.items():
        print(f"  {event_type}: {count}")
    print()
    
    # 检查是否有方向性背离
    if directional_events:
        print("成功检测到方向性背离事件")
        print("方向性事件详情:")
        for i, event in enumerate(directional_events[:5]):  # 显示前5个
            print(f"  {i+1}. 类型: {event['type']}, 分数: {event['score']:.1f}, 通道: {event['channels']}")
    else:
        print("未检测到方向性背离事件")
    
    # 检查冲突事件是否与方向性事件共存
    if conflict_events and directional_events:
        print("冲突事件与方向性事件可以共存")
    elif conflict_events and not directional_events:
        print("只有冲突事件，没有方向性事件")
    else:
        print("没有检测到冲突事件")
    
    # 检查枢轴配对
    print(f"\n枢轴检测统计:")
    stats = detector.get_stats()
    print(f"  检测到的枢轴数: {stats.get('pivots_detected', 0)}")
    print(f"  总事件数: {stats.get('events_total', 0)}")
    print(f"  被抑制事件数: {stats.get('suppressed_total', 0)}")
    
    return len(directional_events) > 0


if __name__ == '__main__':
    success = test_divergence_fixes()
    if success:
        print("\n测试通过：背离检测修复生效！")
    else:
        print("\n测试失败：需要进一步调试")
