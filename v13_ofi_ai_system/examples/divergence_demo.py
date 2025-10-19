"""
背离检测演示脚本

展示OFI-CVD背离检测模块的基本功能：
- 枢轴检测
- 四种背离类型检测
- OFI-CVD冲突检测
- 评分和去噪机制

Author: V13 OFI+CVD AI System
Created: 2025-01-20
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ofi_cvd_divergence import DivergenceConfig, DivergenceDetector


def create_sample_data():
    """创建示例数据"""
    # 创建价格序列（V型反转）
    n_samples = 200
    timestamps = np.linspace(0, n_samples * 0.001, n_samples)
    
    # 价格：先下降后上升
    prices = np.concatenate([
        np.linspace(100, 95, n_samples // 2),  # 下降
        np.linspace(95, 105, n_samples // 2)   # 上升
    ])
    
    # OFI：与价格相反（背离）
    z_ofi = np.concatenate([
        np.linspace(2, -1, n_samples // 2),    # 下降
        np.linspace(-1, 3, n_samples // 2)     # 上升
    ]) + np.random.normal(0, 0.1, n_samples)
    
    # CVD：与价格相反（背离）
    z_cvd = np.concatenate([
        np.linspace(1.5, -0.5, n_samples // 2),  # 下降
        np.linspace(-0.5, 2.5, n_samples // 2)   # 上升
    ]) + np.random.normal(0, 0.1, n_samples)
    
    # 裁剪到[-5, 5]范围
    z_ofi = np.clip(z_ofi, -5, 5)
    z_cvd = np.clip(z_cvd, -5, 5)
    
    # 融合分数
    fusion_scores = (z_ofi + z_cvd) / 2
    
    # 一致性分数
    consistency = np.abs(z_ofi * z_cvd) / (np.abs(z_ofi) + np.abs(z_cvd) + 1e-8)
    
    return timestamps, prices, z_ofi, z_cvd, fusion_scores, consistency


def run_divergence_detection():
    """运行背离检测演示"""
    print("=== OFI-CVD背离检测演示 ===\n")
    
    # 创建配置
    config = DivergenceConfig(
        swing_L=10,          # 枢轴窗口长度
        z_hi=2.0,           # 高强度阈值
        z_mid=1.0,          # 中等强度阈值
        min_separation=5,   # 最小枢轴间距
        cooldown_secs=1.0,  # 冷却时间
        warmup_min=50,      # 暖启动样本数
        use_fusion=True,    # 使用融合指标
        cons_min=0.3        # 最小一致性阈值
    )
    
    # 创建检测器
    detector = DivergenceDetector(config)
    
    # 生成示例数据
    timestamps, prices, z_ofi, z_cvd, fusion_scores, consistency = create_sample_data()
    
    print(f"数据点数量: {len(timestamps)}")
    print(f"价格范围: {prices.min():.2f} - {prices.max():.2f}")
    print(f"OFI范围: {z_ofi.min():.2f} - {z_ofi.max():.2f}")
    print(f"CVD范围: {z_cvd.min():.2f} - {z_cvd.max():.2f}\n")
    
    # 运行检测
    events = []
    detection_times = []
    
    print("开始背离检测...")
    for i, (ts, price, ofi, cvd, fusion, cons) in enumerate(zip(
        timestamps, prices, z_ofi, z_cvd, fusion_scores, consistency
    )):
        start_time = time.perf_counter()
        
        event = detector.update(
            ts=ts,
            price=price,
            z_ofi=ofi,
            z_cvd=cvd,
            fusion_score=fusion,
            consistency=cons,
            warmup=False,
            lag_sec=0.0
        )
        
        detection_time = time.perf_counter() - start_time
        detection_times.append(detection_time)
        
        if event and event.get('type'):
            events.append({
                'index': i,
                'timestamp': ts,
                'price': price,
                'event': event
            })
            print(f"检测到背离事件 #{len(events)}:")
            print(f"  时间: {ts:.3f}s")
            print(f"  价格: {price:.2f}")
            print(f"  类型: {event['type']}")
            print(f"  评分: {event['score']:.1f}")
            print(f"  通道: {event['channels']}")
            print(f"  原因: {event['reason_codes']}")
            print()
    
    # 显示统计信息
    print("=== 检测统计 ===")
    print(f"总事件数: {len(events)}")
    print(f"平均检测延迟: {np.mean(detection_times)*1000:.3f}ms")
    print(f"P95检测延迟: {np.percentile(detection_times, 95)*1000:.3f}ms")
    print(f"最大检测延迟: {np.max(detection_times)*1000:.3f}ms")
    
    # 按类型统计事件
    type_counts = {}
    for event_info in events:
        event_type = event_info['event']['type']
        type_counts[event_type] = type_counts.get(event_type, 0) + 1
    
    print("\n事件类型分布:")
    for event_type, count in type_counts.items():
        print(f"  {event_type}: {count}")
    
    # 获取检测器统计
    stats = detector.get_stats()
    print(f"\n检测器统计:")
    print(f"  总枢轴数: {stats['pivots_detected']}")
    print(f"  抑制事件数: {stats['suppressed_total']}")
    if stats['suppressed_by_reason']:
        print(f"  抑制原因: {stats['suppressed_by_reason']}")
    
    return timestamps, prices, z_ofi, z_cvd, events


def create_visualization(timestamps, prices, z_ofi, z_cvd, events):
    """创建可视化图表"""
    print("\n生成可视化图表...")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 价格序列和背离事件
    ax1 = axes[0]
    ax1.plot(timestamps, prices, 'b-', linewidth=2, label='Price', alpha=0.8)
    
    # 标记背离事件
    colors = {
        'bull_div': 'green',
        'bear_div': 'red',
        'hidden_bull': 'lightgreen',
        'hidden_bear': 'lightcoral',
        'ofi_cvd_conflict': 'orange'
    }
    
    for event_info in events:
        event = event_info['event']
        color = colors.get(event['type'], 'gray')
        ax1.scatter(event_info['timestamp'], event_info['price'], 
                   c=color, s=100, alpha=0.8, edgecolors='black', linewidth=1)
        ax1.annotate(f"{event['type']}\n{event['score']:.1f}", 
                    (event_info['timestamp'], event_info['price']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, ha='left')
    
    ax1.set_title('Price Series with Divergence Events', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # OFI序列
    ax2 = axes[1]
    ax2.plot(timestamps, z_ofi, 'g-', linewidth=2, label='OFI Z-score', alpha=0.8)
    ax2.axhline(y=2.0, color='g', linestyle='--', alpha=0.5, label='High Threshold')
    ax2.axhline(y=-2.0, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(y=1.0, color='g', linestyle=':', alpha=0.5, label='Mid Threshold')
    ax2.axhline(y=-1.0, color='g', linestyle=':', alpha=0.5)
    ax2.set_title('OFI Z-score', fontsize=12)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('OFI Z-score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # CVD序列
    ax3 = axes[2]
    ax3.plot(timestamps, z_cvd, 'r-', linewidth=2, label='CVD Z-score', alpha=0.8)
    ax3.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='High Threshold')
    ax3.axhline(y=-2.0, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=1.0, color='r', linestyle=':', alpha=0.5, label='Mid Threshold')
    ax3.axhline(y=-1.0, color='r', linestyle=':', alpha=0.5)
    ax3.set_title('CVD Z-score', fontsize=12)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('CVD Z-score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = 'divergence_demo_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存: {output_file}")
    
    # 显示图表
    plt.show()


def main():
    """主函数"""
    try:
        # 运行背离检测
        timestamps, prices, z_ofi, z_cvd, events = run_divergence_detection()
        
        # 创建可视化
        create_visualization(timestamps, prices, z_ofi, z_cvd, events)
        
        print("\n=== 演示完成 ===")
        print("背离检测模块功能验证成功！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
