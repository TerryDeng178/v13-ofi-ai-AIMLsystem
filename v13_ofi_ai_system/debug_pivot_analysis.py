"""
调试枢轴分析
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


def debug_pivot_analysis():
    """调试枢轴分析"""
    print("=== 调试枢轴分析 ===\n")
    
    # 创建配置
    config = DivergenceConfig(
        swing_L=6,
        min_separation=3,
        cooldown_secs=1.0,
        warmup_min=10,
        z_hi=1.8,
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
    
    # 添加数据点
    for i in range(n_samples):
        ts = timestamps[i]
        price = prices[i]
        ofi = z_ofi[i]
        cvd = z_cvd[i]
        
        detector.update(ts, price, ofi, cvd, lag_sec=0.1)
    
    # 分析枢轴
    ofi_pivots = detector.price_ofi_detector.get_all_pivots()
    cvd_pivots = detector.price_cvd_detector.get_all_pivots()
    
    print(f"OFI枢轴数: {len(ofi_pivots)}")
    print(f"CVD枢轴数: {len(cvd_pivots)}")
    print()
    
    print("OFI枢轴详情:")
    for i, pivot in enumerate(ofi_pivots):
        print(f"  枢轴 {i+1}: 索引={pivot['index']}, 时间={pivot['ts']}, 价格={pivot['price']:.2f}")
        print(f"    价格高点: {pivot['is_price_high']}, 价格低点: {pivot['is_price_low']}")
        print(f"    指标高点: {pivot['is_indicator_high']}, 指标低点: {pivot['is_indicator_low']}")
        print(f"    指标值: {pivot['indicator']:.2f}")
        print()
    
    print("CVD枢轴详情:")
    for i, pivot in enumerate(cvd_pivots):
        print(f"  枢轴 {i+1}: 索引={pivot['index']}, 时间={pivot['ts']}, 价格={pivot['price']:.2f}")
        print(f"    价格高点: {pivot['is_price_high']}, 价格低点: {pivot['is_price_low']}")
        print(f"    指标高点: {pivot['is_indicator_high']}, 指标低点: {pivot['is_indicator_low']}")
        print(f"    指标值: {pivot['indicator']:.2f}")
        print()
    
    # 测试同型枢轴配对
    print("测试同型枢轴配对:")
    
    # OFI低点
    ofi_lows = [p for p in ofi_pivots if p.get('is_price_low')]
    print(f"OFI价格低点数: {len(ofi_lows)}")
    if len(ofi_lows) >= 2:
        print(f"  最近两个低点: 索引{ofi_lows[-2]['index']} 和 {ofi_lows[-1]['index']}")
    else:
        print("  不足2个低点")
    
    # OFI高点
    ofi_highs = [p for p in ofi_pivots if p.get('is_price_high')]
    print(f"OFI价格高点数: {len(ofi_highs)}")
    if len(ofi_highs) >= 2:
        print(f"  最近两个高点: 索引{ofi_highs[-2]['index']} 和 {ofi_highs[-1]['index']}")
    else:
        print("  不足2个高点")
    
    # CVD低点
    cvd_lows = [p for p in cvd_pivots if p.get('is_price_low')]
    print(f"CVD价格低点数: {len(cvd_lows)}")
    if len(cvd_lows) >= 2:
        print(f"  最近两个低点: 索引{cvd_lows[-2]['index']} 和 {cvd_lows[-1]['index']}")
    else:
        print("  不足2个低点")
    
    # CVD高点
    cvd_highs = [p for p in cvd_pivots if p.get('is_price_high')]
    print(f"CVD价格高点数: {len(cvd_highs)}")
    if len(cvd_highs) >= 2:
        print(f"  最近两个高点: 索引{cvd_highs[-2]['index']} 和 {cvd_highs[-1]['index']}")
    else:
        print("  不足2个高点")
    
    print(f"\n调试统计: {detector._stats}")


if __name__ == '__main__':
    debug_pivot_analysis()
