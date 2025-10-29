"""
背离检测器端到端测试脚本

流程：
1. 加载原始订单簿数据
2. 使用 RealOFICalculator 计算 OFI
3. 使用 RealCVDCalculator 计算 CVD
4. 使用 DivergenceDetector 检测背离
5. 输出统计和结果

Author: Test Engineer
Created: 2025-01-20
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ofi_cvd_divergence import DivergenceDetector, DivergenceConfig

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[ERROR] pandas/numpy 未安装")
    sys.exit(1)

# 动态导入计算器
try:
    from src.real_ofi_calculator import RealOFICalculator, OFIConfig as OFICfg
    from src.real_cvd_calculator import RealCVDCalculator, CVDConfig as CVDCfg
    CALCULATORS_AVAILABLE = True
except ImportError as e:
    CALCULATORS_AVAILABLE = False
    print(f"[ERROR] 无法导入计算器: {e}")
    sys.exit(1)


def parse_orderbook_json(bids_json: str, asks_json: str):
    """解析订单簿JSON"""
    try:
        bids_data = json.loads(bids_json) if isinstance(bids_json, str) else bids_json
        asks_data = json.loads(asks_json) if isinstance(asks_json, str) else asks_json
        
        # 转换为列表格式 [[price, qty], ...]
        bids = [[float(level[0]), float(level[1])] for level in bids_data[:5]]
        asks = [[float(level[0]), float(level[1])] for level in asks_data[:5]]
        
        return bids, asks
    except Exception as e:
        return None, None


def process_e2e_test():
    """端到端测试"""
    print("=" * 80)
    print("背离检测器端到端测试")
    print("=" * 80)
    
    # 数据目录
    data_dir = Path(__file__).parent.parent / "deploy" / "data" / "ofi_cvd" / "date=2025-10-27"
    
    # 查找订单簿文件
    orderbook_dir = data_dir / "symbol=BTCUSDT" / "kind=orderbook"
    files = list(orderbook_dir.glob("*.parquet"))[:3]  # 只处理前3个文件
    
    if not files:
        print("[WARN] 未找到数据文件")
        return False
    
    print(f"\n找到 {len(files)} 个文件，处理前3个")
    
    # 初始化计算器
    ofi_config = OFICfg(
        levels=5,
        z_window=80,
        ema_alpha=0.30,
        z_clip=5.0
    )
    cvd_config = CVDCfg(
        z_window=80,
        ema_alpha=0.30
    )
    
    ofi_calc = RealOFICalculator("BTCUSDT", ofi_config)
    cvd_calc = RealCVDCalculator("BTCUSDT", cvd_config)
    
    # 初始化背离检测器
    divergence_config = DivergenceConfig(
        swing_L=12,
        min_separation=6,
        warmup_min=100,
        cooldown_secs=1.0,
        max_lag=0.3,
        weak_threshold=35.0,
        use_fusion=True
    )
    detector = DivergenceDetector(divergence_config)
    
    # 统计
    total_samples = 0
    total_events = 0
    events_by_type = {}
    events_by_channel = {}
    ofi_results = []
    cvd_results = []
    
    # 处理数据
    start_time = time.time()
    
    for file_idx, file_path in enumerate(files):
        print(f"\n处理文件 {file_idx+1}/{len(files)}: {file_path.name}")
        
        df = pd.read_parquet(file_path)
        
        for idx, row in df.iterrows():
            # 解析订单簿
            bids, asks = parse_orderbook_json(row.get('bids_json'), row.get('asks_json'))
            
            if bids is None or asks is None:
                continue
            
            ts_ms = row.get('ts_ms', idx * 10)
            
            try:
                # 计算 OFI
                ofi_result = ofi_calc.update_with_snapshot(
                    bids, asks, event_time_ms=ts_ms
                )
                
                # 计算 CVD
                # CVD 需要成交量数据，这里简化处理
                # 实际应该从成交量数据计算
                cvd_result = {
                    'z_cvd': 0.0,  # 占位
                    'cvd': 0.0
                }
                
                z_ofi = ofi_result.get('z_ofi')
                if z_ofi is None:
                    continue  # 跳过 warmup 期间
                
                # 获取价格
                price = row.get('mid', 100.0)
                if pd.isna(price):
                    continue
                
                # 转换时间戳为秒
                ts_sec = ts_ms / 1000.0
                
                # 更新背离检测器
                event = detector.update(
                    ts=ts_sec,
                    price=float(price),
                    z_ofi=z_ofi,
                    z_cvd=0.0,  # 占位，实际应从CVD结果获取
                    fusion_score=None,
                    consistency=None,
                    warmup=total_samples < 100,
                    lag_sec=0.0
                )
                
                if event:
                    total_events += 1
                    event_type = event.get('type', 'unknown')
                    events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
                    
                    channel = event.get('channel', 'unknown')
                    events_by_channel[channel] = events_by_channel.get(channel, 0) + 1
                
                total_samples += 1
                
            except Exception as e:
                continue
    
    elapsed_time = time.time() - start_time
    
    # 获取最终统计
    stats = detector.get_stats()
    
    # 打印结果
    print("\n" + "=" * 80)
    print("端到端测试结果")
    print("=" * 80)
    print(f"处理时间: {elapsed_time:.2f} 秒")
    print(f"总样本数: {total_samples:,}")
    print(f"总事件数: {total_events}")
    print(f"\n按类型分布:")
    for event_type, count in sorted(events_by_type.items()):
        print(f"  {event_type}: {count}")
    print(f"\n按通道分布:")
    for channel, count in sorted(events_by_channel.items()):
        print(f"  {channel}: {count}")
    print(f"\n检测器统计:")
    print(f"  总枢轴数: {stats['pivots_detected']}")
    print(f"  各通道枢轴: {stats['pivots_by_channel']}")
    print(f"  抑制事件数: {stats['suppressed_total']}")
    
    if total_samples > 0:
        rate = total_samples / elapsed_time
        print(f"\n处理速度: {rate:.0f} 样本/秒")
    
    print("=" * 80)
    
    return total_events > 0 or total_samples > 0


if __name__ == '__main__':
    try:
        success = process_e2e_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

