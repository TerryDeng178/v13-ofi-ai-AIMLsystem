"""
使用真实市场数据测试背离检测器

从 deploy/data/ofi_cvd/date=2025-10-27 加载6个交易对的真实数据
进行背离检测和性能评估

Author: Test Engineer
Created: 2025-01-20
"""

import sys
import os
import time
import math
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ofi_cvd_divergence import DivergenceDetector, DivergenceConfig

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[ERROR] pandas 未安装，请安装: pip install pandas")
    sys.exit(1)


def find_parquet_files(base_dir: Path) -> List[Path]:
    """查找所有 Parquet 数据文件"""
    data_files = []
    
    if not base_dir.exists():
        print(f"[WARN] 目录不存在: {base_dir}")
        return data_files
    
    try:
        for file_path in base_dir.rglob("*.parquet"):
            if file_path.is_file() and 'deadletter' not in str(file_path):
                data_files.append(file_path)
    except Exception as e:
        print(f"[WARN] 搜索文件时出错: {e}")
    
    return data_files


def load_data_file(file_path: Path):
    """加载数据文件"""
    try:
        df = pd.read_parquet(file_path)
        if df.empty:
            print(f"  [WARN] 文件为空: {file_path.name}")
            return None
        print(f"  [OK] 已加载 {len(df)} 行数据")
        return df
    except Exception as e:
        print(f"  [ERROR] 加载失败: {e}")
        return None


def detect_columns(df):
    """自动检测列名"""
    column_map = {}
    
    # OFI Z-score - 扩展更多可能的列名
    for col in ['ofi_z', 'z_ofi', 'ofi_zscore', 'ofi_z_score', 'ofi_zscore_1m', 'ofi_zscore_5m', 'ofi_zscore_15m']:
        if col in df.columns:
            column_map['ofi_z'] = col
            break
    
    # CVD Z-score - 扩展更多可能的列名
    for col in ['z_cvd', 'cvd_z', 'cvd_zscore', 'cvd_z_score', 'cvd_zscore_1m', 'cvd_zscore_5m', 'cvd_zscore_15m']:
        if col in df.columns:
            column_map['cvd_z'] = col
            break
    
    # Fusion score
    for col in ['fusion_score', 'fusion', 'z_fusion']:
        if col in df.columns:
            column_map['fusion'] = col
            break
    
    # Consistency
    for col in ['consistency', 'cons', 'fusion_consistency']:
        if col in df.columns:
            column_map['consistency'] = col
            break
    
    # 时间戳
    for col in ['ts_ms', 'second_ts', 'timestamp', 'ts']:
        if col in df.columns:
            column_map['ts'] = col
            break
    
    # 价格
    for col in ['mid', 'price', 'last_price']:
        if col in df.columns:
            column_map['price'] = col
            break
    
    return column_map


def test_with_real_data():
    """使用真实数据测试背离检测器"""
    print("=" * 80)
    print("使用真实市场数据测试背离检测器")
    print("=" * 80)
    
    # 数据目录
    data_dir = Path(__file__).parent.parent / "deploy" / "data" / "ofi_cvd" / "date=2025-10-27"
    
    print(f"\n数据目录: {data_dir}")
    
    # 查找数据文件
    data_files = find_parquet_files(data_dir)
    
    if not data_files:
        print("\n[WARN] 未找到数据文件，使用模拟数据")
        print("[INFO] 提示：请在 deploy/data/ofi_cvd/date=2025-10-27 目录下放置数据文件")
        return test_with_simulated_data()
    
    print(f"\n找到 {len(data_files)} 个数据文件")
    
    # 测试配置
    config = DivergenceConfig(
        swing_L=12,
        min_separation=6,
        warmup_min=100,
        cooldown_secs=1.0,
        max_lag=0.3,
        weak_threshold=35.0,
        use_fusion=True
    )
    
    detector = DivergenceDetector(config=config)
    
    # 统计信息
    total_samples = 0
    total_events = 0
    events_by_type = {}
    events_by_channel = {}
    
    # 处理每个文件
    valid_files_processed = 0
    for file_path in data_files[:6]:  # 限制为6个文件
        print(f"\n处理文件: {file_path.name}")

        df = load_data_file(file_path)
        if df is None:
            continue
        
        # 检测列名
        col_map = detect_columns(df)
        
        if 'ofi_z' not in col_map or 'cvd_z' not in col_map:
            print("  [SKIP] 缺少必需列 (ofi_z, cvd_z)，跳过")
            print(f"  [INFO] 可用列: {list(df.columns)}")
            print(f"  [INFO] 检测到的列映射: {col_map}")
            continue
        
        valid_files_processed += 1
        
        # 提取数据
        ts_col = col_map.get('ts', 'ts_ms')
        price_col = col_map.get('price', 'mid')
        ofi_col = col_map['ofi_z']
        cvd_col = col_map['cvd_z']
        
        fusion_col = col_map.get('fusion')
        consistency_col = col_map.get('consistency')
        
        # 转换时间戳为秒
        if ts_col in df.columns:
            if 'ms' in ts_col.lower():
                df['ts_sec'] = df[ts_col] / 1000.0
            else:
                df['ts_sec'] = df[ts_col]
        else:
            df['ts_sec'] = range(len(df))
        
        # 处理数据
        warmup_count = 0
        for idx, row in df.iterrows():
            ts = float(row['ts_sec']) if not pd.isna(row['ts_sec']) else idx * 0.1
            price = float(row[price_col]) if not pd.isna(row[price_col]) else 100.0
            z_ofi = float(row[ofi_col]) if not pd.isna(row[ofi_col]) else 0.0
            z_cvd = float(row[cvd_col]) if not pd.isna(row[cvd_col]) else 0.0
            
            fusion_score = None
            consistency = None
            
            if fusion_col and fusion_col in df.columns:
                fusion_score = row[fusion_col] if not pd.isna(row[fusion_col]) else None
            
            if consistency_col and consistency_col in df.columns:
                consistency = row[consistency_col] if not pd.isna(row[consistency_col]) else None
            
            warmup = warmup_count < config.warmup_min
            warmup_count += 1
            
            event = detector.update(
                ts=ts,
                price=price,
                z_ofi=z_ofi,
                z_cvd=z_cvd,
                fusion_score=fusion_score,
                consistency=consistency,
                warmup=warmup,
                lag_sec=0.0
            )
            
            if event:
                total_events += 1
                event_type = event.get('type', 'unknown')
                events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
                
                channel = event.get('channel', 'unknown')
                events_by_channel[channel] = events_by_channel.get(channel, 0) + 1
            
                total_samples += 1

    # 如果没有处理任何有效文件，使用模拟数据
    if valid_files_processed == 0:
        print("\n[WARN] 没有找到有效的真实数据文件，使用模拟数据")
        return test_with_simulated_data()

    # 打印统计
    print("\n" + "=" * 80)
    print("测试结果统计")
    print("=" * 80)
    print(f"总样本数: {total_samples:,}")
    print(f"总事件数: {total_events}")
    print(f"\n按类型分布:")
    for event_type, count in sorted(events_by_type.items()):
        print(f"  {event_type}: {count}")
    print(f"\n按通道分布:")
    for channel, count in sorted(events_by_channel.items()):
        print(f"  {channel}: {count}")
    
    # 获取检测器统计
    stats = detector.get_stats()
    print(f"\n检测器统计:")
    print(f"  总枢轴数: {stats['pivots_detected']}")
    print(f"  各通道枢轴: {stats['pivots_by_channel']}")
    print(f"  抑制事件数: {stats['suppressed_total']}")
    print(f"  软抑制数: {stats.get('soft_suppressed_total', 0)}")
    
    print("=" * 80)
    
    # 使用断言而不是返回布尔值
    assert total_events > 0, f"应该检测到背离事件，但实际检测到 {total_events} 个"


def test_with_simulated_data():
    """使用模拟数据测试（当真实数据不可用时）"""
    print("\n使用模拟数据替代测试")
    
    detector = DivergenceDetector()
    
    # 生成模拟数据
    total_events = 0
    for i in range(1000):
        ts = 1.0 + i * 0.1
        price = 100.0 + math.sin(i * 0.01) * 10
        z_ofi = math.cos(i * 0.02) * 3
        z_cvd = math.sin(i * 0.03) * 2
        
        event = detector.update(
            ts=ts,
            price=price,
            z_ofi=z_ofi,
            z_cvd=z_cvd,
            fusion_score=None,
            consistency=None,
            warmup=i < 100,
            lag_sec=0.0
        )
        
        if event:
            total_events += 1
    
    print(f"模拟测试完成，检测到 {total_events} 个事件")
    
    # 使用断言而不是返回布尔值
    assert total_events > 0, f"模拟测试应该检测到事件，但实际检测到 {total_events} 个"


if __name__ == '__main__':
    try:
        success = test_with_real_data()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

