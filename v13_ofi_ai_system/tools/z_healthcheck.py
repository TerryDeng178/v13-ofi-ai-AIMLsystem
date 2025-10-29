#!/usr/bin/env python3
"""
Z-Score健康检查脚本
读取最近60分钟的signals_*.jsonl，计算Z-score分布和信号质量指标
"""

import json
import os
import sys
import glob
import time
from pathlib import Path
from typing import List, Dict, Any, Optional


def percentile(values: List[float], p: float) -> float:
    """纯Python百分位函数（排序后线性插值）"""
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n == 1:
        return sorted_values[0]
    
    # 线性插值
    index = (p / 100.0) * (n - 1)
    lower = int(index)
    upper = min(lower + 1, n - 1)
    
    if lower == upper:
        return sorted_values[lower]
    
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def is_finite(value: Any) -> bool:
    """检查数值是否为有限数"""
    try:
        return float(value) == float(value) and abs(float(value)) != float('inf')
    except (ValueError, TypeError):
        return False


def read_signals_files(minutes: int = 60) -> List[Dict[str, Any]]:
    """读取最近N分钟的signals文件"""
    signals_dir = Path("runtime/ready/signal")
    if not signals_dir.exists():
        print("ERROR: runtime/ready/signal directory not found")
        sys.exit(1)
    
    # 查找所有signals文件
    pattern = str(signals_dir / "*" / "signals_*.jsonl")
    files = glob.glob(pattern)
    
    if not files:
        print("ERROR: No signals_*.jsonl files found")
        sys.exit(1)
    
    # 按修改时间排序，取最近的文件
    files.sort(key=os.path.getmtime, reverse=True)
    
    # 计算时间阈值（最近N分钟）
    cutoff_time = time.time() - (minutes * 60)
    recent_files = []
    
    for file_path in files:
        if os.path.getmtime(file_path) >= cutoff_time:
            recent_files.append(file_path)
        if len(recent_files) >= 120:  # 最多120个分片
            break
    
    if not recent_files:
        print("ERROR: No recent signals files found in the last {} minutes".format(minutes))
        sys.exit(1)
    
    # 读取所有信号数据
    signals = []
    for file_path in recent_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            signal = json.loads(line)
                            signals.append(signal)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"WARNING: Failed to read {file_path}: {e}")
            continue
    
    return signals


def calculate_z_health_metrics(signals: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算Z-score健康指标"""
    if not signals:
        print("ERROR: No signals data to analyze")
        sys.exit(1)
    
    # 提取Z-score数据
    z_ofi_values = []
    z_cvd_values = []
    scores = []
    confirms = []
    
    total_signals = len(signals)
    missing_ofi = 0
    missing_cvd = 0
    invalid_scores = 0
    
    for signal in signals:
        # Z-score数据
        z_ofi = signal.get('z_ofi')
        z_cvd = signal.get('z_cvd')
        score = signal.get('score')
        confirm = signal.get('confirm', False)
        
        # 统计缺失值
        if z_ofi is None:
            missing_ofi += 1
        elif is_finite(z_ofi):
            z_ofi_values.append(float(z_ofi))
        
        if z_cvd is None:
            missing_cvd += 1
        elif is_finite(z_cvd):
            z_cvd_values.append(float(z_cvd))
        
        # Score数据
        if score is not None and is_finite(score):
            scores.append(float(score))
        else:
            invalid_scores += 1
        
        # Confirm数据
        confirms.append(bool(confirm))
    
    # 计算指标
    metrics = {}
    
    # P(|z_ofi|>2)%
    if z_ofi_values:
        abs_z_ofi = [abs(z) for z in z_ofi_values]
        p_abs_gt2_ofi = sum(1 for z in abs_z_ofi if z > 2) / len(abs_z_ofi) * 100
        metrics['p_abs_gt2_ofi'] = p_abs_gt2_ofi
    else:
        metrics['p_abs_gt2_ofi'] = 0.0
    
    # P(|z_cvd|>2)%
    if z_cvd_values:
        abs_z_cvd = [abs(z) for z in z_cvd_values]
        p_abs_gt2_cvd = sum(1 for z in abs_z_cvd if z > 2) / len(abs_z_cvd) * 100
        metrics['p_abs_gt2_cvd'] = p_abs_gt2_cvd
    else:
        metrics['p_abs_gt2_cvd'] = 0.0
    
    # weak_ratio% (1.0≤|score|<1.8)
    if scores:
        weak_count = sum(1 for s in scores if 1.0 <= abs(s) < 1.8)
        weak_ratio = weak_count / len(scores) * 100
        metrics['weak_ratio'] = weak_ratio
    else:
        metrics['weak_ratio'] = 0.0
    
    # strong_ratio% (|score|≥1.8)
    if scores:
        strong_count = sum(1 for s in scores if abs(s) >= 1.8)
        strong_ratio = strong_count / len(scores) * 100
        metrics['strong_ratio'] = strong_ratio
    else:
        metrics['strong_ratio'] = 0.0
    
    # confirm_ratio%
    confirm_count = sum(confirms)
    confirm_ratio = confirm_count / len(confirms) * 100
    metrics['confirm_ratio'] = confirm_ratio
    
    # 缺失比例
    metrics['missing_ofi_pct'] = missing_ofi / total_signals * 100
    metrics['missing_cvd_pct'] = missing_cvd / total_signals * 100
    metrics['invalid_scores_pct'] = invalid_scores / total_signals * 100
    
    return metrics


def main():
    """主函数"""
    try:
        print("=== Z-Score Health Check ===")
        
        # 读取信号数据
        signals = read_signals_files(60)
        print(f"Loaded {len(signals)} signals from recent files")
        
        # 计算健康指标
        metrics = calculate_z_health_metrics(signals)
        
        # 输出结果
        print(f"P(|z_ofi|>2): {metrics['p_abs_gt2_ofi']:.2f}%")
        print(f"P(|z_cvd|>2): {metrics['p_abs_gt2_cvd']:.2f}%")
        print(f"Weak ratio (1.0≤|score|<1.8): {metrics['weak_ratio']:.2f}%")
        print(f"Strong ratio (|score|≥1.8): {metrics['strong_ratio']:.2f}%")
        print(f"Confirm ratio: {metrics['confirm_ratio']:.2f}%")
        print(f"Missing z_ofi: {metrics['missing_ofi_pct']:.2f}%")
        print(f"Missing z_cvd: {metrics['missing_cvd_pct']:.2f}%")
        print(f"Invalid scores: {metrics['invalid_scores_pct']:.2f}%")
        
        # 检查是否有NaN/Inf
        if metrics['invalid_scores_pct'] > 0:
            print("ERROR: Found invalid scores (NaN/Inf)")
            sys.exit(1)
        
        print("Z-health check completed successfully")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
