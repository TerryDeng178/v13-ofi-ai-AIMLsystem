#!/usr/bin/env python3
"""
滞后与队列健康检查脚本
计算事件滞后和队列状态
"""

import json
import os
import sys
import glob
import time
import re
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


def calculate_event_lag(signals: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算事件滞后"""
    if not signals:
        return {'lag_p50_ms': 0.0, 'lag_p95_ms': 0.0}
    
    current_time_ms = int(time.time() * 1000)
    lag_values = []
    
    for signal in signals:
        ts_ms = signal.get('ts_ms')
        if ts_ms is not None:
            try:
                ts_ms = int(ts_ms)
                lag_ms = max(0, current_time_ms - ts_ms)
                lag_values.append(lag_ms)
            except (ValueError, TypeError):
                continue
    
    if not lag_values:
        return {'lag_p50_ms': 0.0, 'lag_p95_ms': 0.0}
    
    return {
        'lag_p50_ms': percentile(lag_values, 50),
        'lag_p95_ms': percentile(lag_values, 95)
    }


def extract_queue_metrics() -> Dict[str, Any]:
    """从控制台日志或gate_stats中提取队列指标"""
    metrics = {
        'qsize': None,
        'open_files': None,
        'dropped': None
    }
    
    # 尝试从gate_stats.jsonl中提取
    gate_stats_file = Path("artifacts/gate_stats.jsonl")
    if gate_stats_file.exists():
        try:
            with open(gate_stats_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 查找包含JsonlSink信息的行
            for line in reversed(lines[-10:]):  # 检查最后10行
                if '[JsonlSink]' in line:
                    # 解析格式：[JsonlSink] qsize=x open=y dropped=z
                    match = re.search(r'qsize=(\d+)\s+open=(\d+)\s+dropped=(\d+)', line)
                    if match:
                        metrics['qsize'] = int(match.group(1))
                        metrics['open_files'] = int(match.group(2))
                        metrics['dropped'] = int(match.group(3))
                        break
        except Exception as e:
            print(f"WARNING: Failed to parse gate_stats.jsonl: {e}")
    
    # 如果没找到，尝试从控制台日志中提取（这里简化处理）
    # 在实际环境中，可能需要从日志文件中提取
    
    return metrics


def main():
    """主函数"""
    try:
        print("=== Latency and Queue Health Check ===")
        
        # 读取信号数据
        signals = read_signals_files(60)
        print(f"Loaded {len(signals)} signals from recent files")
        
        # 计算事件滞后
        lag_metrics = calculate_event_lag(signals)
        print(f"Event lag P50: {lag_metrics['lag_p50_ms']:.1f}ms")
        print(f"Event lag P95: {lag_metrics['lag_p95_ms']:.1f}ms")
        
        # 提取队列指标
        queue_metrics = extract_queue_metrics()
        print(f"JsonlSink qsize: {queue_metrics['qsize']}")
        print(f"JsonlSink open files: {queue_metrics['open_files']}")
        print(f"JsonlSink dropped: {queue_metrics['dropped']}")
        
        # 检查阈值
        lag_p95_ok = lag_metrics['lag_p95_ms'] <= 120
        dropped_ok = queue_metrics['dropped'] is None or queue_metrics['dropped'] == 0
        
        print(f"Lag P95 OK (≤120ms): {lag_p95_ok}")
        print(f"Dropped OK (==0): {dropped_ok}")
        
        if lag_p95_ok and dropped_ok:
            print("Latency and queue health check completed successfully")
        else:
            print("Latency and queue health check completed with issues")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
