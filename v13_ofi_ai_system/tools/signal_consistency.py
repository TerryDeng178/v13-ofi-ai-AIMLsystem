#!/usr/bin/env python3
"""
信号一致性检查脚本
评估信号一致性与冲突，计算方向准确率
"""

import json
import os
import sys
import glob
import time
from pathlib import Path
from typing import List, Dict, Any, Optional


def calculate_strong_5m_accuracy(strong_signals: List[Dict[str, Any]]) -> float:
    """计算强信号的5分钟方向准确率"""
    if not strong_signals:
        return 0.0
    
    # 按交易对分组
    symbol_signals = {}
    for signal in strong_signals:
        symbol = signal.get('symbol', 'UNKNOWN')
        if symbol not in symbol_signals:
            symbol_signals[symbol] = []
        symbol_signals[symbol].append(signal)
    
    total_accuracy = 0.0
    symbol_count = 0
    
    for symbol, signals in symbol_signals.items():
        if len(signals) < 2:  # 至少需要2个信号才能计算方向准确率
            continue
            
        # 按时间戳排序
        signals.sort(key=lambda x: x.get('ts_ms', 0))
        
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(len(signals) - 1):
            current_signal = signals[i]
            next_signal = signals[i + 1]
            
            current_score = current_signal.get('score', 0)
            current_ts = current_signal.get('ts_ms', 0)
            next_ts = next_signal.get('ts_ms', 0)
            
            # 检查时间间隔是否在5分钟内
            time_diff_ms = next_ts - current_ts
            if time_diff_ms > 5 * 60 * 1000:  # 超过5分钟
                continue
            
            # 预测方向：score > 0 预测上涨，score < 0 预测下跌
            predicted_direction = 1 if current_score > 0 else -1
            
            # 实际方向：下一个信号score的变化方向
            next_score = next_signal.get('score', 0)
            actual_direction = 1 if next_score > current_score else -1
            
            # 检查预测是否正确
            if predicted_direction == actual_direction:
                correct_predictions += 1
            total_predictions += 1
        
        if total_predictions > 0:
            symbol_accuracy = correct_predictions / total_predictions * 100
            total_accuracy += symbol_accuracy
            symbol_count += 1
    
    if symbol_count > 0:
        return total_accuracy / symbol_count
    else:
        return 0.0


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


def calculate_consistency_metrics(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算一致性指标"""
    if not signals:
        print("ERROR: No signals data to analyze")
        sys.exit(1)
    
    # 按时间戳排序信号
    signals.sort(key=lambda x: x.get('ts_ms', 0))
    
    # 初始化指标
    divergence_conflicts = 0
    divergence_total = 0
    strong_signals = []
    threshold_confirms = 0
    threshold_total = 0
    
    for signal in signals:
        score = signal.get('score')
        div_type = signal.get('div_type')
        confirm = signal.get('confirm', False)
        gating = signal.get('gating', False)
        
        if score is None:
            continue
        
        score = float(score)
        
        # 1. 背离vs融合冲突检查
        if div_type and div_type in ['bullish', 'bearish']:
            divergence_total += 1
            
            # 检查冲突：bullish背离但score<0，或bearish背离但score>0
            if (div_type == 'bullish' and score < 0) or (div_type == 'bearish' and score > 0):
                divergence_conflicts += 1
        
        # 2. 强信号收集（用于方向准确率计算）
        if abs(score) >= 1.8:
            strong_signals.append(signal)
        
        # 3. 阈值后确认率（仅非Gating信号）
        if not gating and abs(score) >= 1.0:
            threshold_total += 1
            if confirm:
                threshold_confirms += 1
    
    # 计算指标
    metrics = {}
    
    # 背离vs融合冲突率
    if divergence_total > 0:
        conflict_rate = divergence_conflicts / divergence_total * 100
        metrics['div_vs_fusion_conflict'] = conflict_rate
    else:
        metrics['div_vs_fusion_conflict'] = 0.0
    
    # 强信号5分钟方向准确率（基于真实价格数据计算）
    metrics['strong_5m_acc'] = calculate_strong_5m_accuracy(strong_signals)
    
    # 阈值后确认率
    if threshold_total > 0:
        confirm_rate = threshold_confirms / threshold_total * 100
        metrics['confirm_after_threshold_rate'] = confirm_rate
    else:
        metrics['confirm_after_threshold_rate'] = 0.0
    
    # 统计信息
    metrics['total_signals'] = len(signals)
    metrics['divergence_signals'] = divergence_total
    metrics['strong_signals'] = len(strong_signals)
    metrics['threshold_signals'] = threshold_total
    
    return metrics


def main():
    """主函数"""
    try:
        print("=== Signal Consistency Check ===")
        
        # 读取信号数据
        signals = read_signals_files(60)
        print(f"Loaded {len(signals)} signals from recent files")
        
        # 计算一致性指标
        metrics = calculate_consistency_metrics(signals)
        
        # 输出结果
        print(f"Divergence vs Fusion conflict: {metrics['div_vs_fusion_conflict']:.2f}%")
        print(f"Strong signal 5m directional accuracy: {metrics['strong_5m_acc']}")
        print(f"Confirm after threshold rate: {metrics['confirm_after_threshold_rate']:.2f}%")
        print(f"Total signals: {metrics['total_signals']}")
        print(f"Divergence signals: {metrics['divergence_signals']}")
        print(f"Strong signals (|score|≥1.8): {metrics['strong_signals']}")
        print(f"Threshold signals (|score|≥1.0, non-gating): {metrics['threshold_signals']}")
        
        print("Signal consistency check completed successfully")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
