#!/usr/bin/env python3
"""
简化的Strong 5m accuracy测试脚本
直接生成测试数据并计算方向准确率
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

def generate_test_signals() -> List[Dict[str, Any]]:
    """生成测试信号数据"""
    signals = []
    base_time = int(time.time() * 1000) - (60 * 60 * 1000)  # 1小时前
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']
    
    for symbol in symbols:
        # 为每个交易对生成100个信号
        for i in range(100):
            ts_ms = base_time + i * 60000  # 每分钟一个信号
            
            # 模拟强信号（|score| >= 1.8）
            if i % 10 == 0:  # 每10个信号中1个是强信号
                score = 2.0 if i % 20 == 0 else -2.0
            else:
                score = (i % 7 - 3) * 0.5  # 其他信号较弱
            
            signal = {
                "timestamp": datetime.fromtimestamp(ts_ms / 1000).isoformat(),
                "ts_ms": ts_ms,
                "symbol": symbol,
                "score": score,
                "z_ofi": score * 0.8,
                "z_cvd": score * 0.6,
                "regime": "normal",
                "div_type": "bullish" if score > 0 else "bearish" if score < 0 else None,
                "confirm": abs(score) >= 1.0,
                "gating": False,
                "guard_reason": None
            }
            signals.append(signal)
    
    return signals

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
            print(f"{symbol}: {correct_predictions}/{total_predictions} = {symbol_accuracy:.1f}%")
    
    if symbol_count > 0:
        return total_accuracy / symbol_count
    else:
        return 0.0

def save_test_signals(signals: List[Dict[str, Any]]):
    """保存测试信号到JSONL文件"""
    output_dir = Path("runtime/ready/signal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 按交易对分组保存
    symbol_signals = {}
    for signal in signals:
        symbol = signal.get('symbol', 'UNKNOWN')
        if symbol not in symbol_signals:
            symbol_signals[symbol] = []
        symbol_signals[symbol].append(signal)
    
    for symbol, symbol_sigs in symbol_signals.items():
        symbol_dir = output_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        # 按分钟分组
        minute_groups = {}
        for signal in symbol_sigs:
            ts_ms = signal.get('ts_ms', 0)
            minute_key = datetime.fromtimestamp(ts_ms / 1000).strftime("%Y%m%d_%H%M")
            if minute_key not in minute_groups:
                minute_groups[minute_key] = []
            minute_groups[minute_key].append(signal)
        
        # 保存每个分钟的文件
        for minute_key, minute_signals in minute_groups.items():
            file_path = symbol_dir / f"signals_{minute_key}.jsonl"
            with open(file_path, 'w', encoding='utf-8') as f:
                for signal in minute_signals:
                    f.write(json.dumps(signal, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(signals)} signals to {output_dir}")

def main():
    """主函数"""
    print("=== Strong 5m Accuracy Test ===")
    
    # 生成测试信号
    signals = generate_test_signals()
    print(f"Generated {len(signals)} test signals")
    
    # 保存信号到文件
    save_test_signals(signals)
    
    # 筛选强信号
    strong_signals = [s for s in signals if abs(s.get('score', 0)) >= 1.8]
    print(f"Found {len(strong_signals)} strong signals (|score| >= 1.8)")
    
    # 计算Strong 5m accuracy
    accuracy = calculate_strong_5m_accuracy(strong_signals)
    print(f"\nStrong 5m directional accuracy: {accuracy:.2f}%")
    
    # 阈值评估
    threshold = 52.0
    status = "PASS" if accuracy >= threshold else "FAIL"
    print(f"Threshold (≥{threshold}%): {status}")
    
    print("\nStrong 5m accuracy test completed successfully")

if __name__ == "__main__":
    main()
