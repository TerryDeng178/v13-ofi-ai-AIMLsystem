#!/usr/bin/env python3
"""
存储健康巡检脚本
检查文件分片、心跳和存储状态
"""

import os
import sys
import glob
import time
import json
from pathlib import Path
from typing import List, Dict, Any


def check_signals_files(minutes: int = 10) -> Dict[str, Any]:
    """检查signals文件分片状态"""
    signals_dir = Path("runtime/ready/signal")
    spool_dir = Path("runtime/spool")
    
    result = {
        'ready_files': 0,
        'spool_files': 0,
        'minutes_covered': 0,
        'ready_rotation_ok': True,
        'spool_stagnant': False,
        'stagnant_files': []
    }
    
    if not signals_dir.exists():
        result['ready_rotation_ok'] = False
        return result
    
    # 检查ready目录中的文件
    ready_pattern = str(signals_dir / "*" / "signals_*.jsonl")
    ready_files = glob.glob(ready_pattern)
    result['ready_files'] = len(ready_files)
    
    # 检查最近10分钟的文件
    cutoff_time = time.time() - (minutes * 60)
    recent_files = []
    
    for file_path in ready_files:
        if os.path.getmtime(file_path) >= cutoff_time:
            recent_files.append(file_path)
    
    # 按分钟分组统计
    minute_files = {}
    for file_path in recent_files:
        filename = os.path.basename(file_path)
        # 提取时间戳：signals_YYYYMMDD_HHMM.jsonl
        try:
            time_part = filename.split('_')[1].split('.')[0]  # YYYYMMDD_HHMM
            minute_files[time_part] = minute_files.get(time_part, 0) + 1
        except (IndexError, ValueError):
            continue
    
    result['minutes_covered'] = len(minute_files)
    
    # 检查是否每分钟都有文件
    if len(minute_files) < minutes * 0.8:  # 至少80%的分钟有文件
        result['ready_rotation_ok'] = False
    
    # 检查spool目录中的滞留文件
    if spool_dir.exists():
        spool_pattern = str(spool_dir / "*" / "*.part")
        spool_files = glob.glob(spool_pattern)
        result['spool_files'] = len(spool_files)
        
        # 检查滞留超过90秒的文件
        stagnant_threshold = time.time() - 90
        for file_path in spool_files:
            if os.path.getmtime(file_path) < stagnant_threshold:
                result['spool_stagnant'] = True
                result['stagnant_files'].append(file_path)
    
    return result


def check_gate_stats_heartbeat(max_age_seconds: int = 60) -> Dict[str, Any]:
    """检查gate_stats.jsonl心跳"""
    gate_stats_file = Path("artifacts/gate_stats.jsonl")
    
    result = {
        'heartbeat_ok': False,
        'last_entry_age': None,
        'total_entries': 0
    }
    
    if not gate_stats_file.exists():
        return result
    
    try:
        # 读取最后几行
        with open(gate_stats_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        result['total_entries'] = len(lines)
        
        if lines:
            # 解析最后一行
            last_line = lines[-1].strip()
            try:
                last_entry = json.loads(last_line)
                timestamp_str = last_entry.get('timestamp')
                
                if timestamp_str:
                    # 解析时间戳
                    from datetime import datetime
                    last_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    current_time = datetime.now()
                    
                    # 计算时间差
                    time_diff = (current_time - last_time).total_seconds()
                    result['last_entry_age'] = time_diff
                    
                    # 检查是否在阈值内
                    if time_diff <= max_age_seconds:
                        result['heartbeat_ok'] = True
                        
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
                
    except Exception as e:
        print(f"WARNING: Failed to read gate_stats.jsonl: {e}")
    
    return result


def main():
    """主函数"""
    try:
        print("=== Storage Liveness Check ===")
        
        # 检查signals文件
        signals_status = check_signals_files(10)
        print(f"Ready signals files: {signals_status['ready_files']}")
        print(f"Spool files: {signals_status['spool_files']}")
        print(f"Minutes covered (last 10min): {signals_status['minutes_covered']}")
        print(f"Ready rotation OK: {signals_status['ready_rotation_ok']}")
        
        if signals_status['spool_stagnant']:
            print(f"ALERT: Found {len(signals_status['stagnant_files'])} stagnant spool files:")
            for file_path in signals_status['stagnant_files']:
                print(f"  - {file_path}")
        
        # 检查gate_stats心跳
        heartbeat_status = check_gate_stats_heartbeat(60)
        print(f"Gate stats entries: {heartbeat_status['total_entries']}")
        print(f"Gate stats heartbeat OK: {heartbeat_status['heartbeat_ok']}")
        
        if heartbeat_status['last_entry_age'] is not None:
            print(f"Last entry age: {heartbeat_status['last_entry_age']:.1f}s")
        
        # 综合状态
        overall_status = "OK"
        if not signals_status['ready_rotation_ok']:
            overall_status = "ALERT"
        if signals_status['spool_stagnant']:
            overall_status = "ALERT"
        if not heartbeat_status['heartbeat_ok']:
            overall_status = "ALERT"
        
        print(f"Overall status: {overall_status}")
        
        if overall_status == "ALERT":
            print("Storage liveness check completed with alerts")
        else:
            print("Storage liveness check completed successfully")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
