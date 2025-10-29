#!/usr/bin/env python3
"""
影子交易Go/No-Go判定脚本
综合所有检查结果，按硬性阈值给出最终判定
"""

import subprocess
import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List


def run_script(script_name: str) -> Dict[str, Any]:
    """运行单个检查脚本并解析结果"""
    script_path = Path("tools") / script_name
    
    if not script_path.exists():
        return {'error': f'Script {script_name} not found'}
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return {'error': f'Script {script_name} failed: {result.stderr}'}
        
        return {'stdout': result.stdout, 'stderr': result.stderr}
    
    except subprocess.TimeoutExpired:
        return {'error': f'Script {script_name} timed out'}
    except Exception as e:
        return {'error': f'Failed to run {script_name}: {e}'}


def parse_z_health_output(output: str) -> Dict[str, float]:
    """解析Z-health检查输出"""
    metrics = {}
    
    # 解析各种指标
    patterns = {
        'p_abs_gt2_ofi': r'P\(\|z_ofi\|>2\): ([\d.]+)%',
        'p_abs_gt2_cvd': r'P\(\|z_cvd\|>2\): ([\d.]+)%',
        'weak_ratio': r'Weak ratio \(1\.0≤\|score\|<1\.8\): ([\d.]+)%',
        'strong_ratio': r'Strong ratio \(\|score\|≥1\.8\): ([\d.]+)%',
        'confirm_ratio': r'Confirm ratio: ([\d.]+)%'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
        else:
            metrics[key] = 0.0
    
    return metrics


def parse_consistency_output(output: str) -> Dict[str, Any]:
    """解析一致性检查输出"""
    metrics = {}
    
    # 解析各种指标
    patterns = {
        'div_vs_fusion_conflict': r'Divergence vs Fusion conflict: ([\d.]+)%',
        'confirm_after_threshold_rate': r'Confirm after threshold rate: ([\d.]+)%'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
        else:
            metrics[key] = 0.0
    
    # 强信号方向准确率
    if 'Strong signal 5m directional accuracy: N/A' in output:
        metrics['strong_5m_acc'] = 'N/A'
    else:
        match = re.search(r'Strong signal 5m directional accuracy: ([\d.]+)%', output)
        if match:
            metrics['strong_5m_acc'] = float(match.group(1))
        else:
            metrics['strong_5m_acc'] = 'N/A'
    
    return metrics


def parse_storage_output(output: str) -> Dict[str, Any]:
    """解析存储健康检查输出"""
    metrics = {}
    
    # 解析各种指标
    patterns = {
        'minutes_covered': r'Minutes covered \(last 10min\): (\d+)',
        'ready_rotation_ok': r'Ready rotation OK: (True|False)',
        'heartbeat_ok': r'Gate stats heartbeat OK: (True|False)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            if key.endswith('_ok'):
                metrics[key] = match.group(1) == 'True'
            else:
                metrics[key] = int(match.group(1))
        else:
            if key.endswith('_ok'):
                metrics[key] = False
            else:
                metrics[key] = 0
    
    return metrics


def parse_latency_output(output: str) -> Dict[str, float]:
    """解析滞后检查输出"""
    metrics = {}
    
    patterns = {
        'lag_p50_ms': r'Event lag P50: ([\d.]+)ms',
        'lag_p95_ms': r'Event lag P95: ([\d.]+)ms'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
        else:
            metrics[key] = 0.0
    
    return metrics


def check_blocking_conditions() -> List[str]:
    """检查阻断条件"""
    blocking_reasons = []
    
    # 检查signals文件是否存在
    signals_dir = Path("runtime/ready/signal")
    if not signals_dir.exists():
        blocking_reasons.append("runtime/ready/signal directory not found")
    else:
        pattern = str(signals_dir / "*" / "signals_*.jsonl")
        import glob
        files = glob.glob(pattern)
        if not files:
            blocking_reasons.append("No signals_*.jsonl files found")
    
    return blocking_reasons


def evaluate_thresholds(metrics: Dict[str, Any]) -> Dict[str, bool]:
    """评估硬性阈值"""
    results = {}
    
    # 数据质量阈值
    z_health = metrics.get('z_health', {})
    results['p_abs_gt2_ofi_ok'] = 3.0 <= z_health.get('p_abs_gt2_ofi', 0) <= 12.0
    results['p_abs_gt2_cvd_ok'] = 3.0 <= z_health.get('p_abs_gt2_cvd', 0) <= 12.0
    results['strong_ratio_ok'] = 0.8 <= z_health.get('strong_ratio', 0) <= 3.5
    results['confirm_ratio_ok'] = z_health.get('confirm_ratio', 0) > 0
    
    # 一致性阈值
    consistency = metrics.get('consistency', {})
    results['div_vs_fusion_conflict_ok'] = consistency.get('div_vs_fusion_conflict', 0) < 2.0
    
    # 性能/稳定性阈值
    latency = metrics.get('latency', {})
    results['lag_p95_ok'] = latency.get('lag_p95_ms', 0) <= 120
    
    # 存储/可观测性阈值
    storage = metrics.get('storage', {})
    results['ready_rotation_ok'] = storage.get('ready_rotation_ok', False)
    results['gate_stats_heartbeat_ok'] = storage.get('heartbeat_ok', False)
    
    return results


def main():
    """主函数"""
    print("=== Shadow Trading Go/No-Go Decision ===")
    
    # 检查阻断条件
    blocking_reasons = check_blocking_conditions()
    if blocking_reasons:
        print("BLOCKED: " + "; ".join(blocking_reasons))
        print("Diagnostic suggestion: Ensure V13_SINK=jsonl and system has been running ≥10 minutes")
        sys.exit(1)
    
    # 运行所有检查脚本
    scripts = [
        'z_healthcheck.py',
        'signal_consistency.py', 
        'storage_liveness.py',
        'latency_and_queue.py'
    ]
    
    results = {}
    all_passed = True
    
    for script in scripts:
        print(f"\n--- Running {script} ---")
        result = run_script(script)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            all_passed = False
            continue
        
        print(result['stdout'])
        
        # 解析结果
        if script == 'z_healthcheck.py':
            results['z_health'] = parse_z_health_output(result['stdout'])
        elif script == 'signal_consistency.py':
            results['consistency'] = parse_consistency_output(result['stdout'])
        elif script == 'storage_liveness.py':
            results['storage'] = parse_storage_output(result['stdout'])
        elif script == 'latency_and_queue.py':
            results['latency'] = parse_latency_output(result['stdout'])
    
    if not all_passed:
        print("\nDECISION: NO-GO")
        print("Reason: One or more checks failed")
        sys.exit(1)
    
    # 评估阈值
    threshold_results = evaluate_thresholds(results)
    
    # 检查是否所有阈值都通过
    all_thresholds_passed = all(threshold_results.values())
    
    # 生成摘要
    summary = {
        'z_health': results.get('z_health', {}),
        'consistency': results.get('consistency', {}),
        'latency': results.get('latency', {}),
        'storage': results.get('storage', {}),
        'decision': 'GO' if all_thresholds_passed else 'NO-GO'
    }
    
    # 打印YAML风格摘要
    print("\n=== Summary ===")
    print(f"z_health: {summary['z_health']}")
    print(f"consistency: {summary['consistency']}")
    print(f"latency: {summary['latency']}")
    print(f"storage: {summary['storage']}")
    print(f"decision: {summary['decision']}")
    
    # 打印阈值检查结果
    print("\n=== Threshold Check Results ===")
    for key, passed in threshold_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{key}: {status}")
    
    # 写入摘要文件
    try:
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        summary_file = artifacts_dir / "shadow_summary.yaml"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"z_health: {summary['z_health']}\n")
            f.write(f"consistency: {summary['consistency']}\n")
            f.write(f"latency: {summary['latency']}\n")
            f.write(f"storage: {summary['storage']}\n")
            f.write(f"decision: {summary['decision']}\n")
        
        print(f"\nSummary written to {summary_file}")
    except Exception as e:
        print(f"WARNING: Failed to write summary file: {e}")
    
    # 最终判定
    if all_thresholds_passed:
        print("\nDECISION: GO")
        print("All thresholds passed - system ready for shadow trading")
    else:
        print("\nDECISION: NO-GO")
        failed_thresholds = [k for k, v in threshold_results.items() if not v]
        print(f"Failed thresholds: {', '.join(failed_thresholds)}")


if __name__ == "__main__":
    main()
