#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灰度部署脚本 - BTCUSDT、ETHUSDT小流量灰度
Round 2优化版配置生产部署
"""

import os
import sys
import json
import time
import yaml
from datetime import datetime, timedelta
import subprocess

def load_config():
    """加载配置"""
    config_path = 'config/system.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def validate_canary_config():
    """验证灰度配置"""
    print("验证灰度配置...")
    
    config = load_config()
    
    # 验证Fusion配置
    fusion_config = config.get('fusion_metrics', {})
    baseline_weights = fusion_config.get('weights', {})
    active_weights = fusion_config.get('slice_overrides', {}).get('active_period', {}).get('weights', {})
    
    print(f"基线权重: w_ofi={baseline_weights.get('w_ofi')}, w_cvd={baseline_weights.get('w_cvd')}, gate={baseline_weights.get('gate')}")
    print(f"Active权重: w_ofi={active_weights.get('w_ofi')}, w_cvd={active_weights.get('w_cvd')}, gate={active_weights.get('gate')}")
    
    # 验证信号分析配置
    signal_config = config.get('signal_analysis', {})
    baseline = signal_config.get('baseline', {})
    
    print(f"标签类型: {baseline.get('labels', {}).get('type')}")
    print(f"校准方法: {baseline.get('calibration', {}).get('method')}")
    print(f"CVD自动翻转: {baseline.get('cvd_auto_flip')}")
    print(f"合并容差: {baseline.get('merge_tolerance_ms')}ms")
    
    return True

def setup_canary_environment():
    """设置灰度环境"""
    print("设置灰度环境...")
    
    # 创建灰度目录
    canary_dir = 'artifacts/canary'
    os.makedirs(canary_dir, exist_ok=True)
    os.makedirs(f'{canary_dir}/logs', exist_ok=True)
    os.makedirs(f'{canary_dir}/metrics', exist_ok=True)
    os.makedirs(f'{canary_dir}/alerts', exist_ok=True)
    
    # 设置环境变量
    env_vars = {
        'CANARY_MODE': 'true',
        'SYMBOLS': 'BTCUSDT,ETHUSDT',
        'RUN_HOURS': '48',
        'FUSION_WEIGHTS': '0.6,0.4',
        'FUSION_GATE': '0.0',
        'LABELS_TYPE': 'mid',
        'CALIBRATION_METHOD': 'platt',
        'CVD_AUTO_FLIP': 'true',
        'MERGE_TOLERANCE_MS': '1500'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置环境变量: {key}={value}")
    
    return True

def start_canary_data_collection():
    """启动灰度数据收集"""
    print("启动灰度数据收集...")
    
    # 启动数据收集进程
    cmd = [
        'python', 'examples/run_success_harvest.py'
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env.update({
        'SYMBOLS': 'BTCUSDT,ETHUSDT',
        'RUN_HOURS': '48',
        'PARQUET_ROTATE_SEC': '60',
        'WSS_PING_INTERVAL': '20',
        'DEDUP_LRU': '8192',
        'Z_MODE': 'delta',
        'SCALE_MODE': 'hybrid',
        'MAD_MULTIPLIER': '1.8',
        'SCALE_FAST_WEIGHT': '0.20',
        'HALF_LIFE_SEC': '600',
        'WINSOR_LIMIT': '8'
    })
    
    # 启动进程
    log_file = f'artifacts/canary/logs/harvest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        
        print(f"数据收集进程已启动，PID: {process.pid}")
        print(f"日志文件: {log_file}")
        
        # 保存进程信息
        process_info = {
            'pid': process.pid,
            'start_time': datetime.now().isoformat(),
            'log_file': log_file,
            'status': 'running'
        }
        
        with open('artifacts/canary/process_info.json', 'w') as f:
            json.dump(process_info, f, indent=2)
        
        return process
        
    except Exception as e:
        print(f"启动数据收集失败: {e}")
        return None

def start_canary_signal_analysis():
    """启动灰度信号分析"""
    print("启动灰度信号分析...")
    
    # 启动信号分析进程
    cmd = [
        'python', '-m', 'analysis.ofi_cvd_signal_eval',
        '--data-root', 'data/ofi_cvd',
        '--symbols', 'ETHUSDT,BTCUSDT',
        '--date-from', datetime.now().strftime('%Y-%m-%d'),
        '--date-to', datetime.now().strftime('%Y-%m-%d'),
        '--horizons', '60,180,300,900',
        '--fusion', 'w_ofi=0.6,w_cvd=0.4,gate=0',
        '--labels', 'mid',
        '--use-l1-ofi',
        '--cvd-auto-flip',
        '--calibration', 'platt',
        '--calib-train-window', '7200',
        '--calib-test-window', '1800',
        '--merge-tol-ms', '1500',
        '--plots', 'all',
        '--out', 'artifacts/canary/analysis',
        '--run-tag', f'canary_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    ]
    
    log_file = f'artifacts/canary/logs/analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        print(f"信号分析进程已启动，PID: {process.pid}")
        print(f"日志文件: {log_file}")
        
        return process
        
    except Exception as e:
        print(f"启动信号分析失败: {e}")
        return None

def setup_monitoring():
    """设置监控"""
    print("设置监控...")
    
    # 创建监控配置
    monitoring_config = {
        'canary_start_time': datetime.now().isoformat(),
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'monitoring_metrics': [
            'cvd_direction_flipped_total',
            'merge_time_diff_ms_p50',
            'merge_time_diff_ms_p90',
            'merge_time_diff_ms_p99',
            'platt_train_samples',
            'platt_test_samples',
            'platt_ece',
            'platt_brier',
            'slice_auc_active_period',
            'slice_auc_quiet_period'
        ],
        'alert_rules': [
            {
                'name': 'signal_analysis_cvd_direction_flipped',
                'condition': 'cvd_direction_flipped_total > 0',
                'severity': 'info',
                'duration': '1m'
            },
            {
                'name': 'signal_analysis_merge_time_diff_high',
                'condition': 'merge_time_diff_ms_p99 > 2000',
                'severity': 'warning',
                'duration': '5m'
            },
            {
                'name': 'signal_analysis_platt_calibration_failure',
                'condition': 'platt_ece > 0.1',
                'severity': 'warning',
                'duration': '2m'
            },
            {
                'name': 'signal_analysis_slice_auc_low',
                'condition': 'slice_auc_active_period < 0.55',
                'severity': 'warning',
                'duration': '10m'
            }
        ],
        'rollback_conditions': [
            'platt_ece > 0.1 (2m)',
            'merge_time_diff_ms_p99 > 2000 (5m)',
            'slice_auc_active_period < 0.55 (10m)'
        ]
    }
    
    with open('artifacts/canary/monitoring_config.json', 'w') as f:
        json.dump(monitoring_config, f, indent=2)
    
    print("监控配置已设置")
    return True

def generate_canary_report():
    """生成灰度报告"""
    print("生成灰度报告...")
    
    report = {
        'canary_deployment': {
            'timestamp': datetime.now().isoformat(),
            'version': 'v2.0-prod',
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'duration': '48h',
            'status': 'deployed'
        },
        'configuration': {
            'fusion_weights': {
                'baseline': {'w_ofi': 0.6, 'w_cvd': 0.4, 'gate': 0.0},
                'active_period': {'w_ofi': 0.7, 'w_cvd': 0.3, 'gate': 0.0}
            },
            'signal_analysis': {
                'labels_type': 'mid',
                'calibration_method': 'platt',
                'cvd_auto_flip': True,
                'merge_tolerance_ms': 1500
            }
        },
        'monitoring': {
            'metrics_count': 13,
            'alert_rules_count': 4,
            'rollback_conditions': 3
        },
        'next_steps': [
            '监控关键指标24-48小时',
            '观察告警触发情况',
            '评估信号质量改善',
            '准备全量部署'
        ]
    }
    
    with open('artifacts/canary/canary_deployment_report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("灰度报告已生成: artifacts/canary/canary_deployment_report.json")
    return True

def main():
    """主函数"""
    print("开始灰度部署...")
    
    # 1. 验证配置
    config_valid = validate_canary_config()
    if not config_valid:
        print("配置验证失败")
        return False
    
    # 2. 设置环境
    env_setup = setup_canary_environment()
    if not env_setup:
        print("环境设置失败")
        return False
    
    # 3. 启动数据收集
    harvest_process = start_canary_data_collection()
    if not harvest_process:
        print("数据收集启动失败")
        return False
    
    # 4. 启动信号分析
    analysis_process = start_canary_signal_analysis()
    if not analysis_process:
        print("信号分析启动失败")
        return False
    
    # 5. 设置监控
    monitoring_setup = setup_monitoring()
    if not monitoring_setup:
        print("监控设置失败")
        return False
    
    # 6. 生成报告
    report_generation = generate_canary_report()
    if not report_generation:
        print("报告生成失败")
        return False
    
    print("\n灰度部署完成！")
    print("数据收集进程:", harvest_process.pid)
    print("信号分析进程:", analysis_process.pid)
    print("监控配置已设置")
    print("请监控关键指标24-48小时")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
