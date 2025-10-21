#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置验证脚本 - 验证Round 2优化版配置是否生效
"""

import yaml
import json
import os
import sys

def validate_config():
    """验证配置文件"""
    print("开始配置验证...")
    
    # 读取配置文件
    config_path = 'config/system.yaml'
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"配置文件读取失败: {e}")
        return False
    
    print("配置文件读取成功")
    
    # 验证Fusion指标配置
    print("\n=== Fusion指标配置验证 ===")
    fusion_config = config.get('fusion_metrics', {})
    if not fusion_config:
        print("Fusion指标配置缺失")
        return False
    
    print(f"版本: {fusion_config.get('version', 'N/A')}")
    
    # 验证基线权重
    weights = fusion_config.get('weights', {})
    print(f"基线权重: w_ofi={weights.get('w_ofi')}, w_cvd={weights.get('w_cvd')}, gate={weights.get('gate')}")
    
    # 验证切片覆盖
    slice_overrides = fusion_config.get('slice_overrides', {})
    if 'active_period' in slice_overrides:
        active_weights = slice_overrides['active_period'].get('weights', {})
        print(f"Active时段权重: w_ofi={active_weights.get('w_ofi')}, w_cvd={active_weights.get('w_cvd')}, gate={active_weights.get('gate')}")
    
    if 'quiet_period' in slice_overrides:
        quiet_weights = slice_overrides['quiet_period'].get('weights', {})
        print(f"Quiet时段权重: w_ofi={quiet_weights.get('w_ofi')}, w_cvd={quiet_weights.get('w_cvd')}, gate={quiet_weights.get('gate')}")
    
    # 验证信号分析配置
    print("\n=== 信号分析配置验证 ===")
    signal_config = config.get('signal_analysis', {})
    if not signal_config:
        print("信号分析配置缺失")
        return False
    
    print(f"版本: {signal_config.get('version', 'N/A')}")
    
    # 验证基线配置
    baseline = signal_config.get('baseline', {})
    labels = baseline.get('labels', {})
    print(f"标签类型: {labels.get('type', 'N/A')}")
    print(f"前瞻对齐: {labels.get('forward_direction', 'N/A')}")
    
    calibration = baseline.get('calibration', {})
    print(f"校准方法: {calibration.get('method', 'N/A')}")
    print(f"训练窗口: {calibration.get('train_window', 'N/A')}s")
    print(f"测试窗口: {calibration.get('test_window', 'N/A')}s")
    
    print(f"CVD自动翻转: {baseline.get('cvd_auto_flip', 'N/A')}")
    print(f"合并容差: {baseline.get('merge_tolerance_ms', 'N/A')}ms")
    
    fusion = baseline.get('fusion', {})
    print(f"Fusion配置: w_ofi={fusion.get('w_ofi')}, w_cvd={fusion.get('w_cvd')}, gate={fusion.get('gate')}")
    
    # 验证监控配置
    print("\n=== 监控配置验证 ===")
    monitoring_config = signal_config.get('monitoring', {})
    prometheus_config = monitoring_config.get('prometheus', {})
    
    print(f"Prometheus端口: {prometheus_config.get('port', 'N/A')}")
    print(f"抓取间隔: {prometheus_config.get('scrape_interval', 'N/A')}")
    
    # 验证告警规则
    alerts_config = monitoring_config.get('alerts', {})
    rules = alerts_config.get('rules', [])
    print(f"告警规则数量: {len(rules)}")
    
    for i, rule in enumerate(rules, 1):
        print(f"  规则{i}: {rule.get('name', 'N/A')} - {rule.get('summary', 'N/A')}")
    
    print("\n配置验证完成")
    return True

def validate_config_fingerprint():
    """验证配置指纹"""
    print("\n=== 配置指纹验证 ===")
    
    fingerprint_path = 'config/config_fingerprint_v2.0.json'
    if not os.path.exists(fingerprint_path):
        print(f"配置指纹文件不存在: {fingerprint_path}")
        return False
    
    try:
        with open(fingerprint_path, 'r', encoding='utf-8') as f:
            fingerprint = json.load(f)
    except Exception as e:
        print(f"配置指纹文件读取失败: {e}")
        return False
    
    print("配置指纹文件读取成功")
    
    config_fp = fingerprint.get('config_fingerprint', {})
    print(f"版本: {config_fp.get('version', 'N/A')}")
    print(f"时间戳: {config_fp.get('timestamp', 'N/A')}")
    print(f"描述: {config_fp.get('description', 'N/A')}")
    
    # 验证验证状态
    validation = config_fp.get('validation', {})
    print(f"DoD Gate: {validation.get('dod_gate', 'N/A')}")
    print(f"Platt校准: {validation.get('platt_calibration', 'N/A')}")
    print(f"Fusion权重: {validation.get('fusion_weights', 'N/A')}")
    print(f"信号质量: {validation.get('signal_quality', 'N/A')}")
    
    # 验证测试结果
    test_results = validation.get('test_results', {})
    print("测试结果:")
    for test_name, result in test_results.items():
        print(f"  {test_name}: {result}")
    
    # 验证兼容性
    compatibility = config_fp.get('compatibility', {})
    print(f"向后兼容: {compatibility.get('backward_compatible', 'N/A')}")
    print(f"迁移需求: {compatibility.get('migration_required', 'N/A')}")
    
    print("\n配置指纹验证完成")
    return True

if __name__ == "__main__":
    print("开始配置验证...")
    
    # 验证主配置
    config_valid = validate_config()
    
    # 验证配置指纹
    fingerprint_valid = validate_config_fingerprint()
    
    if config_valid and fingerprint_valid:
        print("\n所有配置验证通过！")
        sys.exit(0)
    else:
        print("\n配置验证失败！")
        sys.exit(1)
