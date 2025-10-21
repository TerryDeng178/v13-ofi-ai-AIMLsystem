#!/usr/bin/env python3
"""
快速诊断Grafana "No Data" 问题
"""

import requests
import json
import time
import sys
import io
from datetime import datetime, timedelta

# 修复Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_metrics_server():
    """检查指标服务器状态"""
    try:
        # 检查健康状态
        health_response = requests.get('http://localhost:8000/health', timeout=5)
        print(f"✅ 指标服务器健康状态: {health_response.status_code}")
        
        # 检查指标数据
        metrics_response = requests.get('http://localhost:8000/metrics', timeout=5)
        metrics_text = metrics_response.text
        
        # 检查关键指标
        key_metrics = [
            'strategy_mode_active',
            'strategy_mode_transitions_total',
            'strategy_trigger_',
            'strategy_params_update_'
        ]
        
        print("\n📊 指标数据检查:")
        for metric in key_metrics:
            if metric in metrics_text:
                print(f"  ✅ 找到指标: {metric}")
            else:
                print(f"  ❌ 缺少指标: {metric}")
        
        return True
    except Exception as e:
        print(f"❌ 指标服务器检查失败: {e}")
        return False

def check_prometheus():
    """检查Prometheus状态"""
    try:
        # 检查健康状态
        health_response = requests.get('http://localhost:9090/-/healthy', timeout=5)
        print(f"✅ Prometheus健康状态: {health_response.status_code}")
        
        # 检查目标状态
        targets_response = requests.get('http://localhost:9090/api/v1/targets', timeout=5)
        targets_data = targets_response.json()
        
        print("\n🎯 Prometheus目标状态:")
        for target in targets_data['data']['activeTargets']:
            job = target['labels']['job']
            health = target['health']
            last_scrape = target['lastScrape']
            
            if health == 'up':
                print(f"  ✅ {job}: {health} (最后抓取: {last_scrape})")
            else:
                print(f"  ❌ {job}: {health}")
        
        return True
    except Exception as e:
        print(f"❌ Prometheus检查失败: {e}")
        return False

def check_grafana_data_source():
    """检查Grafana数据源"""
    try:
        # 这里需要Grafana API，暂时跳过
        print("ℹ️  Grafana数据源检查需要手动验证")
        print("   请访问: http://localhost:3000/datasources")
        print("   确认Prometheus数据源状态为绿色")
        return True
    except Exception as e:
        print(f"❌ Grafana检查失败: {e}")
        return False

def check_time_range():
    """检查时间范围设置"""
    print("\n⏰ 时间范围建议:")
    print("   1. 在Grafana仪表盘中，点击右上角时间选择器")
    print("   2. 选择 'Last 6 hours' 或 'Last 1 hour'")
    print("   3. 确保时区设置为 'Asia/Hong_Kong'")
    
    # 计算建议的时间范围
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)
    six_hours_ago = now - timedelta(hours=6)
    
    print(f"\n   建议时间范围:")
    print(f"   - 最近1小时: {one_hour_ago.strftime('%Y-%m-%d %H:%M')} 到 {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"   - 最近6小时: {six_hours_ago.strftime('%Y-%m-%d %H:%M')} 到 {now.strftime('%Y-%m-%d %H:%M')}")

def main():
    print("🔍 Grafana 'No Data' 问题诊断")
    print("=" * 50)
    
    # 检查各项服务
    metrics_ok = check_metrics_server()
    prometheus_ok = check_prometheus()
    grafana_ok = check_grafana_data_source()
    
    # 时间范围建议
    check_time_range()
    
    print("\n" + "=" * 50)
    print("📋 诊断总结:")
    
    if metrics_ok and prometheus_ok:
        print("✅ 数据流正常，问题可能在Grafana配置")
        print("\n🔧 建议解决方案:")
        print("1. 检查Grafana时间范围设置")
        print("2. 确认Prometheus数据源连接正常")
        print("3. 尝试刷新仪表盘")
        print("4. 检查仪表盘变量设置")
    else:
        print("❌ 发现数据流问题，需要修复服务")
        print("\n🔧 建议解决方案:")
        if not metrics_ok:
            print("1. 启动指标服务器: python grafana/simple_metrics_server.py 8000")
        if not prometheus_ok:
            print("2. 重启Prometheus: docker compose restart prometheus")

if __name__ == '__main__':
    main()
