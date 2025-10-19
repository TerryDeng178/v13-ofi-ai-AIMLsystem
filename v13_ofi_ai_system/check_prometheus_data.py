#!/usr/bin/env python3
"""
检查Prometheus数据采集状态
"""

import requests
import json

def check_prometheus_targets():
    """检查Prometheus目标状态"""
    try:
        response = requests.get('http://localhost:9090/api/v1/targets', timeout=5)
        if response.status_code == 200:
            data = response.json()
            targets = data.get('data', {}).get('activeTargets', [])
            
            print("Prometheus目标状态:")
            for target in targets:
                job = target.get('labels', {}).get('job', 'unknown')
                health = target.get('health', 'unknown')
                last_error = target.get('lastError', '')
                print(f"  - {job}: {health}")
                if last_error:
                    print(f"    错误: {last_error}")
            return targets
        else:
            print(f"无法获取目标列表: {response.status_code}")
            return []
    except Exception as e:
        print(f"目标检查失败: {e}")
        return []

def check_strategy_metrics():
    """检查策略指标是否被采集"""
    try:
        response = requests.get('http://localhost:9090/api/v1/query?query=strategy_mode_active', timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = data.get('data', {}).get('result', [])
            
            print(f"策略指标查询结果: {len(results)} 个结果")
            for result in results:
                metric = result.get('metric', {})
                value = result.get('value', [])
                print(f"  - {metric}: {value}")
            return len(results) > 0
        else:
            print(f"策略指标查询失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"策略指标查询错误: {e}")
        return False

def check_metrics_endpoint():
    """检查指标端点"""
    try:
        response = requests.get('http://localhost:8000/metrics', timeout=5)
        if response.status_code == 200:
            lines = [line for line in response.text.split('\n') if line.startswith('strategy_')]
            print(f"指标端点状态: 正常，{len(lines)} 个策略指标")
            return True
        else:
            print(f"指标端点异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"指标端点检查失败: {e}")
        return False

def main():
    print("=" * 60)
    print("Prometheus数据采集诊断")
    print("=" * 60)
    
    # 检查指标端点
    print("\n1. 检查指标端点...")
    check_metrics_endpoint()
    
    # 检查Prometheus目标
    print("\n2. 检查Prometheus目标...")
    targets = check_prometheus_targets()
    
    # 检查策略指标
    print("\n3. 检查策略指标采集...")
    has_data = check_strategy_metrics()
    
    print("\n" + "=" * 60)
    print("诊断结果")
    print("=" * 60)
    
    if has_data:
        print("数据采集正常！")
        print("如果Grafana中没有数据，请检查:")
        print("1. Grafana数据源配置")
        print("2. 仪表盘时间范围设置")
        print("3. 查询语法是否正确")
    else:
        print("数据采集异常！")
        print("可能的原因:")
        print("1. Prometheus无法连接到指标服务器")
        print("2. 指标服务器端口被占用")
        print("3. Prometheus配置问题")
        
        print("\n建议解决方案:")
        print("1. 重启指标服务器")
        print("2. 检查Prometheus配置")
        print("3. 重启Docker服务")

if __name__ == "__main__":
    main()
