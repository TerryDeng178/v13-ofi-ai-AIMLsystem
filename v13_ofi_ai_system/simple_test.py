#!/usr/bin/env python3
"""
简化的数据连接测试脚本
"""

import requests
import sys

def test_metrics_server():
    try:
        response = requests.get('http://localhost:8000/metrics', timeout=5)
        if response.status_code == 200:
            lines = [line for line in response.text.split('\n') if line.startswith('strategy_')]
            print(f"OK - 指标服务器正常，生成 {len(lines)} 个策略指标")
            return True
        else:
            print(f"FAIL - 指标服务器响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL - 指标服务器连接失败: {e}")
        return False

def test_prometheus():
    try:
        response = requests.get('http://localhost:9090/api/v1/query?query=up', timeout=5)
        if response.status_code == 200:
            print("OK - Prometheus服务正常")
            return True
        else:
            print(f"FAIL - Prometheus响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL - Prometheus连接失败: {e}")
        return False

def test_grafana():
    try:
        response = requests.get('http://localhost:3000/api/health', timeout=5)
        if response.status_code == 200:
            print("OK - Grafana服务正常")
            return True
        else:
            print(f"FAIL - Grafana响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL - Grafana连接失败: {e}")
        return False

def main():
    print("=" * 50)
    print("V13 策略模式监控数据连接测试")
    print("=" * 50)
    
    results = []
    
    results.append(test_metrics_server())
    results.append(test_prometheus())
    results.append(test_grafana())
    
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    
    services = ["指标服务器", "Prometheus", "Grafana"]
    for i, (service, result) in enumerate(zip(services, results)):
        status = "正常" if result else "异常"
        print(f"{service}: {status}")
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n总体状态: {success_count}/{total_count} 服务正常")
    
    if success_count == total_count:
        print("所有服务正常，数据应该可以正常显示！")
        print("\n访问地址:")
        print("- Grafana: http://localhost:3000")
        print("- Prometheus: http://localhost:9090")
        print("- 指标端点: http://localhost:8000/metrics")
    else:
        print("部分服务异常，需要检查配置")
        print("\n建议操作:")
        if not results[0]:
            print("- 启动指标服务器: python grafana/simple_metrics_server.py 8000")
        if not results[1]:
            print("- 启动Prometheus: docker-compose up -d")
        if not results[2]:
            print("- 启动Grafana: docker-compose up -d")

if __name__ == "__main__":
    main()
