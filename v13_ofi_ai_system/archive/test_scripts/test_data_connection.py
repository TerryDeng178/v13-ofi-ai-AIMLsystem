#!/usr/bin/env python3
"""
数据连接测试脚本
检查指标服务器、Prometheus和Grafana的连接状态
"""

import requests
import time
import sys

def test_metrics_server():
    """测试指标服务器"""
    try:
        print("🔍 测试指标服务器...")
        response = requests.get('http://localhost:8000/metrics', timeout=5)
        if response.status_code == 200:
            lines = [line for line in response.text.split('\n') if line.startswith('strategy_')]
            print(f"✅ 指标服务器正常，生成 {len(lines)} 个策略指标")
            return True
        else:
            print(f"❌ 指标服务器响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 指标服务器连接失败: {e}")
        return False

def test_prometheus():
    """测试Prometheus"""
    try:
        print("🔍 测试Prometheus...")
        response = requests.get('http://localhost:9090/api/v1/query?query=up', timeout=5)
        if response.status_code == 200:
            print("✅ Prometheus服务正常")
            return True
        else:
            print(f"❌ Prometheus响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prometheus连接失败: {e}")
        return False

def test_grafana():
    """测试Grafana"""
    try:
        print("🔍 测试Grafana...")
        response = requests.get('http://localhost:3000/api/health', timeout=5)
        if response.status_code == 200:
            print("✅ Grafana服务正常")
            return True
        else:
            print(f"❌ Grafana响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Grafana连接失败: {e}")
        return False

def check_prometheus_targets():
    """检查Prometheus目标"""
    try:
        print("🔍 检查Prometheus目标...")
        response = requests.get('http://localhost:9090/api/v1/targets', timeout=5)
        if response.status_code == 200:
            data = response.json()
            targets = data.get('data', {}).get('activeTargets', [])
            print(f"✅ 发现 {len(targets)} 个活跃目标")
            
            for target in targets:
                job = target.get('labels', {}).get('job', 'unknown')
                health = target.get('health', 'unknown')
                print(f"   - {job}: {health}")
            return True
        else:
            print(f"❌ 无法获取目标列表: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 目标检查失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("V13 策略模式监控数据连接测试")
    print("=" * 50)
    
    results = []
    
    # 测试各项服务
    results.append(test_metrics_server())
    results.append(test_prometheus())
    results.append(test_grafana())
    results.append(check_prometheus_targets())
    
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    
    services = ["指标服务器", "Prometheus", "Grafana", "Prometheus目标"]
    for i, (service, result) in enumerate(zip(services, results)):
        status = "✅ 正常" if result else "❌ 异常"
        print(f"{service}: {status}")
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n总体状态: {success_count}/{total_count} 服务正常")
    
    if success_count == total_count:
        print("🎉 所有服务正常，数据应该可以正常显示！")
        print("\n📊 访问地址:")
        print("- Grafana: http://localhost:3000")
        print("- Prometheus: http://localhost:9090")
        print("- 指标端点: http://localhost:8000/metrics")
    else:
        print("⚠️  部分服务异常，需要检查配置")
        print("\n🔧 建议操作:")
        if not results[0]:
            print("- 启动指标服务器: python grafana/simple_metrics_server.py 8000")
        if not results[1]:
            print("- 启动Prometheus: docker-compose up -d")
        if not results[2]:
            print("- 启动Grafana: docker-compose up -d")

if __name__ == "__main__":
    main()
