#!/usr/bin/env python3
"""
修复Grafana "No Data" 问题
"""

import requests
import json
import sys
import io

# 修复Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def test_prometheus_query():
    """测试Prometheus查询"""
    try:
        # 测试基本查询
        query = "up"
        response = requests.get(f'http://localhost:9090/api/v1/query?query={query}', timeout=10)
        data = response.json()
        
        print("🔍 Prometheus查询测试:")
        if data['status'] == 'success':
            print("  ✅ Prometheus查询正常")
            results = data['data']['result']
            print(f"  📊 找到 {len(results)} 个目标")
            
            # 检查策略指标
            strategy_query = "strategy_mode_active"
            strategy_response = requests.get(f'http://localhost:9090/api/v1/query?query={strategy_query}', timeout=10)
            strategy_data = strategy_response.json()
            
            if strategy_data['status'] == 'success':
                strategy_results = strategy_data['data']['result']
                print(f"  ✅ 策略指标查询成功，找到 {len(strategy_results)} 个结果")
                
                for result in strategy_results:
                    metric = result['metric']
                    value = result['value'][1]
                    print(f"    - {metric.get('job', 'unknown')}: {value}")
            else:
                print("  ❌ 策略指标查询失败")
        else:
            print("  ❌ Prometheus查询失败")
            
    except Exception as e:
        print(f"❌ 查询测试失败: {e}")

def check_grafana_connection():
    """检查Grafana连接"""
    try:
        response = requests.get('http://localhost:3000/api/health', timeout=5)
        print(f"✅ Grafana连接状态: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Grafana连接失败: {e}")
        return False

def main():
    print("🔧 Grafana 'No Data' 修复工具")
    print("=" * 50)
    
    # 检查Grafana连接
    grafana_ok = check_grafana_connection()
    
    # 测试Prometheus查询
    test_prometheus_query()
    
    print("\n" + "=" * 50)
    print("📋 修复建议:")
    print("1. 在Grafana中设置时间范围为 'Last 1 hour'")
    print("2. 确认仪表盘变量 $env = 'testing'")
    print("3. 确认仪表盘变量 $symbol = 'BTCUSDT' 或 'All'")
    print("4. 刷新仪表盘 (F5)")
    print("5. 如果仍有问题，重启Grafana: docker compose restart grafana")

if __name__ == '__main__':
    main()
