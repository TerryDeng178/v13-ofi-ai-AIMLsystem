#!/usr/bin/env python3
"""
诊断仪表盘数据显示问题
"""

import requests
import json
import sys
import io
from datetime import datetime, timedelta

# 修复Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def test_prometheus_queries():
    """测试Prometheus查询"""
    print("🔍 测试Prometheus查询...")
    
    queries = {
        "strategy_mode_active": "strategy_mode_active{env=\"testing\"}",
        "strategy_mode_transitions_total": "strategy_mode_transitions_total{env=\"testing\"}",
        "strategy_mode_last_change_timestamp": "strategy_mode_last_change_timestamp{env=\"testing\"}",
        "strategy_trigger_volume_usd": "strategy_trigger_volume_usd{env=\"testing\"}",
        "strategy_trigger_spread_bps": "strategy_trigger_spread_bps{env=\"testing\"}",
        "strategy_trigger_volatility": "strategy_trigger_volatility{env=\"testing\"}",
        "strategy_trigger_ofi_signal": "strategy_trigger_ofi_signal{env=\"testing\"}",
        "strategy_trigger_cvd_signal": "strategy_trigger_cvd_signal{env=\"testing\"}"
    }
    
    results = {}
    
    for name, query in queries.items():
        try:
            response = requests.get(
                f'http://localhost:9090/api/v1/query?query={query}',
                timeout=10
            )
            data = response.json()
            
            if data['status'] == 'success':
                results[name] = len(data['data']['result'])
                print(f"  ✅ {name}: {len(data['data']['result'])} 个结果")
                
                # 显示前几个结果
                for i, result in enumerate(data['data']['result'][:2]):
                    metric = result['metric']
                    value = result['value'][1]
                    print(f"    - {metric.get('job', 'unknown')}: {value}")
            else:
                results[name] = 0
                print(f"  ❌ {name}: 查询失败")
                
        except Exception as e:
            results[name] = -1
            print(f"  ❌ {name}: 错误 - {e}")
    
    return results

def test_dashboard_queries():
    """测试仪表盘中的具体查询"""
    print("\n📊 测试仪表盘查询...")
    
    dashboard_queries = {
        "Current Mode": "max without(instance,pod,symbol) (strategy_mode_active{env=\"testing\",symbol=~\"BTCUSDT\"})",
        "Last Switch Ago": "time() - max without(instance,pod) (strategy_mode_last_change_timestamp{env=\"testing\",symbol=~\"BTCUSDT\"})",
        "Switches Today": "sum without(instance,pod,symbol) (increase(strategy_mode_transitions_total{env=\"testing\",symbol=~\"BTCUSDT\"}[24h]))",
        "Switch Reason Distribution": "sum by (reason) (increase(strategy_mode_transitions_total{env=\"testing\",symbol=~\"BTCUSDT\"}[1h]))",
        "Mode Duration Trend": "sum by (mode) (increase(strategy_time_in_mode_seconds_total{env=\"testing\",symbol=~\"BTCUSDT\"}[1h])) / 3600"
    }
    
    for name, query in dashboard_queries.items():
        try:
            response = requests.get(
                f'http://localhost:9090/api/v1/query?query={query}',
                timeout=10
            )
            data = response.json()
            
            if data['status'] == 'success':
                results = data['data']['result']
                print(f"  ✅ {name}: {len(results)} 个结果")
                if results:
                    for result in results[:1]:
                        value = result['value'][1]
                        print(f"    - 值: {value}")
                else:
                    print(f"    - 无数据")
            else:
                print(f"  ❌ {name}: 查询失败 - {data.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  ❌ {name}: 错误 - {e}")

def check_metrics_server_data():
    """检查指标服务器数据"""
    print("\n📡 检查指标服务器数据...")
    
    try:
        response = requests.get('http://localhost:8000/metrics', timeout=10)
        metrics_text = response.text
        
        # 检查关键指标
        key_metrics = [
            'strategy_mode_active',
            'strategy_mode_transitions_total',
            'strategy_mode_last_change_timestamp',
            'strategy_trigger_',
            'strategy_time_in_mode_seconds_total'
        ]
        
        found_metrics = []
        for metric in key_metrics:
            if metric in metrics_text:
                found_metrics.append(metric)
                print(f"  ✅ 找到: {metric}")
            else:
                print(f"  ❌ 缺少: {metric}")
        
        # 显示一些示例数据
        print(f"\n📊 指标服务器数据示例:")
        lines = metrics_text.split('\n')
        for line in lines[:10]:
            if line and not line.startswith('#'):
                print(f"  {line}")
                
    except Exception as e:
        print(f"❌ 指标服务器检查失败: {e}")

def suggest_fixes(results):
    """建议修复方案"""
    print("\n🔧 修复建议:")
    
    if results.get('strategy_mode_active', 0) == 0:
        print("1. ❌ strategy_mode_active 指标缺失 - 检查指标服务器是否生成此指标")
    
    if results.get('strategy_mode_transitions_total', 0) == 0:
        print("2. ❌ strategy_mode_transitions_total 指标缺失 - 检查模式切换逻辑")
    
    if results.get('strategy_mode_last_change_timestamp', 0) == 0:
        print("3. ❌ strategy_mode_last_change_timestamp 指标缺失 - 检查时间戳记录")
    
    print("\n🎯 通用修复步骤:")
    print("1. 检查仪表盘时间范围设置为 'Last 1 hour'")
    print("2. 确认仪表盘变量 $env = 'testing'")
    print("3. 确认仪表盘变量 $symbol = 'BTCUSDT'")
    print("4. 检查Prometheus是否正在抓取指标服务器")
    print("5. 重启指标服务器: python grafana/simple_metrics_server.py 8000")

def main():
    print("🔍 仪表盘数据显示问题诊断")
    print("=" * 60)
    
    # 检查指标服务器数据
    check_metrics_server_data()
    
    # 测试Prometheus查询
    results = test_prometheus_queries()
    
    # 测试仪表盘查询
    test_dashboard_queries()
    
    # 建议修复方案
    suggest_fixes(results)
    
    print("\n" + "=" * 60)
    print("📋 诊断完成！请根据上述建议进行修复。")

if __name__ == '__main__':
    main()
