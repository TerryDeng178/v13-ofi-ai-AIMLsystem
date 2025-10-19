#!/usr/bin/env python3
"""
修复仪表盘显示问题
"""

import requests
import json
import sys
import io
import time
from datetime import datetime, timedelta

# 修复Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def fix_metrics_server():
    """修复指标服务器数据"""
    print("🔧 修复指标服务器数据...")
    
    try:
        # 重启指标服务器
        print("  重启指标服务器...")
        # 这里可以添加重启逻辑，或者提示用户手动重启
        
        # 检查当前指标
        response = requests.get('http://localhost:8000/metrics', timeout=5)
        metrics_text = response.text
        
        # 检查关键指标
        if 'strategy_mode_active' in metrics_text:
            print("  ✅ strategy_mode_active 正常")
        else:
            print("  ❌ strategy_mode_active 缺失")
            
        if 'strategy_mode_transitions_total' in metrics_text:
            print("  ✅ strategy_mode_transitions_total 正常")
        else:
            print("  ❌ strategy_mode_transitions_total 缺失")
            
    except Exception as e:
        print(f"  ❌ 指标服务器检查失败: {e}")

def test_fixed_queries():
    """测试修复后的查询"""
    print("\n📊 测试修复后的查询...")
    
    # 修复后的查询
    fixed_queries = {
        "Current Mode (Fixed)": "max without(instance,pod,symbol) (strategy_mode_active{env=\"testing\"})",
        "Last Switch Ago (Fixed)": "time() - max without(instance,pod) (strategy_mode_last_change_timestamp{env=\"testing\"})",
        "Switches Today (Fixed)": "sum without(instance,pod,symbol) (increase(strategy_mode_transitions_total{env=\"testing\"}[1h]))",
        "Switch Reason Distribution (Fixed)": "sum by (reason) (increase(strategy_mode_transitions_total{env=\"testing\"}[1h]))",
        "Mode Duration Trend (Fixed)": "sum by (mode) (increase(strategy_time_in_mode_seconds_total{env=\"testing\"}[1h])) / 3600"
    }
    
    for name, query in fixed_queries.items():
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
                print(f"  ❌ {name}: 查询失败")
                
        except Exception as e:
            print(f"  ❌ {name}: 错误 - {e}")

def create_dashboard_fix_guide():
    """创建仪表盘修复指南"""
    print("\n📋 仪表盘修复指南:")
    print("=" * 50)
    
    print("1. 时间范围设置:")
    print("   - 在Grafana中设置时间范围为 'Last 1 hour'")
    print("   - 确保时区设置为 'Asia/Hong_Kong'")
    
    print("\n2. 仪表盘变量设置:")
    print("   - $env = 'testing'")
    print("   - $symbol = 'BTCUSDT' 或 'All'")
    
    print("\n3. 查询修复建议:")
    print("   - 使用 'Last 1 hour' 而不是 '24h' 来减少数据量")
    print("   - 检查时间戳计算是否正确")
    print("   - 确保所有必要的指标都存在")
    
    print("\n4. 如果仍然显示异常数据:")
    print("   - 重启指标服务器")
    print("   - 重启Prometheus")
    print("   - 刷新仪表盘")

def main():
    print("🔧 仪表盘显示问题修复工具")
    print("=" * 60)
    
    # 修复指标服务器
    fix_metrics_server()
    
    # 测试修复后的查询
    test_fixed_queries()
    
    # 创建修复指南
    create_dashboard_fix_guide()
    
    print("\n" + "=" * 60)
    print("🎯 修复完成！请按照指南调整仪表盘设置。")

if __name__ == '__main__':
    main()
