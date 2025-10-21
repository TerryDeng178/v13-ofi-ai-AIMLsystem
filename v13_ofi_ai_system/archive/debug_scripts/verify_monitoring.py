#!/usr/bin/env python3
"""
V13监控系统验证脚本
检查所有服务是否正常运行，数据是否正常流动
"""

import requests
import json
import time
import sys
from urllib.parse import urljoin

# 服务配置
SERVICES = {
    'grafana': 'http://localhost:3000',
    'prometheus': 'http://localhost:9090',
    'alertmanager': 'http://localhost:9093',
    'loki': 'http://localhost:3100',
    'metrics_server': 'http://localhost:8000'
}

def check_service(name, url, endpoint='', expected_status=200):
    """检查单个服务是否可访问"""
    try:
        full_url = urljoin(url, endpoint)
        response = requests.get(full_url, timeout=5)
        if response.status_code == expected_status:
            print(f"✅ {name}: {response.status_code}")
            return True
        else:
            print(f"❌ {name}: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {name}: {str(e)}")
        return False

def check_prometheus_targets():
    """检查Prometheus目标状态"""
    try:
        response = requests.get('http://localhost:9090/api/v1/targets')
        data = response.json()
        
        targets = data['data']['activeTargets']
        print(f"\n📊 Prometheus目标状态:")
        
        for target in targets:
            name = target['labels']['job']
            health = target['health']
            last_scrape = target['lastScrape']
            
            if health == 'up':
                print(f"  ✅ {name}: {health} (最后抓取: {last_scrape})")
            else:
                print(f"  ❌ {name}: {health}")
                
        return True
    except Exception as e:
        print(f"❌ 无法获取Prometheus目标: {e}")
        return False

def check_prometheus_rules():
    """检查Prometheus规则是否加载"""
    try:
        response = requests.get('http://localhost:9090/api/v1/rules')
        data = response.json()
        
        groups = data['data']['groups']
        print(f"\n📋 Prometheus规则状态:")
        
        if groups:
            for group in groups:
                name = group['name']
                rules = group['rules']
                print(f"  ✅ {name}: {len(rules)} 条规则")
        else:
            print("  ⚠️  未找到规则组")
            
        return True
    except Exception as e:
        print(f"❌ 无法获取Prometheus规则: {e}")
        return False

def check_grafana_datasources():
    """检查Grafana数据源"""
    try:
        # 使用基本认证
        auth = ('admin', 'admin')  # 默认密码，实际应从.env读取
        
        response = requests.get('http://localhost:3000/api/datasources', auth=auth)
        datasources = response.json()
        
        print(f"\n🔗 Grafana数据源:")
        
        for ds in datasources:
            name = ds['name']
            type_name = ds['type']
            url = ds['url']
            print(f"  ✅ {name} ({type_name}): {url}")
            
        return True
    except Exception as e:
        print(f"❌ 无法获取Grafana数据源: {e}")
        return False

def check_metrics_data():
    """检查指标数据是否正常"""
    try:
        response = requests.get('http://localhost:8000/metrics')
        metrics_text = response.text
        
        # 检查关键指标
        key_metrics = [
            'strategy_mode_active',
            'strategy_mode_transitions_total',
            'strategy_trigger_',
            'strategy_params_update_'
        ]
        
        print(f"\n📈 指标数据检查:")
        
        for metric in key_metrics:
            if metric in metrics_text:
                print(f"  ✅ 找到指标: {metric}")
            else:
                print(f"  ❌ 缺少指标: {metric}")
                
        return True
    except Exception as e:
        print(f"❌ 无法获取指标数据: {e}")
        return False

def main():
    print("🔍 V13监控系统验证开始...")
    print("=" * 50)
    
    # 检查基础服务
    print("\n1. 检查基础服务:")
    service_results = []
    for name, url in SERVICES.items():
        result = check_service(name, url)
        service_results.append(result)
    
    # 检查Prometheus目标
    print("\n2. 检查Prometheus目标:")
    check_prometheus_targets()
    
    # 检查Prometheus规则
    print("\n3. 检查Prometheus规则:")
    check_prometheus_rules()
    
    # 检查Grafana数据源
    print("\n4. 检查Grafana数据源:")
    check_grafana_datasources()
    
    # 检查指标数据
    print("\n5. 检查指标数据:")
    check_metrics_data()
    
    # 总结
    print("\n" + "=" * 50)
    success_count = sum(service_results)
    total_count = len(service_results)
    
    if success_count == total_count:
        print(f"🎉 所有服务正常运行! ({success_count}/{total_count})")
        print("\n📱 访问地址:")
        print("  - Grafana: http://localhost:3000")
        print("  - Prometheus: http://localhost:9090")
        print("  - Alertmanager: http://localhost:9093")
        return 0
    else:
        print(f"⚠️  部分服务异常 ({success_count}/{total_count})")
        return 1

if __name__ == '__main__':
    sys.exit(main())
