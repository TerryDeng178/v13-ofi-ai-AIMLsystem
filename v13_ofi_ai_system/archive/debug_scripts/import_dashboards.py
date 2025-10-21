#!/usr/bin/env python3
"""
Grafana仪表盘导入脚本
通过API自动导入仪表盘配置
"""

import requests
import json
import os
import sys

def import_dashboard(grafana_url, api_key, dashboard_file):
    """导入单个仪表盘"""
    
    # 读取仪表盘JSON文件
    with open(dashboard_file, 'r', encoding='utf-8') as f:
        dashboard_data = json.load(f)
    
    # 准备API请求
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # 导入仪表盘
    import_url = f"{grafana_url}/api/dashboards/import"
    payload = {
        "dashboard": dashboard_data["dashboard"],
        "overwrite": True
    }
    
    try:
        response = requests.post(import_url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 成功导入仪表盘: {result.get('title', 'Unknown')}")
            return True
        else:
            print(f"❌ 导入失败: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ 导入错误: {e}")
        return False

def setup_prometheus_datasource(grafana_url, api_key):
    """配置Prometheus数据源"""
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    datasource_config = {
        "name": "Prometheus",
        "type": "prometheus",
        "url": "http://prometheus:9090",
        "access": "proxy",
        "isDefault": True,
        "editable": True
    }
    
    try:
        # 检查数据源是否已存在
        check_url = f"{grafana_url}/api/datasources/name/Prometheus"
        response = requests.get(check_url, headers=headers)
        
        if response.status_code == 200:
            print("✅ Prometheus数据源已存在")
            return True
        
        # 创建数据源
        create_url = f"{grafana_url}/api/datasources"
        response = requests.post(create_url, headers=headers, json=datasource_config)
        
        if response.status_code == 200:
            print("✅ 成功创建Prometheus数据源")
            return True
        else:
            print(f"❌ 创建数据源失败: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ 数据源配置错误: {e}")
        return False

def main():
    """主函数"""
    
    # 配置
    grafana_url = "http://localhost:3000"
    api_key = input("请输入Grafana API Key (或按Enter跳过API导入): ").strip()
    
    if not api_key:
        print("⚠️  跳过API导入，请使用手动方式导入仪表盘")
        print("📋 请按照 GRAFANA_MANUAL_SETUP.md 中的步骤手动导入")
        return
    
    print("🚀 开始导入Grafana配置...")
    
    # 配置数据源
    print("\n1. 配置Prometheus数据源...")
    setup_prometheus_datasource(grafana_url, api_key)
    
    # 导入仪表盘
    dashboard_files = [
        "grafana/dashboards/strategy_mode_overview.json",
        "grafana/dashboards/strategy_performance.json", 
        "grafana/dashboards/strategy_alerts.json"
    ]
    
    print("\n2. 导入仪表盘...")
    success_count = 0
    
    for dashboard_file in dashboard_files:
        if os.path.exists(dashboard_file):
            print(f"   导入 {dashboard_file}...")
            if import_dashboard(grafana_url, api_key, dashboard_file):
                success_count += 1
        else:
            print(f"   ❌ 文件不存在: {dashboard_file}")
    
    print(f"\n📊 导入完成: {success_count}/{len(dashboard_files)} 个仪表盘")
    
    if success_count > 0:
        print("🎉 配置完成！请访问 http://localhost:3000 查看仪表盘")
    else:
        print("⚠️  请使用手动方式导入仪表盘")

if __name__ == "__main__":
    main()
