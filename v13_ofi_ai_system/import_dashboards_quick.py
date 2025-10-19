#!/usr/bin/env python3
"""
快速导入Grafana仪表盘
"""

import requests
import json
import os
import sys
import io

# 修复Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def import_dashboard(dashboard_file, grafana_url="http://localhost:3000"):
    """导入单个仪表盘"""
    try:
        # 读取仪表盘JSON文件
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            dashboard_json = json.load(f)
        
        # 准备导入数据
        import_data = {
            "dashboard": dashboard_json,
            "overwrite": True,
            "inputs": []
        }
        
        # 发送导入请求
        response = requests.post(
            f"{grafana_url}/api/dashboards/db",
            json=import_data,
            headers={'Content-Type': 'application/json'},
            auth=('admin', 'admin')  # 默认用户名密码
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 成功导入: {dashboard_json.get('title', 'Unknown')}")
            return True
        else:
            print(f"❌ 导入失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 导入错误: {e}")
        return False

def main():
    print("📊 快速导入Grafana仪表盘")
    print("=" * 50)
    
    # 检查仪表盘文件
    dashboard_files = [
        "grafana/dashboards/strategy_mode_overview.json",
        "grafana/dashboards/strategy_performance.json", 
        "grafana/dashboards/strategy_alerts.json"
    ]
    
    print("🔍 检查仪表盘文件:")
    for file_path in dashboard_files:
        if os.path.exists(file_path):
            print(f"  ✅ 找到: {file_path}")
        else:
            print(f"  ❌ 缺少: {file_path}")
    
    print("\n📥 开始导入仪表盘:")
    
    success_count = 0
    for file_path in dashboard_files:
        if os.path.exists(file_path):
            if import_dashboard(file_path):
                success_count += 1
    
    print(f"\n📋 导入结果: {success_count}/{len(dashboard_files)} 个仪表盘导入成功")
    
    if success_count > 0:
        print("\n🎉 导入完成！现在请:")
        print("1. 刷新Grafana页面 (F5)")
        print("2. 点击左侧导航栏的 'Dashboards'")
        print("3. 您应该能看到导入的仪表盘")
        print("4. 点击任意一个仪表盘名称进入")
    else:
        print("\n❌ 导入失败，请检查:")
        print("1. Grafana是否正在运行")
        print("2. 仪表盘文件是否存在")
        print("3. 网络连接是否正常")

if __name__ == '__main__':
    main()
