#!/usr/bin/env python3
"""
修复Prometheus网络连接问题
"""

import requests
import time
import subprocess
import sys

def restart_prometheus():
    """重启Prometheus服务"""
    try:
        print("重启Prometheus服务...")
        result = subprocess.run(['docker-compose', 'restart', 'prometheus'], 
                              capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            print("Prometheus重启成功")
            return True
        else:
            print(f"Prometheus重启失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"重启错误: {e}")
        return False

def wait_for_prometheus():
    """等待Prometheus启动"""
    print("等待Prometheus启动...")
    for i in range(30):
        try:
            response = requests.get('http://localhost:9090/api/v1/query?query=up', timeout=5)
            if response.status_code == 200:
                print("Prometheus已启动")
                return True
        except:
            pass
        time.sleep(2)
        print(f"等待中... ({i+1}/30)")
    
    print("Prometheus启动超时")
    return False

def check_target_status():
    """检查目标状态"""
    try:
        response = requests.get('http://localhost:9090/api/v1/targets', timeout=5)
        if response.status_code == 200:
            data = response.json()
            targets = data.get('data', {}).get('activeTargets', [])
            
            for target in targets:
                job = target.get('labels', {}).get('job', 'unknown')
                health = target.get('health', 'unknown')
                if job == 'strategy-mode-manager':
                    if health == 'up':
                        print("指标服务器连接成功！")
                        return True
                    else:
                        print(f"指标服务器连接失败: {health}")
                        return False
            print("未找到strategy-mode-manager目标")
            return False
        else:
            print(f"无法获取目标状态: {response.status_code}")
            return False
    except Exception as e:
        print(f"目标状态检查失败: {e}")
        return False

def main():
    print("=" * 50)
    print("修复Prometheus网络连接")
    print("=" * 50)
    
    # 重启Prometheus
    if restart_prometheus():
        # 等待启动
        if wait_for_prometheus():
            # 等待配置生效
            print("等待配置生效...")
            time.sleep(10)
            
            # 检查连接状态
            if check_target_status():
                print("网络连接修复成功！")
                print("现在应该可以在Grafana中看到数据了")
            else:
                print("网络连接仍然有问题")
                print("请尝试手动重启Docker服务")
        else:
            print("Prometheus启动失败")
    else:
        print("无法重启Prometheus")

if __name__ == "__main__":
    main()
