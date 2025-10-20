"""
端口配置管理器

统一管理系统中所有组件的端口分配，避免端口冲突
提供端口分配、冲突检测、配置验证等功能

Author: V13 OFI+CVD AI System
Created: 2025-10-20
"""

import os
import socket
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PortConfig:
    """端口配置"""
    name: str
    port: int
    description: str
    component: str
    required: bool = True

class PortManager:
    """端口管理器"""
    
    def __init__(self):
        self.ports: Dict[str, PortConfig] = {}
        self._load_default_ports()
    
    def _load_default_ports(self):
        """加载默认端口配置"""
        default_ports = [
            # 核心服务端口
            PortConfig("prometheus", 9090, "Prometheus监控服务", "monitoring", True),
            PortConfig("grafana", 3000, "Grafana仪表盘服务", "monitoring", True),
            PortConfig("loki", 3100, "Loki日志聚合服务", "monitoring", True),
            
            # 应用指标端口
            PortConfig("main_metrics", 8003, "主系统指标服务", "monitoring", True),
            PortConfig("divergence_metrics", 8004, "背离检测指标服务", "divergence", True),
            PortConfig("fusion_metrics", 8005, "融合指标服务", "fusion", True),
            PortConfig("strategy_metrics", 8006, "策略模式指标服务", "strategy", True),
            
            # WebSocket服务端口
            PortConfig("websocket_proxy", 8007, "WebSocket代理服务", "websocket", False),
            PortConfig("trade_stream", 8008, "交易流服务", "trading", False),
            
            # 数据库端口
            PortConfig("postgresql", 5432, "PostgreSQL数据库", "database", False),
            PortConfig("redis", 6379, "Redis缓存", "cache", False),
            
            # 开发调试端口
            PortConfig("debug_server", 8009, "调试服务器", "debug", False),
            PortConfig("test_server", 8010, "测试服务器", "test", False),
        ]
        
        for port_config in default_ports:
            self.ports[port_config.name] = port_config
    
    def get_port(self, name: str) -> int:
        """获取指定组件的端口"""
        if name not in self.ports:
            raise ValueError(f"Unknown port name: {name}")
        
        port_config = self.ports[name]
        
        # 检查环境变量覆盖
        env_var = f"V13_PORT_{name.upper()}"
        if env_var in os.environ:
            try:
                port = int(os.environ[env_var])
                logger.info(f"Port {name} overridden by environment variable {env_var}={port}")
                return port
            except ValueError:
                logger.warning(f"Invalid port value in {env_var}: {os.environ[env_var]}")
        
        return port_config.port
    
    def check_port_available(self, port: int) -> bool:
        """检查端口是否可用"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def find_available_port(self, start_port: int = 8000, max_port: int = 8999) -> Optional[int]:
        """查找可用端口"""
        for port in range(start_port, max_port + 1):
            if self.check_port_available(port):
                return port
        return None
    
    def validate_ports(self) -> Dict[str, bool]:
        """验证所有端口配置"""
        results = {}
        
        for name, port_config in self.ports.items():
            port = self.get_port(name)
            available = self.check_port_available(port)
            results[name] = available
            
            if not available and port_config.required:
                logger.error(f"Required port {name} ({port}) is not available")
            elif not available:
                logger.warning(f"Optional port {name} ({port}) is not available")
            else:
                logger.debug(f"Port {name} ({port}) is available")
        
        return results
    
    def get_conflicts(self) -> List[str]:
        """获取端口冲突列表"""
        conflicts = []
        used_ports: Set[int] = set()
        
        for name, port_config in self.ports.items():
            port = self.get_port(name)
            if port in used_ports:
                conflicts.append(f"Port {port} used by multiple components")
            used_ports.add(port)
        
        return conflicts
    
    def get_port_summary(self) -> Dict[str, Dict[str, any]]:
        """获取端口配置摘要"""
        summary = {}
        
        for name, port_config in self.ports.items():
            port = self.get_port(name)
            available = self.check_port_available(port)
            
            summary[name] = {
                'port': port,
                'description': port_config.description,
                'component': port_config.component,
                'required': port_config.required,
                'available': available,
                'env_var': f"V13_PORT_{name.upper()}"
            }
        
        return summary
    
    def print_port_status(self):
        """打印端口状态"""
        print("=" * 80)
        print("V13系统端口配置状态")
        print("=" * 80)
        
        summary = self.get_port_summary()
        
        # 按组件分组
        components = {}
        for name, info in summary.items():
            component = info['component']
            if component not in components:
                components[component] = []
            components[component].append((name, info))
        
        for component, ports in components.items():
            print(f"\n📦 {component.upper()} 组件:")
            for name, info in ports:
                status = "✅" if info['available'] else "❌"
                required = "必需" if info['required'] else "可选"
                print(f"  {status} {name:20} : {info['port']:4} ({info['description']}) [{required}]")
        
        # 检查冲突
        conflicts = self.get_conflicts()
        if conflicts:
            print(f"\n⚠️  端口冲突:")
            for conflict in conflicts:
                print(f"  - {conflict}")
        else:
            print(f"\n✅ 无端口冲突")
        
        print("=" * 80)

# 全局端口管理器实例
port_manager = PortManager()

def get_port(name: str) -> int:
    """获取端口号的便捷函数"""
    return port_manager.get_port(name)

def check_ports() -> bool:
    """检查所有端口是否可用"""
    results = port_manager.validate_ports()
    return all(results.values())

def print_port_status():
    """打印端口状态"""
    port_manager.print_port_status()

if __name__ == "__main__":
    # 打印端口状态
    print_port_status()
    
    # 检查端口可用性
    if check_ports():
        print("\n🎉 所有端口配置正常")
    else:
        print("\n❌ 存在端口配置问题")
