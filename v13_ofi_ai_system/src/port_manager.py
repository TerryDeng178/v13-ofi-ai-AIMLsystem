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
    
    def __init__(self, config_loader=None):
        """
        初始化端口管理器
        
        Args:
            config_loader: 统一配置加载器实例，用于加载端口配置
        """
        self.config_loader = config_loader
        self.ports: Dict[str, PortConfig] = {}
        self._load_default_ports()
        self._load_from_config()
    
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
    
    def _load_from_config(self):
        """从配置加载器加载端口配置"""
        if not self.config_loader:
            return
        
        try:
            # 从配置中获取端口配置 (monitoring.prometheus.port 等)
            ports_config = self.config_loader.get("monitoring.prometheus", {})
            if isinstance(ports_config, dict) and "port" in ports_config:
                port = ports_config["port"]
                if "prometheus" in self.ports:
                    self.ports["prometheus"].port = int(port)
            
            # divergence_metrics, fusion_metrics 等
            div_metrics = self.config_loader.get("monitoring.divergence_metrics", {})
            if isinstance(div_metrics, dict) and "port" in div_metrics:
                if "divergence_metrics" in self.ports:
                    self.ports["divergence_metrics"].port = int(div_metrics["port"])
            
            fusion_metrics = self.config_loader.get("monitoring.fusion_metrics", {})
            if isinstance(fusion_metrics, dict) and "port" in fusion_metrics:
                if "fusion_metrics" in self.ports:
                    self.ports["fusion_metrics"].port = int(fusion_metrics["port"])
            
            strategy_metrics = self.config_loader.get("strategy_mode.monitoring.prometheus", {})
            if isinstance(strategy_metrics, dict) and "port" in strategy_metrics:
                if "strategy_metrics" in self.ports:
                    self.ports["strategy_metrics"].port = int(strategy_metrics["port"])
            
            trade_stream = self.config_loader.get("trade_stream.monitoring.prometheus", {})
            if isinstance(trade_stream, dict) and "port" in trade_stream:
                if "trade_stream" in self.ports:
                    self.ports["trade_stream"].port = int(trade_stream["port"])
        
        except Exception as e:
            logger.warning(f"Failed to load ports from config: {e}, using defaults")
    
    def get_port(self, name: str) -> int:
        """获取指定组件的端口"""
        if name not in self.ports:
            raise ValueError(f"Unknown port name: {name}")
        
        port_config = self.ports[name]
        
        # 优先从配置加载器获取环境变量覆盖（V13__PORT_*）
        # 配置加载器已经处理了 V13__ 前缀的环境变量，我们可以直接查配置
        if self.config_loader:
            try:
                # 尝试从配置中获取端口（配置加载器已将 V13__PORT_* 映射到配置中）
                env_port = self.config_loader.get(f"ports.{name}", None)
                if env_port is not None:
                    port = int(env_port)
                    logger.info(f"Port {name} overridden by config: {port}")
                    return port
            except Exception:
                pass
        
        # 向后兼容：如果没有配置加载器或配置中没有，尝试直接读取环境变量（已废弃）
        env_var = f"V13_PORT_{name.upper()}"
        env_port_val = os.environ.get(env_var)
        if env_port_val:
            try:
                import warnings
                warnings.warn(
                    f"Direct environment variable {env_var} is deprecated. "
                    f"Please use config system or V13__PORT_{name.upper()} environment variable.",
                    DeprecationWarning
                )
                port = int(env_port_val)
                logger.info(f"Port {name} overridden by environment variable {env_var}={port}")
                return port
            except ValueError:
                logger.warning(f"Invalid port value in {env_var}: {env_port_val}")
        
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

# 全局端口管理器实例（延迟初始化，支持配置加载器）
_port_manager_instance: Optional[PortManager] = None

def get_port_manager(config_loader=None) -> PortManager:
    """获取全局端口管理器实例，支持配置加载器注入"""
    global _port_manager_instance
    if _port_manager_instance is None:
        _port_manager_instance = PortManager(config_loader=config_loader)
    return _port_manager_instance

port_manager = get_port_manager()  # 默认实例（向后兼容）

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
