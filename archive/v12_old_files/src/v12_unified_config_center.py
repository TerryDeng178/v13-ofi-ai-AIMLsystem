"""
V12 统一配置中心
实现配置集中管理和自动分发，解决组件间参数同步问题
"""

import json
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V12UnifiedConfigCenter:
    """V12统一配置中心 - 实现配置集中管理和自动分发"""
    
    def __init__(self, config_file: str = "config/v12_unified_config.json"):
        self.config_file = config_file
        self.config = {}
        self.subscribers = {}
        self.version = 0
        self.lock = threading.Lock()
        self.last_update = None
        
        # 初始化配置
        self._load_initial_config()
        
        logger.info("V12统一配置中心初始化完成")
    
    def _load_initial_config(self):
        """加载初始配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"从文件加载配置: {self.config_file}")
            else:
                # 创建默认配置
                self.config = self._create_default_config()
                self._save_config()
                logger.info("创建默认配置")
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            self.config = self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """创建默认配置"""
        return {
            "ai_models": {
                "lstm": {
                    "input_size": 31,
                    "hidden_size": 128,
                    "num_layers": 3,
                    "dropout": 0.2,
                    "learning_rate": 0.001
                },
                "transformer": {
                    "input_size": 31,
                    "d_model": 128,
                    "nhead": 8,
                    "num_layers": 3,
                    "dropout": 0.2,
                    "learning_rate": 0.001
                },
                "cnn": {
                    "input_size": 31,
                    "sequence_length": 60,
                    "dropout": 0.2,
                    "learning_rate": 0.001
                },
                "ensemble": {
                    "fusion_weights": {
                        "ofi_expert": 0.4,
                        "lstm": 0.25,
                        "transformer": 0.25,
                        "cnn": 0.1
                    }
                }
            },
            "signal_processing": {
                "quality_threshold": 0.35,
                "confidence_threshold": 0.55,
                "strength_threshold": 0.15,
                "max_daily_trades": 50
            },
            "execution": {
                "max_orders_per_second": 100,
                "max_position_size": 100000,
                "slippage_budget": 0.25,
                "commission_bps": 1.0
            },
            "risk_management": {
                "max_drawdown": 0.1,
                "stop_loss_multiplier": 1.1,
                "take_profit_multiplier": 0.9,
                "position_size_limit": 0.05
            },
            "monitoring": {
                "performance_threshold": 0.6,
                "update_frequency": 50,
                "alert_threshold": 0.8
            }
        }
    
    def update_config(self, component: str, new_config: Dict[str, Any]) -> bool:
        """
        更新组件配置并通知订阅者
        
        Args:
            component: 组件名称
            new_config: 新配置
            
        Returns:
            是否更新成功
        """
        try:
            with self.lock:
                # 更新配置
                if component not in self.config:
                    self.config[component] = {}
                
                self.config[component].update(new_config)
                self.version += 1
                self.last_update = datetime.now()
                
                # 保存配置
                self._save_config()
                
                # 通知订阅者
                self._notify_subscribers(component, new_config)
                
                logger.info(f"配置更新成功 - 组件: {component}, 版本: {self.version}")
                return True
                
        except Exception as e:
            logger.error(f"配置更新失败: {e}")
            return False
    
    def get_config(self, component: str = None) -> Dict[str, Any]:
        """
        获取配置
        
        Args:
            component: 组件名称，None表示获取全部配置
            
        Returns:
            配置字典
        """
        try:
            with self.lock:
                if component is None:
                    return self.config.copy()
                else:
                    return self.config.get(component, {}).copy()
        except Exception as e:
            logger.error(f"获取配置失败: {e}")
            return {}
    
    def subscribe(self, component: str, callback: Callable[[str, Dict], None]) -> bool:
        """
        订阅配置变更
        
        Args:
            component: 组件名称
            callback: 回调函数
            
        Returns:
            是否订阅成功
        """
        try:
            if component not in self.subscribers:
                self.subscribers[component] = []
            
            self.subscribers[component].append(callback)
            logger.info(f"订阅成功 - 组件: {component}")
            return True
            
        except Exception as e:
            logger.error(f"订阅失败: {e}")
            return False
    
    def unsubscribe(self, component: str, callback: Callable[[str, Dict], None]) -> bool:
        """
        取消订阅
        
        Args:
            component: 组件名称
            callback: 回调函数
            
        Returns:
            是否取消订阅成功
        """
        try:
            if component in self.subscribers and callback in self.subscribers[component]:
                self.subscribers[component].remove(callback)
                logger.info(f"取消订阅成功 - 组件: {component}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"取消订阅失败: {e}")
            return False
    
    def _notify_subscribers(self, component: str, new_config: Dict[str, Any]):
        """通知订阅者配置变更"""
        try:
            if component in self.subscribers:
                for callback in self.subscribers[component]:
                    try:
                        callback(component, new_config)
                    except Exception as e:
                        logger.error(f"通知订阅者失败: {e}")
                        
        except Exception as e:
            logger.error(f"通知订阅者失败: {e}")
    
    def _save_config(self):
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def get_version(self) -> int:
        """获取当前配置版本"""
        return self.version
    
    def get_last_update(self) -> Optional[datetime]:
        """获取最后更新时间"""
        return self.last_update
    
    def reset_config(self) -> bool:
        """重置配置为默认值"""
        try:
            with self.lock:
                self.config = self._create_default_config()
                self.version += 1
                self.last_update = datetime.now()
                self._save_config()
                
                # 通知所有订阅者
                for component in self.subscribers:
                    self._notify_subscribers(component, self.config.get(component, {}))
                
                logger.info("配置重置成功")
                return True
                
        except Exception as e:
            logger.error(f"配置重置失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'version': self.version,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'components': list(self.config.keys()),
            'subscribers': {k: len(v) for k, v in self.subscribers.items()},
            'config_file': self.config_file
        }


def test_v12_unified_config_center():
    """测试V12统一配置中心"""
    logger.info("开始测试V12统一配置中心...")
    
    # 创建配置中心
    config_center = V12UnifiedConfigCenter()
    
    # 测试配置更新
    ai_config = {
        "lstm": {
            "input_size": 31,
            "hidden_size": 256,
            "learning_rate": 0.002
        }
    }
    
    success = config_center.update_config("ai_models", ai_config)
    logger.info(f"配置更新结果: {success}")
    
    # 测试配置获取
    config = config_center.get_config("ai_models")
    logger.info(f"获取配置: {config}")
    
    # 测试订阅机制
    def config_callback(component: str, new_config: Dict):
        logger.info(f"收到配置变更通知 - 组件: {component}, 配置: {new_config}")
    
    config_center.subscribe("ai_models", config_callback)
    
    # 测试配置变更通知
    new_config = {"lstm": {"learning_rate": 0.003}}
    config_center.update_config("ai_models", new_config)
    
    # 获取统计信息
    stats = config_center.get_statistics()
    logger.info(f"配置中心统计信息: {stats}")
    
    logger.info("V12统一配置中心测试完成")


if __name__ == "__main__":
    test_v12_unified_config_center()
