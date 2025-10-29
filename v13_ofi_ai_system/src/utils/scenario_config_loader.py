#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景配置热加载管理器
负责监控配置文件变化，自动热加载场景参数配置

功能：
1. 文件监控和自动重载
2. 配置版本管理
3. 回滚机制
4. 健康检查
"""

import os
import sys
import time
import json
import yaml
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib

logger = logging.getLogger(__name__)

class ScenarioConfigFileHandler(FileSystemEventHandler):
    """配置文件监控处理器"""
    
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.last_modified = 0
        
    def on_modified(self, event):
        """文件修改事件处理"""
        if event.is_directory:
            return
            
        config_path = Path(event.src_path)
        if config_path.suffix.lower() not in {'.yaml', '.yml', '.json'}:
            return
            
        # 防抖：避免短时间内多次触发
        current_time = time.time()
        if current_time - self.last_modified < 2.0:
            return
        self.last_modified = current_time
        
        logger.info(f"检测到配置文件变化: {config_path}")
        
        # 异步重新加载配置
        threading.Thread(
            target=self.config_loader.reload_config,
            args=(str(config_path),),
            daemon=True
        ).start()

class ScenarioConfigLoader:
    """场景配置热加载器"""
    
    def __init__(self, config_path: str, strategy_manager, 
                 backup_dir: str = "config_backups", 
                 auto_reload: bool = True):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
            strategy_manager: StrategyModeManager实例
            backup_dir: 备份目录
            auto_reload: 是否自动重载
        """
        self.config_path = Path(config_path)
        self.strategy_manager = strategy_manager
        self.backup_dir = Path(backup_dir)
        self.auto_reload = auto_reload
        
        # 配置历史管理
        self.config_history = []
        self.max_history = 10
        
        # 文件监控
        self.observer = None
        self.file_handler = None
        
        # 状态管理
        self.last_config_hash = None
        self.reload_lock = threading.RLock()
        
        # 初始化
        self._setup_backup_dir()
        self._load_initial_config()
        
        if self.auto_reload:
            self._start_file_monitoring()
    
    def _setup_backup_dir(self):
        """设置备份目录"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"配置备份目录: {self.backup_dir}")
    
    def _load_initial_config(self):
        """加载初始配置"""
        if self.config_path.exists():
            success = self.reload_config(str(self.config_path))
            if success:
                logger.info("✅ 初始配置加载成功")
            else:
                logger.error("❌ 初始配置加载失败")
        else:
            logger.warning(f"配置文件不存在: {self.config_path}")
    
    def _start_file_monitoring(self):
        """启动文件监控"""
        try:
            self.observer = Observer()
            self.file_handler = ScenarioConfigFileHandler(self)
            
            # 监控配置文件所在目录
            watch_dir = self.config_path.parent
            self.observer.schedule(self.file_handler, str(watch_dir), recursive=False)
            self.observer.start()
            
            logger.info(f"✅ 启动文件监控: {watch_dir}")
            
        except Exception as e:
            logger.error(f"❌ 启动文件监控失败: {e}")
    
    def _calculate_config_hash(self, config_data: Dict[str, Any]) -> str:
        """计算配置数据哈希"""
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _backup_config(self, config_data: Dict[str, Any]) -> str:
        """备份配置"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_dir / f"scenario_config_{timestamp}.yaml"
        
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✅ 配置已备份: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"❌ 配置备份失败: {e}")
            return None
    
    def _validate_config(self, config_data: Dict[str, Any]) -> bool:
        """验证配置数据"""
        try:
            # 检查必需字段
            required_fields = ['signal_kind', 'scenarios']
            for field in required_fields:
                if field not in config_data:
                    logger.error(f"配置缺少必需字段: {field}")
                    return False
            
            # 检查场景数据
            scenarios = config_data.get('scenarios', {})
            valid_scenarios = {'A_H', 'A_L', 'Q_H', 'Q_L'}
            
            if not scenarios:
                logger.error("场景配置为空")
                return False
            
            for scenario in scenarios:
                if scenario not in valid_scenarios:
                    logger.error(f"无效场景: {scenario}")
                    return False
                
                scenario_params = scenarios[scenario]
                required_params = ['Z_HI_LONG', 'Z_HI_SHORT', 'TP_BPS', 'SL_BPS']
                
                for param in required_params:
                    if param not in scenario_params:
                        logger.error(f"场景 {scenario} 缺少参数: {param}")
                        return False
            
            logger.info("✅ 配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def reload_config(self, config_path: Optional[str] = None) -> bool:
        """
        重新加载配置
        
        Args:
            config_path: 配置文件路径，None时使用默认路径
            
        Returns:
            bool: 是否加载成功
        """
        with self.reload_lock:
            try:
                path = Path(config_path) if config_path else self.config_path
                
                if not path.exists():
                    logger.error(f"配置文件不存在: {path}")
                    return False
                
                # 读取配置文件
                if path.suffix.lower() in {'.yaml', '.yml'}:
                    with open(path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                
                # 计算配置哈希
                config_hash = self._calculate_config_hash(config_data)
                
                # 检查是否与上次配置相同
                if config_hash == self.last_config_hash:
                    logger.debug("配置未变化，跳过重载")
                    return True
                
                # 验证配置
                if not self._validate_config(config_data):
                    logger.error("配置验证失败")
                    return False
                
                # 备份当前配置
                if self.strategy_manager.current_params_by_scenario:
                    backup_file = self._backup_config({
                        'version': self.strategy_manager.scenario_config_version,
                        'scenarios': self.strategy_manager.current_params_by_scenario
                    })
                
                # 加载新配置到策略管理器
                success = self.strategy_manager.load_scenario_params(config_data)
                
                if success:
                    # 更新配置历史
                    self.config_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'version': config_data.get('version', 'unknown'),
                        'config_hash': config_hash,
                        'backup_file': backup_file,
                        'success': True
                    })
                    
                    # 保持历史记录数量
                    if len(self.config_history) > self.max_history:
                        self.config_history = self.config_history[-self.max_history:]
                    
                    self.last_config_hash = config_hash
                    
                    logger.info(f"✅ 配置重载成功 v{config_data.get('version', 'unknown')}")
                    return True
                else:
                    logger.error("❌ 策略管理器加载配置失败")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ 配置重载失败: {e}")
                return False
    
    def rollback_config(self, steps: int = 1) -> bool:
        """
        回滚配置
        
        Args:
            steps: 回滚步数
            
        Returns:
            bool: 是否回滚成功
        """
        with self.reload_lock:
            try:
                if not self.config_history or len(self.config_history) < steps:
                    logger.error("没有可回滚的配置历史")
                    return False
                
                # 获取回滚目标配置
                target_config = self.config_history[-steps-1]
                
                if not target_config['success']:
                    logger.error("目标配置加载失败，无法回滚")
                    return False
                
                backup_file = target_config.get('backup_file')
                if not backup_file or not Path(backup_file).exists():
                    logger.error("备份文件不存在，无法回滚")
                    return False
                
                # 加载备份配置
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_config = yaml.safe_load(f)
                
                success = self.strategy_manager.load_scenario_params(backup_config)
                
                if success:
                    logger.info(f"✅ 配置回滚成功到 v{target_config['version']}")
                    return True
                else:
                    logger.error("❌ 配置回滚失败")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ 配置回滚失败: {e}")
                return False
    
    def get_config_status(self) -> Dict[str, Any]:
        """获取配置状态"""
        return {
            'config_path': str(self.config_path),
            'auto_reload': self.auto_reload,
            'last_config_hash': self.last_config_hash,
            'config_history_count': len(self.config_history),
            'backup_dir': str(self.backup_dir),
            'strategy_manager_version': self.strategy_manager.scenario_config_version,
            'available_scenarios': list(self.strategy_manager.current_params_by_scenario.keys())
        }
    
    def stop_monitoring(self):
        """停止文件监控"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("✅ 文件监控已停止")
    
    def __del__(self):
        """析构函数"""
        self.stop_monitoring()

# 示例使用
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 模拟策略管理器
    class MockStrategyManager:
        def __init__(self):
            self.current_params_by_scenario = {}
            self.scenario_config_version = 'unknown'
        
        def load_scenario_params(self, config_data):
            self.current_params_by_scenario = config_data.get('scenarios', {})
            self.scenario_config_version = config_data.get('version', 'unknown')
            return True
    
    # 创建测试配置
    test_config = {
        'signal_kind': 'fusion',
        'horizon_s': 300,
        'cost_bps': 3,
        'version': 'test_v1',
        'scenarios': {
            'A_H': {'Z_HI_LONG': 2.75, 'Z_HI_SHORT': 2.50, 'Z_MID': 0.75, 'TP_BPS': 15, 'SL_BPS': 10},
            'A_L': {'Z_HI_LONG': 2.25, 'Z_HI_SHORT': 2.25, 'Z_MID': 0.60, 'TP_BPS': 12, 'SL_BPS': 9}
        }
    }
    
    # 保存测试配置
    test_config_path = "test_scenario_config.yaml"
    with open(test_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(test_config, f, default_flow_style=False, allow_unicode=True)
    
    # 创建配置加载器
    strategy_manager = MockStrategyManager()
    config_loader = ScenarioConfigLoader(
        config_path=test_config_path,
        strategy_manager=strategy_manager,
        auto_reload=False
    )
    
    # 测试配置加载
    success = config_loader.reload_config()
    print(f"配置加载: {'成功' if success else '失败'}")
    
    # 获取状态
    status = config_loader.get_config_status()
    print(f"配置状态: {status}")
    
    # 清理测试文件
    import os
    os.remove(test_config_path)
