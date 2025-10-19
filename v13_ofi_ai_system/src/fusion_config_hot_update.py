#!/usr/bin/env python3
"""
融合指标配置热更新模块

支持融合指标配置的动态更新，包括：
- 文件监控和自动重载
- 配置验证和回滚
- 原子更新确保一致性
- 支持环境变量覆盖

Author: V13 OFI+CVD AI Trading System
Date: 2025-10-20
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.utils.config_loader import ConfigLoader
from src.ofi_cvd_fusion import OFI_CVD_Fusion, OFICVDFusionConfig

logger = logging.getLogger(__name__)


class FusionConfigHotUpdater:
    """融合指标配置热更新器"""
    
    def __init__(self, config_loader: ConfigLoader, 
                 fusion_instance: Optional[OFI_CVD_Fusion] = None,
                 watch_paths: Optional[list] = None):
        """
        初始化配置热更新器
        
        Args:
            config_loader: 配置加载器实例
            fusion_instance: 融合指标实例，如果为None则创建新实例
            watch_paths: 监控的文件路径列表
        """
        self.config_loader = config_loader
        self.fusion_instance = fusion_instance
        self.watch_paths = watch_paths or [
            "config/system.yaml",
            "config/environments/production.yaml",
            "config/environments/development.yaml",
            "config/environments/testing.yaml"
        ]
        
        # 配置更新回调
        self.update_callbacks: list[Callable] = []
        
        # 文件监控器
        self.observer = Observer()
        self.event_handler = FusionConfigFileHandler(self)
        
        # 配置备份
        self.config_backup: Optional[Dict[str, Any]] = None
        self.last_update_time = 0.0
        
        # 更新统计
        self.update_stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'last_update_time': 0.0,
            'last_error': None
        }
    
    def start_watching(self):
        """开始监控配置文件变化"""
        try:
            for watch_path in self.watch_paths:
                path_obj = Path(watch_path)
                if path_obj.exists():
                    self.observer.schedule(
                        self.event_handler, 
                        str(path_obj.parent), 
                        recursive=False
                    )
                    logger.info(f"开始监控配置文件: {watch_path}")
                else:
                    logger.warning(f"配置文件不存在，跳过监控: {watch_path}")
            
            self.observer.start()
            logger.info("融合指标配置热更新监控已启动")
            
        except Exception as e:
            logger.error(f"启动配置监控失败: {e}")
            raise
    
    def stop_watching(self):
        """停止监控配置文件变化"""
        try:
            self.observer.stop()
            self.observer.join()
            logger.info("融合指标配置热更新监控已停止")
        except Exception as e:
            logger.error(f"停止配置监控失败: {e}")
    
    def add_update_callback(self, callback: Callable):
        """
        添加配置更新回调
        
        Args:
            callback: 配置更新时的回调函数
        """
        self.update_callbacks.append(callback)
    
    def remove_update_callback(self, callback: Callable):
        """
        移除配置更新回调
        
        Args:
            callback: 要移除的回调函数
        """
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
    
    def update_config(self, force: bool = False) -> bool:
        """
        更新融合指标配置
        
        Args:
            force: 是否强制更新（忽略时间间隔）
            
        Returns:
            更新是否成功
        """
        try:
            current_time = time.time()
            
            # 检查更新间隔
            if not force and current_time - self.last_update_time < 1.0:
                logger.debug("配置更新间隔太短，跳过")
                return False
            
            # 备份当前配置
            if self.fusion_instance:
                self.config_backup = {
                    'w_ofi': self.fusion_instance.cfg.w_ofi,
                    'w_cvd': self.fusion_instance.cfg.w_cvd,
                    'fuse_buy': self.fusion_instance.cfg.fuse_buy,
                    'fuse_strong_buy': self.fusion_instance.cfg.fuse_strong_buy,
                    'fuse_sell': self.fusion_instance.cfg.fuse_sell,
                    'fuse_strong_sell': self.fusion_instance.cfg.fuse_strong_sell,
                    'min_consistency': self.fusion_instance.cfg.min_consistency,
                    'strong_min_consistency': self.fusion_instance.cfg.strong_min_consistency,
                    'z_clip': self.fusion_instance.cfg.z_clip,
                    'max_lag': self.fusion_instance.cfg.max_lag,
                    'hysteresis_exit': self.fusion_instance.cfg.hysteresis_exit,
                    'cooldown_secs': self.fusion_instance.cfg.cooldown_secs,
                    'min_consecutive': self.fusion_instance.cfg.min_consecutive,
                    'min_warmup_samples': self.fusion_instance.cfg.min_warmup_samples
                }
            
            # 重新加载配置
            self.config_loader.load(reload=True)
            
            # 创建新的融合指标配置
            new_config = self.fusion_instance._load_from_config_loader(self.config_loader)
            
            # 验证新配置
            if not self._validate_config(new_config):
                logger.error("新配置验证失败，回滚到备份配置")
                self._rollback_config()
                return False
            
            # 更新融合指标实例
            if self.fusion_instance:
                self.fusion_instance.cfg = new_config
                # 重新归一化权重
                total_weight = new_config.w_ofi + new_config.w_cvd
                if total_weight > 0:
                    self.fusion_instance.cfg.w_ofi = new_config.w_ofi / total_weight
                    self.fusion_instance.cfg.w_cvd = new_config.w_cvd / total_weight
            
            # 更新统计
            self.update_stats['total_updates'] += 1
            self.update_stats['successful_updates'] += 1
            self.update_stats['last_update_time'] = current_time
            self.last_update_time = current_time
            
            # 调用更新回调
            for callback in self.update_callbacks:
                try:
                    callback(new_config)
                except Exception as e:
                    logger.error(f"配置更新回调执行失败: {e}")
            
            logger.info("融合指标配置更新成功")
            return True
            
        except Exception as e:
            logger.error(f"融合指标配置更新失败: {e}")
            self.update_stats['failed_updates'] += 1
            self.update_stats['last_error'] = str(e)
            
            # 回滚配置
            self._rollback_config()
            return False
    
    def _validate_config(self, config: OFICVDFusionConfig) -> bool:
        """
        验证配置的有效性
        
        Args:
            config: 要验证的配置
            
        Returns:
            配置是否有效
        """
        try:
            # 检查权重
            if config.w_ofi < 0 or config.w_cvd < 0:
                logger.error("权重不能为负数")
                return False
            
            if config.w_ofi + config.w_cvd <= 0:
                logger.error("权重和必须大于0")
                return False
            
            # 检查阈值
            if config.fuse_strong_buy <= config.fuse_buy:
                logger.error("强买入阈值必须大于买入阈值")
                return False
            
            if config.fuse_strong_sell >= config.fuse_sell:
                logger.error("强卖出阈值必须小于卖出阈值")
                return False
            
            # 检查一致性阈值
            if config.min_consistency < 0 or config.min_consistency > 1:
                logger.error("最小一致性阈值必须在0-1之间")
                return False
            
            if config.strong_min_consistency < 0 or config.strong_min_consistency > 1:
                logger.error("强信号一致性阈值必须在0-1之间")
                return False
            
            if config.strong_min_consistency <= config.min_consistency:
                logger.error("强信号一致性阈值必须大于最小一致性阈值")
                return False
            
            # 检查数据处理参数
            if config.z_clip <= 0:
                logger.error("Z值裁剪阈值必须大于0")
                return False
            
            if config.max_lag < 0:
                logger.error("最大滞后时间不能为负数")
                return False
            
            if config.min_warmup_samples < 0:
                logger.error("暖启动样本数不能为负数")
                return False
            
            # 检查去噪参数
            if config.hysteresis_exit < 0:
                logger.error("迟滞退出阈值不能为负数")
                return False
            
            if config.cooldown_secs < 0:
                logger.error("冷却时间不能为负数")
                return False
            
            if config.min_consecutive < 0:
                logger.error("最小持续次数不能为负数")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"配置验证异常: {e}")
            return False
    
    def _rollback_config(self):
        """回滚到备份配置"""
        try:
            if self.config_backup and self.fusion_instance:
                # 恢复备份配置
                for key, value in self.config_backup.items():
                    setattr(self.fusion_instance.cfg, key, value)
                
                logger.info("已回滚到备份配置")
            else:
                logger.warning("没有可用的备份配置")
                
        except Exception as e:
            logger.error(f"配置回滚失败: {e}")
    
    def get_update_stats(self) -> Dict[str, Any]:
        """获取更新统计信息"""
        return self.update_stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.update_stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'last_update_time': 0.0,
            'last_error': None
        }


class FusionConfigFileHandler(FileSystemEventHandler):
    """融合指标配置文件监控处理器"""
    
    def __init__(self, updater: FusionConfigHotUpdater):
        self.updater = updater
        self.last_modified = 0.0
    
    def on_modified(self, event):
        """文件修改事件处理"""
        if event.is_directory:
            return
        
        # 检查是否是监控的配置文件
        file_path = Path(event.src_path)
        if not any(str(file_path) == watch_path for watch_path in self.updater.watch_paths):
            return
        
        # 防止重复触发
        current_time = time.time()
        if current_time - self.last_modified < 0.5:
            return
        
        self.last_modified = current_time
        
        logger.info(f"检测到配置文件变化: {file_path}")
        
        # 延迟更新，确保文件写入完成
        time.sleep(0.1)
        
        # 执行配置更新
        self.updater.update_config()


def create_fusion_hot_updater(config_loader: Optional[ConfigLoader] = None,
                             fusion_instance: Optional[OFI_CVD_Fusion] = None,
                             watch_paths: Optional[list] = None) -> FusionConfigHotUpdater:
    """
    创建融合指标配置热更新器
    
    Args:
        config_loader: 配置加载器实例
        fusion_instance: 融合指标实例
        watch_paths: 监控的文件路径列表
        
    Returns:
        配置热更新器实例
    """
    if config_loader is None:
        config_loader = ConfigLoader()
    
    return FusionConfigHotUpdater(
        config_loader=config_loader,
        fusion_instance=fusion_instance,
        watch_paths=watch_paths
    )


if __name__ == "__main__":
    # 测试配置热更新
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置加载器
    config_loader = ConfigLoader()
    
    # 创建融合指标实例
    fusion = OFI_CVD_Fusion(config_loader=config_loader)
    
    # 创建热更新器
    hot_updater = create_fusion_hot_updater(
        config_loader=config_loader,
        fusion_instance=fusion
    )
    
    # 添加更新回调
    def on_config_update(new_config):
        print(f"配置已更新: w_ofi={new_config.w_ofi}, w_cvd={new_config.w_cvd}")
    
    hot_updater.add_update_callback(on_config_update)
    
    try:
        # 开始监控
        hot_updater.start_watching()
        
        print("融合指标配置热更新监控已启动，按Ctrl+C停止...")
        
        # 保持运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n正在停止监控...")
        hot_updater.stop_watching()
        print("监控已停止")
