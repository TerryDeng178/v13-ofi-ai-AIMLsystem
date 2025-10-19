#!/usr/bin/env python3
"""
参数固化与热更新脚本
支持配置文件热更新和条件参数选择
"""

import argparse
import json
import sys
import time
import threading
import io
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

# 修复Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))


class ConfigHotReloader:
    """配置热更新器"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = None
        self.version = 0
        self.lock = threading.Lock()
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 加载初始配置
        self.load_config()
    
    def load_config(self) -> bool:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            self.version += 1
            self.logger.info(f"✅ 配置已加载 (版本: {self.version})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 配置加载失败: {e}")
            return False
    
    def select_params(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """根据上下文选择参数"""
        if not self.config:
            return {}
        
        with self.lock:
            # 从背离检测配置的默认参数开始
            divergence_config = self.config.get('divergence_detection', {})
            params = divergence_config.get('default', {}).copy()
            
            # 应用覆盖规则
            overrides = divergence_config.get('overrides', [])
            for rule in overrides:
                when_conditions = rule.get('when', {})
                set_params = rule.get('set', {})
                
                # 检查是否匹配条件
                if self._matches_conditions(context, when_conditions):
                    params.update(set_params)
                    self.logger.debug(f"应用覆盖规则: {when_conditions} -> {set_params}")
            
            return params
    
    def _matches_conditions(self, context: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
        """检查上下文是否匹配条件"""
        for key, expected_value in conditions.items():
            if key not in context:
                return False
            if context[key] != expected_value:
                return False
        return True
    
    def get_calibration(self) -> Optional[Dict[str, Any]]:
        """获取校准配置"""
        if not self.config:
            return None
        
        divergence_config = self.config.get('divergence_detection', {})
        calibration_file = divergence_config.get('calibration', {}).get('file')
        if not calibration_file:
            return None
        
        try:
            calibration_path = Path(calibration_file)
            if calibration_path.exists():
                with open(calibration_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"❌ 校准配置加载失败: {e}")
        
        return None
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取监控配置"""
        if not self.config:
            return {}
        
        divergence_config = self.config.get('divergence_detection', {})
        return divergence_config.get('monitoring', {})
    
    def start_watch(self):
        """开始监控配置文件变化"""
        if not self.config_path.exists():
            self.logger.error(f"❌ 配置文件不存在: {self.config_path}")
            return
        
        event_handler = ConfigFileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.config_path.parent), recursive=False)
        observer.start()
        
        self.logger.info(f"👀 开始监控配置文件: {self.config_path}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            self.logger.info("🛑 配置监控已停止")
        
        observer.join()


class ConfigFileHandler(FileSystemEventHandler):
    """配置文件变化处理器"""
    
    def __init__(self, reloader: ConfigHotReloader):
        self.reloader = reloader
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if Path(event.src_path) == self.reloader.config_path:
            self.reloader.logger.info("📝 检测到配置文件变化，重新加载...")
            time.sleep(0.5)  # 等待文件写入完成
            self.reloader.load_config()


class DivergenceConfigManager:
    """背离检测配置管理器"""
    
    def __init__(self, config_path: str):
        self.reloader = ConfigHotReloader(config_path)
        self.current_context = {}
    
    def set_context(self, **kwargs):
        """设置当前上下文"""
        self.current_context.update(kwargs)
    
    def get_params(self) -> Dict[str, Any]:
        """获取当前参数"""
        return self.reloader.select_params(self.current_context)
    
    def get_calibration(self) -> Optional[Dict[str, Any]]:
        """获取校准配置"""
        return self.reloader.get_calibration()
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取监控配置"""
        return self.reloader.get_monitoring_config()
    
    def start_hot_reload(self):
        """启动热更新"""
        self.reloader.start_watch()


def create_config_loader():
    """创建配置加载器（供其他模块使用）"""
    config_path = Path(__file__).parent.parent / "config" / "system.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    return DivergenceConfigManager(str(config_path))


def test_config_selection():
    """测试配置选择功能"""
    print("🧪 测试配置选择功能...")
    
    config_manager = create_config_loader()
    
    # 测试不同上下文
    test_cases = [
        {
            'name': '默认配置',
            'context': {}
        },
        {
            'name': 'OFI清淡时段',
            'context': {'source': 'OFI_ONLY', 'liquidity': 'quiet'}
        },
        {
            'name': 'CVD活跃时段',
            'context': {'source': 'CVD_ONLY', 'liquidity': 'active'}
        },
        {
            'name': '融合白天时段',
            'context': {'source': 'FUSION', 'session': 'day'}
        },
        {
            'name': '夜间时段',
            'context': {'session': 'night'}
        },
        {
            'name': 'BTCUSDT高频',
            'context': {'symbol': 'BTCUSDT'}
        }
    ]
    
    for test_case in test_cases:
        config_manager.set_context(**test_case['context'])
        params = config_manager.get_params()
        
        print(f"\n📋 {test_case['name']}:")
        print(f"   上下文: {test_case['context']}")
        print(f"   参数: {params}")


def test_calibration():
    """测试校准配置"""
    print("\n🎯 测试校准配置...")
    
    config_manager = create_config_loader()
    calibration = config_manager.get_calibration()
    
    if calibration:
        print("✅ 校准配置加载成功")
        print(f"   版本: {calibration.get('version', 'unknown')}")
        print(f"   描述: {calibration.get('description', 'unknown')}")
    else:
        print("⚠️ 校准配置未找到")


def test_monitoring():
    """测试监控配置"""
    print("\n📊 测试监控配置...")
    
    config_manager = create_config_loader()
    monitoring = config_manager.get_monitoring_config()
    
    if monitoring:
        print("✅ 监控配置加载成功")
        print(f"   Prometheus端口: {monitoring.get('prometheus', {}).get('port', 'unknown')}")
        print(f"   Grafana仪表盘: {monitoring.get('grafana', {}).get('dashboard_uid', 'unknown')}")
    else:
        print("⚠️ 监控配置未找到")


def main():
    parser = argparse.ArgumentParser(description='配置热更新工具')
    parser.add_argument('--config', default='config/system.yaml', help='配置文件路径')
    parser.add_argument('--test', action='store_true', help='运行测试')
    parser.add_argument('--watch', action='store_true', help='启动文件监控')
    
    args = parser.parse_args()
    
    if args.test:
        test_config_selection()
        test_calibration()
        test_monitoring()
    elif args.watch:
        config_manager = create_config_loader()
        config_manager.start_hot_reload()
    else:
        print("请指定 --test 或 --watch 参数")


if __name__ == "__main__":
    main()
