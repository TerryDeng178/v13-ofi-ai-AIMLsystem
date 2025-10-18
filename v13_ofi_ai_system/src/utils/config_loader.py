"""
统一系统配置加载器

提供加载和管理系统配置的功能：
- 加载 system.yaml 主配置
- 根据环境加载环境特定配置
- 支持环境变量覆盖
- 配置验证和类型转换
- 路径自动转换为绝对路径

使用示例:
    from src.utils.config_loader import load_config, get_config
    
    # 加载配置
    config = load_config()
    
    # 获取特定配置项
    queue_size = config['performance']['queue']['max_size']
    
    # 或使用便捷方法
    queue_size = get_config('performance.queue.max_size', default=50000)

作者: V13 Team
创建日期: 2025-10-19
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """系统配置加载器"""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录，默认为项目根目录下的config
        """
        if config_dir is None:
            # 自动检测项目根目录
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent  # src/utils -> src -> v13_ofi_ai_system
            self.config_dir = project_root / "config"
        else:
            self.config_dir = Path(config_dir)
        
        self.project_root = self.config_dir.parent
        self._config: Optional[Dict[str, Any]] = None
    
    def load(self, reload: bool = False) -> Dict[str, Any]:
        """
        加载配置
        
        Args:
            reload: 是否强制重新加载配置
        
        Returns:
            配置字典
        """
        if self._config is not None and not reload:
            return self._config
        
        # 1. 加载主配置文件
        system_config = self._load_yaml_file(self.config_dir / "system.yaml")
        
        # 2. 加载环境特定配置
        env = os.getenv("ENV", system_config.get("system", {}).get("environment", "development"))
        env_config = self._load_environment_config(env)
        
        # 3. 合并配置
        config = self._deep_merge(system_config, env_config)
        
        # 4. 应用环境变量覆盖
        config = self._apply_env_overrides(config)
        
        # 5. 转换相对路径为绝对路径
        config = self._resolve_paths(config)
        
        # 6. 验证配置
        self._validate_config(config)
        
        self._config = config
        logger.info(f"Configuration loaded successfully (environment: {env})")
        
        return config
    
    def _load_yaml_file(self, filepath: Path) -> Dict[str, Any]:
        """
        加载YAML文件
        
        Args:
            filepath: YAML文件路径
        
        Returns:
            解析后的字典
        """
        if not filepath.exists():
            logger.warning(f"Configuration file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config if config is not None else {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration file {filepath}: {e}")
            raise
    
    def _load_environment_config(self, env: str) -> Dict[str, Any]:
        """
        加载环境特定配置
        
        Args:
            env: 环境名称 (development/testing/production)
        
        Returns:
            环境配置字典
        """
        env_file = self.config_dir / "environments" / f"{env}.yaml"
        return self._load_yaml_file(env_file)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典
        
        Args:
            base: 基础字典
            override: 覆盖字典
        
        Returns:
            合并后的字典
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用环境变量覆盖
        
        环境变量格式: SECTION_KEY (大写，用下划线连接)
        例如: PERFORMANCE_QUEUE_MAX_SIZE=10000
        
        Args:
            config: 配置字典
        
        Returns:
            应用环境变量后的配置字典
        """
        # 简单实现：支持两级配置
        # 例如: PERFORMANCE_QUEUE_MAX_SIZE -> config['performance']['queue']['max_size']
        
        for env_key, env_value in os.environ.items():
            # 尝试解析为配置路径
            parts = env_key.lower().split('_')
            if len(parts) >= 2:
                try:
                    # 尝试两级结构: section_key
                    if len(parts) == 2:
                        section, key = parts
                        if section in config and key in config[section]:
                            config[section][key] = self._convert_type(env_value, config[section][key])
                            logger.debug(f"Environment override: {env_key} = {env_value}")
                    
                    # 尝试三级结构: section_subsection_key
                    elif len(parts) == 3:
                        section, subsection, key = parts
                        if (section in config and 
                            isinstance(config[section], dict) and
                            subsection in config[section] and
                            key in config[section][subsection]):
                            config[section][subsection][key] = self._convert_type(
                                env_value, config[section][subsection][key]
                            )
                            logger.debug(f"Environment override: {env_key} = {env_value}")
                except Exception as e:
                    logger.debug(f"Could not apply environment override {env_key}: {e}")
        
        return config
    
    def _convert_type(self, value: str, reference: Any) -> Any:
        """
        根据参考值的类型转换字符串值
        
        Args:
            value: 字符串值
            reference: 参考值（用于推断类型）
        
        Returns:
            转换后的值
        """
        if isinstance(reference, bool):
            return value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(reference, int):
            return int(value)
        elif isinstance(reference, float):
            return float(value)
        else:
            return value
    
    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析配置中的相对路径为绝对路径
        
        Args:
            config: 配置字典
        
        Returns:
            路径解析后的配置字典
        """
        if 'paths' in config:
            for key, path in config['paths'].items():
                if isinstance(path, str):
                    path_obj = Path(path)
                    if not path_obj.is_absolute():
                        # 相对路径转换为相对于项目根目录的绝对路径
                        config['paths'][key] = str((self.project_root / path).resolve())
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证配置的完整性
        
        Args:
            config: 配置字典
        
        Raises:
            ValueError: 如果配置不完整或无效
        """
        # 检查必需的顶级配置项
        required_sections = ['system', 'data_source', 'components', 'paths', 'performance', 'logging']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # 检查系统配置
        if 'name' not in config['system']:
            raise ValueError("Missing required system.name")
        
        # 检查路径配置
        required_paths = ['data_dir', 'logs_dir', 'reports_dir']
        for path_key in required_paths:
            if path_key not in config['paths']:
                raise ValueError(f"Missing required path configuration: {path_key}")
        
        logger.debug("Configuration validation passed")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号路径）
        
        Args:
            key_path: 配置键路径，如 'performance.queue.max_size'
            default: 默认值
        
        Returns:
            配置值
        """
        if self._config is None:
            self.load()
        
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value


# 全局配置加载器实例
_global_loader: Optional[ConfigLoader] = None


def load_config(config_dir: Optional[Union[str, Path]] = None, reload: bool = False) -> Dict[str, Any]:
    """
    加载系统配置
    
    Args:
        config_dir: 配置文件目录
        reload: 是否强制重新加载
    
    Returns:
        配置字典
    """
    global _global_loader
    
    if _global_loader is None or reload:
        _global_loader = ConfigLoader(config_dir)
    
    return _global_loader.load(reload)


def get_config(key_path: str, default: Any = None) -> Any:
    """
    获取配置值（便捷方法）
    
    Args:
        key_path: 配置键路径，如 'performance.queue.max_size'
        default: 默认值
    
    Returns:
        配置值
    """
    global _global_loader
    
    if _global_loader is None:
        _global_loader = ConfigLoader()
        _global_loader.load()
    
    return _global_loader.get(key_path, default)


def reload_config() -> Dict[str, Any]:
    """
    重新加载配置
    
    Returns:
        配置字典
    """
    return load_config(reload=True)


if __name__ == "__main__":
    # Windows UTF-8 兼容性
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # 测试配置加载器
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        config = load_config()
        print("✅ Configuration loaded successfully!")
        print(f"\n📋 System: {config['system']['name']} v{config['system']['version']}")
        print(f"🌍 Environment: {config['system']['environment']}")
        print(f"📁 Data directory: {config['paths']['data_dir']}")
        print(f"🔧 Queue size: {config['performance']['queue']['max_size']}")
        print(f"📊 Log level: {config['logging']['level']}")
        
        # 测试get方法
        queue_size = get_config('performance.queue.max_size')
        print(f"\n✅ get_config test: queue_size = {queue_size}")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        import traceback
        traceback.print_exc()

