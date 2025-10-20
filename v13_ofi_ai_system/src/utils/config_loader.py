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
        应用环境变量覆盖（支持任意深度）
        
        优先使用双下划线 `__` 作为层级分隔符（推荐）：
            V13__performance__queue__max_size=100000  -> performance.queue.max_size
            V13__logging__level=DEBUG                 -> logging.level
            V13__features__verbose_logging=true       -> features.verbose_logging
        
        兼容旧格式（单下划线）：
            - 前两段作为层级，其余段合并为叶子键（用下划线拼回）
            - PERFORMANCE_QUEUE_MAX_SIZE -> performance.queue.max_size
            - LOGGING_FILE_MAX_SIZE_MB   -> logging.file.max_size_mb
        
        规则：
            - 仅覆盖已存在的配置项（避免误拼写污染配置）
            - 根据参考值类型自动转换
        
        Args:
            config: 配置字典
        
        Returns:
            应用环境变量后的配置字典
        """
        for env_key, env_value in os.environ.items():
            key_lower = env_key.lower()
            
            # 1) 新格式：双下划线分隔（推荐）
            if "__" in env_key:
                # 允许加项目前缀，如 V13__... 或 CFG__...
                parts = [p for p in env_key.split("__") if p]
                # 去掉可选前缀（不区分大小写）
                while parts and parts[0].upper() in ("V13", "CFG", "CONFIG", "OFI", "CVD"):
                    parts.pop(0)
                if not parts:
                    continue
                path = [p.lower() for p in parts]
                self._set_by_path(config, path, env_value)
                continue
            
            # 2) 旧格式：单下划线（向后兼容）
            parts = key_lower.split('_')
            if len(parts) >= 2:
                if len(parts) == 2:
                    # section_key
                    path = [parts[0], parts[1]]
                else:
                    # section_subsection_leaf(with_underscores)
                    # 前两段作为层级，其余合并为叶子键
                    section, subsection = parts[0], parts[1]
                    leaf = '_'.join(parts[2:])
                    path = [section, subsection, leaf]
                self._set_by_path(config, path, env_value)
        
        return config
    
    def _set_by_path(self, cfg: Dict[str, Any], path: list, raw_value: str) -> None:
        """
        按路径设置配置值（只在完整路径存在时才覆盖）
        
        Args:
            cfg: 配置字典
            path: 配置路径（列表形式）
            raw_value: 原始字符串值
        """
        node = cfg
        # 遍历到倒数第二层
        for key in path[:-1]:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                # 路径不存在，跳过（避免创建新键）
                return
        
        # 设置叶子节点
        leaf = path[-1]
        if isinstance(node, dict):
            # 尝试精确匹配
            if leaf in node:
                converted_value = self._convert_type(raw_value, node[leaf])
                node[leaf] = converted_value
                logger.debug(f"Environment override: {'.'.join(path)} = {converted_value}")
            else:
                # 尝试大小写不敏感匹配
                for key in node.keys():
                    if key.lower() == leaf.lower():
                        converted_value = self._convert_type(raw_value, node[key])
                        node[key] = converted_value
                        logger.debug(f"Environment override: {'.'.join(path)} = {converted_value} (matched {key})")
                        break
        # else: 叶子键不存在，跳过
    
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
        
        递归扫描所有包含路径的配置项（以 *_dir, *_path, *_file 结尾的键）
        
        Args:
            config: 配置字典
        
        Returns:
            路径解析后的配置字典
        """
        def resolve_recursive(obj: Any, parent_key: str = '') -> Any:
            """递归解析路径"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # 检查是否是路径相关的键
                    if isinstance(value, str) and (
                        key.endswith('_dir') or 
                        key.endswith('_path') or 
                        key.endswith('_file') or
                        key in ('database', 'filename')  # 特殊情况
                    ):
                        path_obj = Path(value)
                        if not path_obj.is_absolute():
                            obj[key] = str((self.project_root / value).resolve())
                    elif isinstance(value, (dict, list)):
                        obj[key] = resolve_recursive(value, key)
            elif isinstance(obj, list):
                return [resolve_recursive(item, parent_key) for item in obj]
            return obj
        
        # 优先处理 paths 配置节（保持向后兼容）
        if 'paths' in config:
            for key, path in config['paths'].items():
                if isinstance(path, str):
                    path_obj = Path(path)
                    if not path_obj.is_absolute():
                        config['paths'][key] = str((self.project_root / path).resolve())
        
        # 递归处理其他配置节中的路径
        for section_key in config:
            if section_key != 'paths':  # 已经处理过了
                config[section_key] = resolve_recursive(config[section_key], section_key)
        
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
        print(f"\n📋 System: {config['system'].get('name', 'Unknown')} v{config['system'].get('version', 'n/a')}")
        print(f"🌍 Environment: {config['system'].get('environment', 'unknown')}")
        print(f"📁 Data directory: {config.get('paths', {}).get('data_dir', 'N/A')}")
        print(f"🔧 Queue size: {config.get('performance', {}).get('queue', {}).get('max_size', 'N/A')}")
        print(f"📊 Log level: {config.get('logging', {}).get('level', 'N/A')}")
        
        # 测试get方法
        queue_size = get_config('performance.queue.max_size')
        print(f"\n✅ get_config test: queue_size = {queue_size}")
        
        # 测试环境变量覆盖（如果有设置）
        if os.getenv('V13__PERFORMANCE__QUEUE__MAX_SIZE') or os.getenv('PERFORMANCE_QUEUE_MAX_SIZE'):
            print(f"\n🔧 Environment variable override detected:")
            print(f"   V13__PERFORMANCE__QUEUE__MAX_SIZE = {os.getenv('V13__PERFORMANCE__QUEUE__MAX_SIZE')}")
            print(f"   PERFORMANCE_QUEUE_MAX_SIZE = {os.getenv('PERFORMANCE_QUEUE_MAX_SIZE')}")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        import traceback
        traceback.print_exc()

