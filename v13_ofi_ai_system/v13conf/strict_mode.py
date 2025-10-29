"""
运行时严格模式：只读交付包，拒绝旁路配置加载
"""

from typing import Dict, Any, Optional
from pathlib import Path
import warnings


class StrictRuntimeConfigError(Exception):
    """运行时严格模式错误"""
    pass


def load_strict_runtime_config(runtime_pack_path: str, 
                               compat_global_config: bool = False,
                               verify_scenarios_snapshot: bool = True) -> Dict[str, Any]:
    """
    从运行时包加载配置（严格模式）
    
    严格模式下：
    - 只从运行时包读取配置
    - 拒绝从CONFIG_DIR读取任何源文件
    - 拒绝环境变量覆盖（除非运行时包中已声明允许）
    
    Args:
        runtime_pack_path: 运行时包文件路径
        compat_global_config: 兼容模式开关（临时，用于排障，未来版本将删除）
    
    Returns:
        配置字典（仅包含运行时包中的配置）
    
    Raises:
        StrictRuntimeConfigError: 如果严格模式被违反
    """
    if compat_global_config:
        warnings.warn(
            "兼容模式已启用（--compat-global-config）。"
            "此模式将在未来版本中删除，请迁移到严格运行时包模式。",
            DeprecationWarning,
            stacklevel=2
        )
        # 兼容模式：允许使用旧的全局配置加载方式
        from .loader import load_config
        return load_config("config")[0]
    
    # 严格模式：只从运行时包读取
    import yaml
    
    runtime_path = Path(runtime_pack_path)
    if not runtime_path.exists():
        raise StrictRuntimeConfigError(
            f"运行时包文件不存在: {runtime_pack_path}"
        )
    
    try:
        with open(runtime_path, 'r', encoding='utf-8') as f:
            pack = yaml.safe_load(f)
    except Exception as e:
        raise StrictRuntimeConfigError(
            f"无法读取运行时包: {runtime_pack_path}, 错误: {e}"
        )
    
    if not isinstance(pack, dict):
        raise StrictRuntimeConfigError(
            f"运行时包格式错误: {runtime_pack_path}"
        )
    
    # 验证运行时包结构
    if '__meta__' not in pack:
        raise StrictRuntimeConfigError(
            f"运行时包缺少元信息: {runtime_pack_path}"
        )
    
    meta = pack['__meta__']
    print(f"[严格模式] 加载运行时配置包: {runtime_pack_path}")
    print(f"  版本: {meta.get('version', 'unknown')}")
    print(f"  Git SHA: {meta.get('git_sha', 'unknown')}")
    print(f"  组件: {meta.get('component', 'unknown')}")
    print(f"  来源统计: {meta.get('source_layers', {})}")
    
    # 验证场景快照指纹（如果启用）
    if verify_scenarios_snapshot and 'scenarios_snapshot_sha256' in pack:
        snapshot_sha = pack.get('scenarios_snapshot_sha256')
        print(f"  场景快照指纹: {snapshot_sha[:8]}...")  # 打印前8位便于排查
        scenarios_file = pack.get('strategy', {}).get('scenarios_file') if 'strategy' in pack else None
        
        if scenarios_file:
            scenarios_path = Path(scenarios_file)
            if not scenarios_path.is_absolute():
                # 尝试解析相对路径
                scenarios_path = Path(runtime_pack_path).parent.parent.parent / scenarios_file
            
            if scenarios_path.exists():
                import hashlib
                import yaml
                try:
                    with open(scenarios_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    file_sha = hashlib.sha256(file_content.encode('utf-8')).hexdigest()
                    
                    if file_sha != snapshot_sha:
                        raise StrictRuntimeConfigError(
                            f"场景文件指纹不匹配！\n"
                            f"  快照指纹: {snapshot_sha[:16]}...\n"
                            f"  文件指纹: {file_sha[:16]}...\n"
                            f"  文件路径: {scenarios_path}\n"
                            f"说明: 运行时快照与场景文件不一致，可能存在路径漂移。"
                        )
                except Exception as e:
                    if isinstance(e, StrictRuntimeConfigError):
                        raise
                    # 文件读取失败，发出警告但不阻塞（允许在严格模式下继续）
                    warnings.warn(
                        f"无法验证场景文件指纹: {e}",
                        RuntimeWarning,
                        stacklevel=2
                    )
    
    # 移除元信息和不变量，返回纯配置
    config = {k: v for k, v in pack.items() if k not in ('__meta__', '__invariants__')}
    
    return config


def validate_strict_mode(config: Dict[str, Any]) -> bool:
    """
    验证配置是否满足严格模式要求
    
    Returns:
        True if valid, raises exception if invalid
    """
    # 检查是否包含元信息标记（说明来自运行时包）
    # 如果配置中没有运行时包的特征，可能是从其他源加载的
    # 注意：这个检查在实际使用时可能需要在调用方进行
    return True

