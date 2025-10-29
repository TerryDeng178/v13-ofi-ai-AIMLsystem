"""
运行时配置包加载工具 - 统一组件入口
"""

from typing import Dict, Any, Optional
from pathlib import Path
import os
from .strict_mode import load_strict_runtime_config


def load_component_runtime_config(
    component: str,
    pack_path: Optional[str] = None,
    compat_global: bool = False,
    verify_scenarios_snapshot: bool = True
) -> Dict[str, Any]:
    """
    加载组件的运行时配置包
    
    Args:
        component: 组件名称（'ofi', 'cvd', 'fusion', 'divergence', 'strategy', 'core_algo'）
        pack_path: 显式指定运行时包路径（None时使用默认路径或环境变量）
        compat_global: 是否启用兼容模式（临时过渡选项）
        verify_scenarios_snapshot: 是否验证场景快照指纹（仅strategy需要）
    
    Returns:
        配置字典（仅包含组件和运行时配置，不包含__meta__和__invariants__）
    
    Raises:
        StrictRuntimeConfigError: 如果严格模式被违反或运行时包加载失败
    """
    if compat_global:
        import warnings
        warnings.warn(
            f"兼容模式已启用（{component}）。"
            "此模式将在未来版本中删除，请迁移到严格运行时包模式。",
            DeprecationWarning,
            stacklevel=2
        )
        # 兼容模式：使用旧的全局配置加载
        from .loader import load_config
        cfg, _ = load_config("config")
        return cfg
    
    # 确定运行时包路径
    if pack_path is None:
        # 尝试从环境变量获取
        env_key = f"V13_{component.upper()}_RUNTIME_PACK"
        pack_path = os.getenv(env_key)
        
        if pack_path is None:
            # 使用默认路径（相对于项目根目录）
            script_dir = Path(__file__).resolve().parent.parent
            pack_path = str(script_dir / 'dist' / 'config' / f"{component}.runtime.current.yaml")
    
    # 严格模式加载（仅strategy需要场景快照验证）
    if component == 'strategy':
        verify_scenarios = verify_scenarios_snapshot
    else:
        verify_scenarios = False
    
    return load_strict_runtime_config(
        pack_path,
        compat_global_config=False,  # 严格模式不允许兼容全局配置
        verify_scenarios_snapshot=verify_scenarios
    )


def print_component_effective_config(cfg: Dict[str, Any], component: str, verbose: bool = False):
    """
    打印组件有效配置（脱敏、折叠、尊重要）
    
    Args:
        cfg: 配置字典
        component: 组件名称
        verbose: 是否详细模式（显示完整列表内容和逐键来源）
    """
    from .printer import print_config_tree, print_source_summary
    
    # 提取组件配置
    component_cfg = cfg.get(f"components.{component}", cfg.get("components", {}).get(component, {}))
    
    if not component_cfg:
        print(f"[警告] 未找到组件 '{component}' 的配置")
        return
    
    print(f"\n{'='*80}")
    print(f"组件 '{component}' 有效配置")
    print(f"{'='*80}\n")
    
    # 打印组件配置
    print_config_tree(component_cfg, {}, component, verbose, indent="", prefix=f"components.{component}")
    
    # 打印运行时配置
    runtime_keys = ['logging', 'performance', 'guards', 'output']
    for key in runtime_keys:
        if key in cfg:
            print(f"\n{key}:")
            print_config_tree(cfg[key], {}, component, verbose, indent="  ", prefix=key)
    
    # 打印来源统计（如果有meta信息）
    if '__meta__' in cfg:
        meta = cfg['__meta__']
        print(f"\n来源统计: {meta.get('source_layers', {})}")
        if component == 'strategy':
            sha = cfg.get('scenarios_snapshot_sha256')
            if sha:
                print(f"场景快照指纹: {sha[:8]}...")

