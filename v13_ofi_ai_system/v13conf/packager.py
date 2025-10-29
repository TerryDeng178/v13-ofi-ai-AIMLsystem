"""
交付包打包器：从统一配置中提取组件子树并生成运行时配置包
"""

import hashlib
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import subprocess
import copy
import hashlib
import os
import platform
import sys
try:
    import getpass
except ImportError:
    getpass = None


def _get_git_sha() -> str:
    """获取当前Git SHA（强制8位十六进制）"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short=8', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            check=True  # 确保命令成功
        )
        git_sha = result.stdout.strip()
        # 验证格式：必须是8位十六进制
        import re
        if not re.match(r'^[0-9a-f]{8}$', git_sha):
            raise ValueError(f"Git SHA格式无效: {git_sha} (必须是8位十六进制)")
        return git_sha
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"获取Git SHA失败: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Git SHA验证失败: {e}") from e


def _calculate_checksum(data: Dict[str, Any]) -> str:
    """计算配置的校验和"""
    content = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def _extract_component_config(cfg: Dict[str, Any], component: str, base_config_dir: Optional[str] = None) -> Dict[str, Any]:
    """从合并配置中提取组件配置子树"""
    component_map = {
        'ofi': ('components', 'ofi'),
        'cvd': ('components', 'cvd'),
        'fusion': ('components', 'fusion'),
        'divergence': ('components', 'divergence'),
        'strategy': ('components', 'strategy'),
        'core_algo': ('components', 'core_algo'),
        'harvester': ('components', 'harvester'),
    }
    
    if component not in component_map:
        raise ValueError(f"Unknown component: {component}")
    
    # 提取组件配置
    path_parts = component_map[component]
    component_cfg = cfg
    for part in path_parts:
        if isinstance(component_cfg, dict) and part in component_cfg:
            component_cfg = component_cfg[part]
        else:
            # 组件配置不存在，返回空字典
            return {}
    
    # 提取必要的运行时配置（所有组件都可能需要的）
    runtime_base = {}
    
    # 添加logging配置（如果存在）
    if 'logging' in cfg:
        runtime_base['logging'] = cfg['logging']
    
    # 添加performance配置（如果存在）
    if 'performance' in cfg:
        runtime_base['performance'] = cfg['performance']
    
    # 对于core_algo，还需要添加guards和output
    if component == 'core_algo':
        for key in ['guards', 'output']:
            if key in cfg:
                runtime_base[key] = cfg[key]
    
    # 对于strategy，检查并包含场景文件快照（如果存在）
    if component == 'strategy':
        strategy = component_cfg
        scenarios_file = strategy.get('scenarios_file') if isinstance(strategy, dict) else None
        if scenarios_file:
            scenarios_path = Path(scenarios_file)
            # 如果是相对路径，尝试从配置目录解析
            if not scenarios_path.is_absolute():
                if base_config_dir:
                    base_path = Path(base_config_dir)
                else:
                    base_path = Path('config')  # 默认值
                scenarios_path = base_path.parent / scenarios_file
            
            if scenarios_path.exists():
                try:
                    import yaml
                    with open(scenarios_path, 'r', encoding='utf-8') as f:
                        scenarios_content = f.read()
                        scenarios_snapshot = yaml.safe_load(scenarios_content)
                    # 计算场景文件的SHA256指纹
                    scenarios_sha256 = hashlib.sha256(scenarios_content.encode('utf-8')).hexdigest()
                    runtime_base['scenarios_snapshot'] = scenarios_snapshot
                    runtime_base['scenarios_snapshot_sha256'] = scenarios_sha256
                except Exception as e:
                    # 场景文件读取失败，记录但不阻塞构建
                    runtime_base['scenarios_snapshot'] = None
                    runtime_base['scenarios_error'] = str(e)
    
    # 合并组件配置和运行时基础配置
    # 将组件配置作为顶层键，例如 fusion: {...}
    result = {
        component: component_cfg,
        **runtime_base
    }
    
    # 如果没有提取到组件配置，至少返回基础运行时配置
    if not component_cfg:
        result = runtime_base
    
    return result


def _count_source_layers(sources: Dict[str, str], component: str) -> Dict[str, int]:
    """统计每个来源层的键数量"""
    counts = {'defaults': 0, 'system': 0, 'overrides': 0, 'env': 0, 'locked': 0}
    
    component_prefix = f"components.{component}."
    
    for key, source in sources.items():
        if key.startswith(component_prefix) or key in ['logging', 'performance', 'guards', 'output']:
            if source in counts:
                counts[source] += 1
    
    return counts


def build_runtime_pack(cfg: Dict[str, Any], component: str, 
                      sources: Dict[str, str],
                      version: str = "1.0.0",
                      check_unconsumed: bool = True,
                      fail_on_unconsumed: bool = False,
                      base_config_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    构建运行时配置包
    
    Args:
        cfg: 合并后的完整配置
        component: 组件名称 (ofi/cvd/fusion/divergence/strategy/core_algo)
        sources: 来源追踪字典
        version: 版本号
    
    Returns:
        包含 __meta__ 和 __invariants__ 的运行时配置包
    """
    # 提取组件配置
    component_cfg = _extract_component_config(cfg, component, base_config_dir)
    
    # 构建元信息
    git_sha = _get_git_sha()
    build_ts = datetime.utcnow().isoformat() + 'Z'
    
    source_counts = _count_source_layers(sources, component)
    
    # 计算校验和
    checksum = _calculate_checksum(component_cfg)
    
    # 获取构建环境信息
    if getpass:
        build_user = os.getenv('USER', os.getenv('USERNAME', getpass.getuser()))
    else:
        build_user = os.getenv('USER', os.getenv('USERNAME', 'unknown'))
    build_host = platform.node()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # 构建元数据
    meta = {
        'version': version,
        'git_sha': git_sha,
        'build_ts': build_ts,
        'component': component,
        'source_layers': source_counts,
        'checksum': checksum,
        'build_user': build_user,
        'build_host': build_host,
        'python_version': python_version,
    }
    
    # 构建不变量校验摘要
    from .invariants import validate_invariants
    from .unconsumed_keys import check_unconsumed_keys
    
    errors = validate_invariants(cfg, component)
    
    # 检查未消费键
    unconsumed = []
    if check_unconsumed:
        # 检查提取出的组件配置中的未消费键
        unconsumed = check_unconsumed_keys(component_cfg, component, fail_on_unconsumed=fail_on_unconsumed)
    
    invariants = {
        'validation_passed': len(errors) == 0 and len(unconsumed) == 0,
        'errors': [
            {
                'message': e.message,
                'path': e.path,
                'suggestion': e.suggestion
            }
            for e in errors
        ] if errors else [],
        'unconsumed_keys': unconsumed if unconsumed else [],
        'checks': {
            'weights_sum_to_one': _check_weights_sum(cfg, component),
            'thresholds_valid': _check_thresholds_valid(cfg, component),
            'ranges_valid': _check_ranges_valid(cfg, component),
        }
    }
    
    # 组装最终包
    pack = {
        '__meta__': meta,
        '__invariants__': invariants,
        **component_cfg
    }
    
    return pack


def _check_weights_sum(cfg: Dict[str, Any], component: str) -> Dict[str, Any]:
    """检查权重和是否等于1"""
    if component not in ('fusion', 'core_algo'):
        return {'applicable': False}
    
    fusion_weights = _get_nested_value(cfg, 'components.fusion.weights')
    if not fusion_weights:
        return {'applicable': False, 'found': False}
    
    w_ofi = fusion_weights.get('w_ofi', fusion_weights.get('ofi', 0.0))
    w_cvd = fusion_weights.get('w_cvd', fusion_weights.get('cvd', 0.0))
    total = w_ofi + w_cvd
    
    return {
        'applicable': True,
        'w_ofi': w_ofi,
        'w_cvd': w_cvd,
        'sum': total,
        'valid': abs(total - 1.0) < 1e-6
    }


def _check_thresholds_valid(cfg: Dict[str, Any], component: str) -> Dict[str, Any]:
    """检查阈值有效性"""
    if component not in ('fusion', 'core_algo'):
        return {'applicable': False}
    
    thresholds = _get_nested_value(cfg, 'components.fusion.thresholds')
    if not thresholds:
        return {'applicable': False, 'found': False}
    
    fuse_buy = thresholds.get('fuse_buy', 0.0)
    fuse_strong_buy = thresholds.get('fuse_strong_buy', 0.0)
    fuse_sell = thresholds.get('fuse_sell', 0.0)
    fuse_strong_sell = thresholds.get('fuse_strong_sell', 0.0)
    
    buy_valid = fuse_strong_buy >= fuse_buy
    sell_valid = fuse_strong_sell <= fuse_sell
    
    return {
        'applicable': True,
        'buy_valid': buy_valid,
        'sell_valid': sell_valid,
        'all_valid': buy_valid and sell_valid
    }


def _check_ranges_valid(cfg: Dict[str, Any], component: str) -> Dict[str, Any]:
    """检查范围有效性"""
    checks = {}
    
    if component == 'ofi':
        ofi = _get_nested_value(cfg, 'components.ofi')
        if ofi:
            checks['z_window'] = ofi.get('z_window', 0) > 0
            checks['ema_alpha'] = 0 < ofi.get('ema_alpha', 0) <= 1
            checks['z_clip'] = ofi.get('z_clip', 0) >= 0
    
    elif component == 'cvd':
        cvd = _get_nested_value(cfg, 'components.cvd')
        if cvd:
            checks['winsor_limit'] = cvd.get('winsor_limit', 0) > 0
            checks['mad_multiplier'] = cvd.get('mad_multiplier', 0) > 0
            checks['z_window'] = cvd.get('z_window', 0) > 0
    
    elif component in ('fusion', 'core_algo'):
        smoothing = _get_nested_value(cfg, 'components.fusion.smoothing')
        if smoothing:
            winsor_pct = smoothing.get('winsorize_percentile')
            checks['winsorize_percentile'] = winsor_pct is None or (1 <= winsor_pct <= 100)
    
    return {
        'applicable': len(checks) > 0,
        'checks': checks,
        'all_valid': all(checks.values()) if checks else True
    }


def _get_nested_value(cfg: Dict[str, Any], path: str) -> Any:
    """通过点号路径获取嵌套值"""
    keys = path.split('.')
    current = cfg
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def save_runtime_pack(pack: Dict[str, Any], output_path: Path,
                     create_current_link: bool = True) -> None:
    """
    保存运行时配置包到文件
    
    Args:
        pack: 运行时配置包字典
        output_path: 输出文件路径（应该包含版本和git sha）
        create_current_link: 是否创建current软链
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(pack, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    # 创建current软链（指向最新版本）
    if create_current_link:
        meta = pack.get('__meta__', {})
        component = meta.get('component', 'unknown')
        current_path = output_path.parent / f"{component}.runtime.current.yaml"
        
        # Windows不支持符号链接，使用复制
        if platform.system() == 'Windows':
            import shutil
            if current_path.exists():
                current_path.unlink()
            shutil.copy2(output_path, current_path)
        else:
            # Unix系统使用符号链接
            if current_path.exists() or current_path.is_symlink():
                current_path.unlink()
            current_path.symlink_to(output_path.name)

