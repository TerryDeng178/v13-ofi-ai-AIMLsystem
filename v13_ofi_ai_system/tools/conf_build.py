#!/usr/bin/env python3
"""
配置构建工具：从统一源生成组件运行时配置包
"""

import sys
import os
import argparse
from pathlib import Path
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from v13conf import load_config, normalize, validate_invariants, build_runtime_pack, save_runtime_pack
from v13conf.printer import print_config_tree, print_source_summary


COMPONENTS = ['ofi', 'cvd', 'fusion', 'divergence', 'strategy', 'core_algo', 'harvester']


def print_effective_config(cfg: dict, sources: dict, component: str = None, verbose: bool = False):
    """打印最终配置值和来源层（脱敏、折叠、降噪）"""
    from v13conf.printer import print_config_tree, print_source_summary
    
    print(f"\n{'='*80}")
    print(f"有效配置（组件: {component or 'all'}）")
    print(f"{'='*80}\n")
    
    # 使用优化后的打印函数
    if component:
        prefix = f"components.{component}"
        if prefix in cfg:
            print_config_tree(cfg[prefix], sources, component, verbose, indent="", prefix=prefix)
        # 也打印运行时相关配置
        for key in ['logging', 'performance', 'guards', 'output']:
            if key in cfg:
                print(f"\n{key}:")
                print_config_tree(cfg[key], sources, component, verbose, indent="  ", prefix=key)
    else:
        print_config_tree(cfg, sources, component, verbose, indent="", prefix="")
    
    # 打印来源统计摘要（默认不详细）
    print_source_summary(sources, component)


def build_component(base_dir: str, component: str, dry_run: bool = False,
                   print_effective: bool = False, version: str = "1.0.0",
                   allow_env_override_locked: bool = False) -> int:
    """构建单个组件的运行时包"""
    try:
        # 加载配置
        cfg, sources = load_config(base_dir, allow_env_override_locked=allow_env_override_locked)
        
        # 归一化
        cfg = normalize(cfg, warn_compat=True)
        
        # 验证不变量
        errors = validate_invariants(cfg, component)
        if errors:
            print(f"\n[错误] 组件 '{component}' 不变量验证失败：\n", file=sys.stderr)
            for error in errors:
                print(f"  - {error.path}: {error.message}", file=sys.stderr)
                if error.suggestion:
                    print(f"    建议: {error.suggestion}", file=sys.stderr)
            return 1
        
        # 打印有效配置（如果需要）
        if print_effective:
            # verbose模式需要从外部传入（通过build_component的verbose参数）
            verbose_mode = getattr(sys.modules[__name__], '_verbose_mode', False)
            print_effective_config(cfg, sources, component, verbose=verbose_mode)
        
        if dry_run:
            print(f"\n[DRY-RUN] 组件 '{component}' 配置验证通过")
            return 0
        
        # P1修复：主分支必须失败（未消费键治理）
        # 可通过环境变量CI_BRANCH或CI_DEFAULT_BRANCH判断
        is_main_branch = os.getenv('CI_BRANCH', '') in ('main', 'master') or \
                        os.getenv('CI_DEFAULT_BRANCH', '') in ('main', 'master') or \
                        os.getenv('GITHUB_REF', '').endswith('/main') or \
                        os.getenv('GITHUB_REF', '').endswith('/master')
        fail_on_unconsumed = is_main_branch  # 主分支失败，feature分支警告（P1修复：未消费键治理）
        
        # 构建运行时包
        pack = build_runtime_pack(cfg, component, sources, version, 
                                 check_unconsumed=True, 
                                 fail_on_unconsumed=fail_on_unconsumed,
                                 base_config_dir=base_dir)
        
        # 检查包中的不变量和未消费键
        invariants = pack.get('__invariants__', {})
        if not invariants.get('validation_passed', False):
            print(f"\n[警告] 组件 '{component}' 运行时包中检测到问题：", file=sys.stderr)
            for error in invariants.get('errors', []):
                print(f"  - {error['path']}: {error['message']}", file=sys.stderr)
            unconsumed = invariants.get('unconsumed_keys', [])
            if unconsumed:
                print(f"\n[警告] 发现未消费的配置键（可能是拼写错误）：", file=sys.stderr)
                for key in unconsumed[:10]:  # 只显示前10个
                    print(f"  - {key}", file=sys.stderr)
                if len(unconsumed) > 10:
                    print(f"  ... 还有 {len(unconsumed) - 10} 个未显示", file=sys.stderr)
            # 如果未消费键存在，作为警告而非错误（可配置为错误）
            if invariants.get('errors'):
                return 1
        
        # 保存到文件（使用版本化命名）
        output_dir = Path(base_dir).parent / "dist" / "config"
        meta = pack.get('__meta__', {})
        git_sha_short = meta.get('git_sha', 'unknown')
        version_tag = meta.get('version', '1.0.0')
        
        # 验证文件名格式（P0修复：确保文件名符合规范）
        import re
        filename_pattern = re.compile(r'^[a-z_]+\.runtime\.\d+\.\d+\.\d+\.[0-9a-f]{8}\.ya?ml$')
        output_filename = f"{component}.runtime.{version_tag}.{git_sha_short}.yaml"
        
        # 验证Git SHA格式（必须是8位十六进制）
        if not re.match(r'^[0-9a-f]{8}$', git_sha_short):
            print(f"\n[错误] Git SHA格式无效: {git_sha_short} (必须是8位十六进制)", file=sys.stderr)
            return 1
        
        # 验证文件名是否符合规范
        if not filename_pattern.match(output_filename):
            print(f"\n[错误] 文件名不符合规范: {output_filename}", file=sys.stderr)
            print(f"  期望格式: {{component}}.runtime.{{semver}}.{{git_sha8}}.yaml", file=sys.stderr)
            return 1
        
        output_path = output_dir / output_filename
        
        save_runtime_pack(pack, output_path, create_current_link=True)
        
        # 同时保存无版本号的别名（向后兼容）
        compat_path = output_dir / f"{component}.runtime.yaml"
        save_runtime_pack(pack, compat_path, create_current_link=False)
        
        print(f"\n[成功] 组件 '{component}' 运行时包已生成：")
        # P2修复：统一使用POSIX分隔符展示（内部用pathlib，展示统一用/）
        print_path = str(output_path).replace('\\', '/')
        print(f"  路径: {print_path}")
        print(f"  版本: {pack['__meta__']['version']}")
        print(f"  Git SHA: {pack['__meta__']['git_sha']}")
        print(f"  校验和: {pack['__meta__']['checksum']}")
        print(f"  来源统计: {pack['__meta__']['source_layers']}")
        
        return 0
        
    except Exception as e:
        print(f"\n[错误] 构建组件 '{component}' 时发生异常：{e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='构建V13系统组件运行时配置包',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 构建所有组件
  python conf_build.py all
  
  # 仅构建fusion组件
  python conf_build.py fusion
  
  # 干运行（仅验证，不写文件）
  python conf_build.py fusion --dry-run-config
  
  # 打印有效配置和来源
  python conf_build.py fusion --print-effective
  
  # 指定配置目录
  python conf_build.py all --base-dir v13_ofi_ai_system/config
        """
    )
    
    parser.add_argument(
        'component',
        choices=['all'] + COMPONENTS,
        help='要构建的组件（或"all"构建所有组件）'
    )
    
    parser.add_argument(
        '--base-dir',
        default='config',
        help='配置目录路径（默认: config）'
    )
    
    parser.add_argument(
        '--print-effective',
        action='store_true',
        help='打印最终有效配置值和来源层'
    )
    
    parser.add_argument(
        '--dry-run-config',
        action='store_true',
        dest='dry_run',
        help='仅验证配置，不生成文件'
    )
    
    parser.add_argument(
        '--version',
        default='1.0.0',
        help='版本号（默认: 1.0.0）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细模式：显示每个键的来源和完整列表内容'
    )
    
    parser.add_argument(
        '--allow-env-override-locked',
        action='store_true',
        help='允许环境变量覆盖OFI锁定参数（紧急场景）'
    )
    
    args = parser.parse_args()
    
    # 确定要构建的组件列表
    if args.component == 'all':
        components = COMPONENTS
    else:
        components = [args.component]
    
    # 构建每个组件
    exit_code = 0
    for component in components:
        code = build_component(
            args.base_dir,
            component,
            dry_run=args.dry_run,
            print_effective=args.print_effective,
            version=args.version,
            allow_env_override_locked=args.allow_env_override_locked
        )
        if code != 0:
            exit_code = code
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

