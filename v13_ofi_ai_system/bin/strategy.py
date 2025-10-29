#!/usr/bin/env python3
"""
STRATEGY MODE策略模式管理器服务式入口（严格运行时模式）
(will verify scenarios_snapshot_sha256)
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from v13conf.runtime_loader import load_component_runtime_config, print_component_effective_config


def main():
    parser = argparse.ArgumentParser(description='STRATEGY MODE策略模式管理器组件')
    parser.add_argument('--config', default=None, help='显式指定运行时包路径')
    parser.add_argument('--dry-run-config', action='store_true', help='仅验证配置，不运行组件')
    parser.add_argument('--compat-global-config', action='store_true',
                       help='启用兼容模式：从全局配置目录加载（临时过渡选项）')
    parser.add_argument('--print-effective', action='store_true', help='打印有效配置')
    parser.add_argument('--verbose', action='store_true', help='详细模式')
    args = parser.parse_args()
    
    # 加载运行时配置（strategy组件会自动验证scenarios_snapshot_sha256）
    try:
        cfg = load_component_runtime_config(
            component='strategy',
            pack_path=args.config,
            compat_global=args.compat_global_config,
            verify_scenarios_snapshot=True  # Strategy组件启用场景快照指纹验证
        )
    except Exception as e:
        print(f"[错误] 加载运行时配置失败: {e}", file=sys.stderr)
        return 1
    
    # 打印有效配置（会显示场景快照指纹）
    if args.print_effective:
        print_component_effective_config(cfg, 'strategy', verbose=args.verbose)
        # 额外打印场景快照信息
        if 'scenarios_snapshot_sha256' in cfg:
            sha = cfg.get('scenarios_snapshot_sha256')
            print(f"\n场景快照指纹: {sha[:8]}...")
    
    # 干运行模式
    if args.dry_run_config:
        print("[DRY-RUN] STRATEGY组件配置验证通过")
        return 0
    
    # 实际运行组件（TODO: 添加自测或演示逻辑）
    print("[INFO] STRATEGY组件初始化成功（库式调用模式）")
    print("[提示] 请通过CoreAlgorithm或其他上层组件调用StrategyModeManager类")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

