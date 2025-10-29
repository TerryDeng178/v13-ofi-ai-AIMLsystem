#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置校验脚本 - GCC (Global Config Check)
验证 system.yaml 配置是否符合 schema，并检查类型匹配
"""

import sys
import yaml
import json
import pathlib
from typing import Any, Dict, Tuple, List, Union


# Schema 定义（基于实际 system.yaml 结构）
# 支持严格模式和宽松模式
SCHEMA = {
    # 系统基本信息
    "system": {
        "name": str,
        "version": str,
        "environment": str,
        "description": str,
        "author": str,
        "created_at": str,
        "timezone": (str, type(None)),  # 可选
        "env": (str, type(None))  # 可选
    },
    
    # 背离检测配置
    "divergence_detection": dict,  # 复杂结构，允许任意键
    
    # 融合指标配置
    "fusion_metrics": {
        "version": (str, type(None)),
        "description": (str, type(None)),
        "weights": dict,
        "thresholds": {
            "fuse_buy": float,  # 细化类型：必须是 float
            "fuse_sell": float,
            "fuse_strong_buy": float,
            "fuse_strong_sell": float,
            "regime_thresholds": (dict, type(None))
        },
        "consistency": dict,
        "data_processing": (dict, type(None)),
        "warmup": (dict, type(None)),
        "denoising": (dict, type(None)),
        "advanced_mechanisms": (dict, type(None)),
        "performance": (dict, type(None)),
        "alerts": (dict, type(None)),
        "hot_reload": (dict, type(None)),
        "logging": (dict, type(None))
    },
    
    # 门控配置
    "gating": dict,
    
    # 组件配置
    "components": {
        "ofi": dict,
        "cvd": dict,
        "ai": (dict, type(None)),
        "trading": (dict, type(None))
    },
    
    # 数据源配置
    "data_source": dict,
    
    # 路径配置
    "paths": dict,
    
    # 日志配置
    "logging": {
        "level": str,
        "level_by_mode": (dict, type(None)),
        "format": (str, type(None)),
        "date_format": (str, type(None)),
        "file": (dict, type(None)),
        "console": (dict, type(None)),
        "dir": (str, type(None))  # 向后兼容
    },
    
    # 监控配置 consolidated
    "monitoring": {
        "enabled": bool,
        "interval_seconds": (int, type(None)),
        "metrics": (dict, type(None)),
        "prometheus": (dict, type(None)),
        "divergence_metrics": (dict, type(None)),
        "fusion_metrics": (dict, type(None)),
        "metrics_path": (str, type(None)),  # 向后兼容
        "enable": (bool, type(None))  # 向后兼容
    },
    
    # 策略模式配置
    "strategy_mode": dict,
    
    # 信号分析配置
    "signal_analysis": (dict, type(None)),
    
    # 融合指标收集器配置
    "fusion_metrics_collector": (dict, type(None)),
    
    # 交易流配置
    "trade_stream": (dict, type(None)),
    
    # 数据库配置
    "database": (dict, type(None)),
    
    # 测试配置
    "testing": (dict, type(None)),
    
    # 数据采集配置
    "data_harvest": (dict, type(None)),
    
    # 功能开关
    "features": dict,
    
    # 通知配置
    "notifications": (dict, type(None)),
    
    # 安全配置
    "security": (dict, type(None)),
    
    # 向后兼容的旧键名
    "binance": (dict, type(None)),
    "storage": (dict, type(None)),
    "harvest": (dict, type(None)),
    "ofi": (dict, type(None)),
    "cvd": (dict, type(None)),
    "fusion": (dict, type(None)),
    "divergence": (dict, type(None)),
    "strategymode": (dict, type(None)),
    "core_algo": (dict, type(None))
}

# 宽松模式允许的额外段落（不报错，只警告）
LENIENT_ALLOWED = set()


def _check_types(tree: Dict[str, Any], schema: Dict[str, Any], prefix: str = "") -> Tuple[List[str], List[str]]:
    """
    递归检查配置树是否符合 schema
    
    Args:
        tree: 配置树（字典）
        schema: schema 定义（字典）
        prefix: 当前路径前缀（用于错误消息）
    
    Returns:
        (errs, unknown): 类型错误列表和未知键列表
    """
    errs: List[str] = []
    unknown: List[str] = []
    
    if not isinstance(tree, dict):
        return [f"type:{prefix} expected dict"], []
    
    for k, v in tree.items():
        if k not in schema:
            unknown.append(prefix + k)
            continue
        
        expected = schema[k]
        
        # 如果期望的是字典，递归检查
        if isinstance(expected, dict):
            if not isinstance(v, dict):
                errs.append(f"type:{prefix + k} expected dict")
                continue
            e2, u2 = _check_types(v, expected, prefix + k + ".")
            errs.extend(e2)
            unknown.extend(u2)
        # 如果期望的是 dict 类型（表示允许任意键值）
        elif expected is dict:
            if not isinstance(v, dict):
                errs.append(f"type:{prefix + k} expected dict, got {type(v).__name__}")
                continue
            # dict 类型表示允许任意内容，不递归检查
        # 如果期望的是类型或类型元组
        elif isinstance(expected, tuple):
            # 检查是否为 None 或期望类型之一
            if v is None and type(None) in expected:
                continue
            if not isinstance(v, expected):
                errs.append(f"type:{prefix + k} expected one of {expected}, got {type(v).__name__}")
        else:
            # 单个类型
            if not isinstance(v, expected):
                errs.append(f"type:{prefix + k} expected {expected.__name__}, got {type(v).__name__}")
    
    return errs, unknown


def _get_nested_value(cfg: Dict[str, Any], path: str) -> Tuple[bool, Any]:
    """获取嵌套字典中的值"""
    parts = path.split(".")
    cur = cfg
    for i, part in enumerate(parts):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            if i == len(parts) - 1:
                return True, cur
        else:
            return False, None
    return False, None


def _check_legacy_key_conflicts(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    检查旧键与新真源的冲突（Fail Gate）
    
    Args:
        cfg: 配置字典
    
    Returns:
        冲突列表（每个冲突为字典）
    """
    conflicts = []
    
    # 定义旧键到新真源的映射
    LEGACY_CONFLICTS = [
        {
            "legacy": "components.fusion.thresholds",
            "canonical": "fusion_metrics.thresholds",
            "description": "components.fusion.thresholds.* (已废弃) vs fusion_metrics.thresholds.* (单一真源)"
        },
        {
            "legacy": "components.strategy.triggers.market",
            "canonical": "strategy_mode.triggers.market",
            "description": "components.strategy.triggers.market.* (已废弃) vs strategy_mode.triggers.market.* (单一真源)"
        }
    ]
    
    # 检查每个冲突
    for conflict in LEGACY_CONFLICTS:
        legacy_exists, _ = _get_nested_value(cfg, conflict["legacy"])
        canonical_exists, _ = _get_nested_value(cfg, conflict["canonical"])
        
        # 如果两者都存在，报告冲突
        if legacy_exists and canonical_exists:
            conflicts.append({
                "description": conflict["description"],
                "legacy_path": conflict["legacy"],
                "canonical_path": conflict["canonical"],
                "recommendation": f"移除 {conflict['legacy']}，统一使用 {conflict['canonical']}"
            })
    
    return conflicts


def validate_config(config_path: str = "config/system.yaml", strict: bool = True) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
    """
    验证配置文件
    
    Args:
        config_path: 配置文件路径
        strict: 是否严格模式
    
    Returns:
        (type_errors, unknown_keys, legacy_conflicts): 类型错误、未知键和旧键冲突
    """
    import os
    
    config_file = pathlib.Path(config_path)
    
    if not config_file.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[ERROR] 解析配置文件失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not isinstance(cfg, dict):
        print(f"[ERROR] 配置文件必须是字典格式", file=sys.stderr)
        sys.exit(1)
    
    errs, unknown = _check_types(cfg, SCHEMA)
    
    # 检查旧键冲突（Fail Gate）- 检查合并后的有效配置
    legacy_conflicts = []
    try:
        # 使用 UnifiedConfigLoader 获取合并后的配置（包括 defaults.yaml 和 system.yaml）
        import sys as _sys
        from pathlib import Path as _Path
        _tools_dir = _Path(__file__).parent
        _project_root = _tools_dir.parent
        _sys.path.insert(0, str(_project_root))
        from config.unified_config_loader import UnifiedConfigLoader
        loader = UnifiedConfigLoader(base_dir=config_file.parent)
        merged_cfg = loader.get()
        legacy_conflicts = _check_legacy_key_conflicts(merged_cfg)
    except Exception:
        # 如果加载失败，只检查当前文件
        legacy_conflicts = _check_legacy_key_conflicts(cfg)
    
    allow_legacy = os.environ.get("ALLOW_LEGACY_KEYS", "0") == "1"
    
    if legacy_conflicts and not allow_legacy:
        # 默认失败（除非设置了 ALLOW_LEGACY_KEYS=1）
        for conflict in legacy_conflicts:
            errs.append(f"LEGACY_CONFLICT: {conflict['description']} - {conflict['recommendation']}")
    
    return errs, unknown, legacy_conflicts


def main():
    """主函数"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="配置校验脚本 - GCC")
    parser.add_argument(
        "--config",
        default="config/system.yaml",
        help="配置文件路径（默认: config/system.yaml）"
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="输出格式（默认: json）"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：未知键报错（默认）"
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="宽松模式：未知键只警告，不报错"
    )
    
    args = parser.parse_args()
    
    # 处理相对路径
    if not pathlib.Path(args.config).is_absolute():
        # 从 tools 目录向上找到项目根目录
        tools_dir = pathlib.Path(__file__).parent
        project_root = tools_dir.parent
        config_path = project_root / args.config
    else:
        config_path = pathlib.Path(args.config)
    
    # 确定模式
    strict_mode = args.strict or (not args.lenient)  # 默认严格模式
    
    errs, unknown, legacy_conflicts = validate_config(str(config_path), strict=strict_mode)
    
    # 在宽松模式下，未知键只警告，不报错
    if not strict_mode:
        warnings = unknown
        unknown = []
    else:
        warnings = []
    
    allow_legacy = os.environ.get("ALLOW_LEGACY_KEYS", "0") == "1"
    if legacy_conflicts:
        if allow_legacy:
            warnings.extend([f"LEGACY: {c['description']}" for c in legacy_conflicts])
        else:
            # 在严格模式下，旧键冲突会添加到 errs，这里只记录信息
            pass
    
    # 构建结果摘要
    result = {
        "type_errors": errs,
        "unknown_keys": unknown,
        "warnings": warnings,
        "legacy_conflicts": legacy_conflicts,
        "allow_legacy_keys": allow_legacy,
        "valid": len(errs) == 0 and len(unknown) == 0,
        "mode": "strict" if strict_mode else "lenient",
        "conflicts_count": len(legacy_conflicts),
        "errors_count": len(errs),
        "unknown_count": len(unknown),
        "overall_pass": len(errs) == 0 and len(unknown) == 0 and (allow_legacy or len(legacy_conflicts) == 0),
        "config_path": str(config_path)
    }
    
    # 输出到 reports/ 目录
    reports_dir = pathlib.Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    summary_path = reports_dir / "validate_config_summary.json"
    
    try:
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] 无法写入摘要文件 {summary_path}: {e}", file=sys.stderr)
    
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"模式: {'严格' if strict_mode else '宽松'}")
        
        if errs:
            print("[TYPE ERRORS]")
            for err in errs:
                print(f"  - {err}")
        
        if unknown:
            print("[UNKNOWN KEYS]")
            for key in unknown:
                print(f"  - {key}")
        
        if legacy_conflicts:
            print("[LEGACY KEY CONFLICTS]")
            for conflict in legacy_conflicts:
                print(f"  - {conflict['description']}")
                print(f"    建议: {conflict['recommendation']}")
                if allow_legacy:
                    print(f"    (当前允许，设置了 ALLOW_LEGACY_KEYS=1)")
                else:
                    print(f"    (默认失败，设置 ALLOW_LEGACY_KEYS=1 可临时放行)")
        
        if warnings:
            print("[WARNINGS] (宽松模式)")
            for key in warnings:
                print(f"  - {key}")
        
        if not errs and not unknown:
            if warnings:
                print(f"[WARN] 配置验证通过，但有 {len(warnings)} 个警告")
            else:
                print("[OK] 配置验证通过")
        else:
            print(f"\n[FAIL] 发现 {len(errs)} 个类型错误, {len(unknown)} 个未知键")
    
    # 退出码逻辑（独立处理冲突和其他错误）
    allow_legacy = os.environ.get("ALLOW_LEGACY_KEYS", "0") == "1"
    conflict_count = len(legacy_conflicts)
    other_errors = len(errs) + len(unknown)
    
    # 如果未允许但有冲突，失败
    if conflict_count and not allow_legacy:
        sys.exit(1)  # 冲突即失败
    
    # 如果有其他错误，失败
    if other_errors:
        sys.exit(1)  # 其余错误同样失败
    
    # 通过
    sys.exit(0)


if __name__ == "__main__":
    main()
