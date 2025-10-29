#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 RELEASE_NOTES.md
自动汇总所有测试报告和验证结果
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_json_report(report_path: Path) -> dict:
    """加载 JSON 报告"""
    if not report_path.exists():
        return None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {report_path}: {e}", file=sys.stderr)
        return None

def generate_release_notes():
    """生成发布说明"""
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # 加载所有报告
    validate_config = load_json_report(reports_dir / "validate_config_summary.json")
    runtime_probe = load_json_report(reports_dir / "runtime_probe_report.json")
    paper_canary = load_json_report(reports_dir / "paper_canary_report.json")
    fingerprint = load_json_report(reports_dir / "fingerprint_consistency.json")
    
    # 生成 Markdown
    lines = []
    lines.append("# Config 收口 & 防回归合并：Fail Gate/指纹/热更抗抖/旧键清理")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 概述")
    lines.append("")
    lines.append("本次合并包含以下改进：")
    lines.append("- Fail Gate：冲突键检测机制")
    lines.append("- 指纹一致性：日志与 metrics 双重验证")
    lines.append("- 热更新抗抖：连续 reload 稳定性验证")
    lines.append("- 旧键清理：配置路径统一管理")
    lines.append("")
    
    # 配置验证报告
    lines.append("## 配置验证 (validate_config)")
    lines.append("")
    if validate_config:
        overall_pass = validate_config.get('overall_pass', False)
        mode = validate_config.get('mode', 'unknown')
        conflicts = validate_config.get('conflicts_count', 0)
        errors = validate_config.get('errors_count', 0)
        unknown = validate_config.get('unknown_count', 0)
        allow_legacy = validate_config.get('allow_legacy_keys', False)
        
        lines.append(f"- **模式**: {mode}")
        lines.append(f"- **状态**: {'✅ 通过' if overall_pass else '❌ 失败'}")
        lines.append(f"- **冲突数**: {conflicts}")
        lines.append(f"- **错误数**: {errors}")
        lines.append(f"- **未知键数**: {unknown}")
        
        # 只有在实际允许旧键且有冲突时才显示临时放行
        if allow_legacy and conflicts > 0:
            lines.append("- **允许旧键**: 是（临时放行）")
        elif allow_legacy and conflicts == 0:
            lines.append("- **状态说明**: 允许旧键但无冲突，配置正常")
        elif not allow_legacy and conflicts == 0 and errors == 0 and unknown == 0:
            lines.append("- **状态说明**: 严格模式，无冲突无错误，配置完全通过 ✅")
    else:
        lines.append("- ⚠️ 报告不可用")
    lines.append("")
    
    # 运行时探针报告
    lines.append("## 运行时探针 (runtime_probe)")
    lines.append("")
    if runtime_probe:
        smoke = runtime_probe.get('smoke_test', {})
        reload = runtime_probe.get('stress_reload', {})
        lines.append(f"- **冒烟测试**: {'✅ 通过' if smoke.get('passed') else '❌ 失败'}")
        lines.append(f"  - 时长: {smoke.get('duration_secs', 0):.1f}s")
        lines.append(f"  - 错误数: {smoke.get('error_count', 0)}")
        lines.append(f"- **热更新测试**: {'✅ 通过' if reload.get('passed') else '❌ 失败'}")
        lines.append(f"  - reload 次数: {reload.get('reload_count', 0)}")
        lines.append(f"  - p50 时延: {reload.get('reload_latency_p50_ms', 0):.2f}ms")
        lines.append(f"  - p95 时延: {reload.get('reload_latency_p95_ms', 0):.2f}ms")
        lines.append(f"  - p99 时延: {reload.get('reload_latency_p99_ms', 0):.2f}ms")
    else:
        lines.append("- ⚠️ 报告不可用")
    lines.append("")
    
    # 纸上交易金丝雀报告
    lines.append("## 纸上交易金丝雀测试 (paper_canary)")
    lines.append("")
    if paper_canary:
        duration_mins = paper_canary.get('duration_minutes', 0)
        is_ci_short = duration_mins < 60
        
        lines.append(f"- **状态**: {'✅ 通过' if paper_canary.get('overall_pass') else '❌ 失败'}")
        lines.append(f"- **运行时长**: {duration_mins:.1f} 分钟" + (" (CI 短版，生产建议运行 60 分钟)" if is_ci_short else ""))
        lines.append(f"- **撮合错误**: {paper_canary.get('matching_errors', 0)}")
        lines.append(f"- **p99 时延**: {paper_canary.get('latency', {}).get('p99', 0):.2f}ms "
                    f"({'✅' if paper_canary.get('p99_passed') else '❌'})")
        
        signals = paper_canary.get('signals', {})
        lines.append(f"- **信号触发率** (活跃时段):")
        lines.append(f"  - OFI: {signals.get('ofi', {}).get('trigger_rate', 0):.4f}/s")
        lines.append(f"  - CVD: {signals.get('cvd', {}).get('trigger_rate', 0):.4f}/s")
        lines.append(f"  - Fusion: {signals.get('fusion', {}).get('trigger_rate', 0):.4f}/s")
        lines.append(f"  - Divergence: {signals.get('divergence', {}).get('trigger_rate', 0):.4f}/s")
    else:
        lines.append("- ⚠️ 报告不可用")
    lines.append("")
    
    # 指纹一致性报告
    lines.append("## 指纹一致性 (fingerprint_consistency)")
    lines.append("")
    if fingerprint:
        lines.append(f"- **状态**: {'✅ 一致' if fingerprint.get('consistent') else '❌ 不一致'}")
        lines.append(f"- **日志指纹**: {fingerprint.get('fingerprint_logs', 'N/A')}")
        lines.append(f"- **Metrics 指纹**: {fingerprint.get('fingerprint_metrics', 'N/A')}")
        if fingerprint.get('error'):
            lines.append(f"- **错误**: {fingerprint.get('error')}")
    else:
        lines.append("- ⚠️ 报告不可用")
    lines.append("")
    
    # 总体状态
    lines.append("## 总体状态")
    lines.append("")
    all_passed = all([
        (validate_config and validate_config.get('overall_pass')) if validate_config else False,
        (runtime_probe and runtime_probe.get('overall_pass')) if runtime_probe else False,
        (paper_canary and paper_canary.get('overall_pass')) if paper_canary else False,
        (fingerprint and fingerprint.get('overall_pass')) if fingerprint else False
    ])
    
    lines.append(f"**整体状态**: {'✅ 全部通过' if all_passed else '❌ 存在问题'}")
    lines.append("")
    
    # 报告文件路径
    lines.append("## 报告文件")
    lines.append("")
    lines.append("详细报告位于 `reports/` 目录：")
    lines.append("- `validate_config_summary.json`")
    lines.append("- `runtime_probe_report.json`")
    lines.append("- `paper_canary_report.json`")
    lines.append("- `fingerprint_consistency.json`")
    lines.append("")
    
    # 写入文件
    notes_path = reports_dir / "RELEASE_NOTES.md"
    with open(notes_path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    
    print(f"[OK] Release notes generated: {notes_path}")
    return notes_path

if __name__ == "__main__":
    generate_release_notes()

