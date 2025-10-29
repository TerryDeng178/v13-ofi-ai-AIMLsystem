#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指纹一致性双重校验
验证 print_config_origin 打印的指纹 == Prometheus 指标暴露的指纹
"""

import sys
import re
import subprocess
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_print_config_fingerprint():
    """获取 print_config_origin.py 打印的指纹"""
    result = subprocess.run(
        [sys.executable, "tools/print_config_origin.py"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=Path(__file__).parent.parent
    )
    
    # 从输出中提取指纹（尝试两种格式）
    output = result.stdout
    # 格式1: CONFIG_FINGERPRINT=xxxx
    match = re.search(r'CONFIG_FINGERPRINT=([0-9a-f]{16})', output)
    if match:
        return match.group(1)
    # 格式2: 指纹: xxxx
    match = re.search(r'指纹:\s*([0-9a-f]{16})', output)
    if match:
        return match.group(1)
    return None


def get_prometheus_fingerprint():
    """获取 Prometheus 指标中的指纹"""
    result = subprocess.run(
        [sys.executable, "tools/export_prometheus_metrics.py"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=Path(__file__).parent.parent
    )
    
    # 从输出中提取指纹
    output = result.stdout
    match = re.search(r'config_fingerprint\{.*\}\s+"([0-9a-f]{16})"', output)
    if match:
        return match.group(1)
    return None


def main():
    print("=" * 60)
    print("指纹一致性双重校验")
    print("=" * 60)
    
    fp_print = get_print_config_fingerprint()
    fp_prom = get_prometheus_fingerprint()
    
    print(f"\nprint_config_origin 指纹: {fp_print}")
    print(f"Prometheus 指标指纹: {fp_prom}")
    
    if not fp_print or not fp_prom:
        print("[FAIL] 未能提取到指纹")
        result = {
            "fingerprint_logs": fp_print,
            "fingerprint_metrics": fp_prom,
            "consistent": False,
            "error": "未能提取到指纹",
            "overall_pass": False
        }
    else:
        consistent = fp_print == fp_prom
        if consistent:
            print("\n[PASS] 指纹一致")
            print("[OK] 日志和指标暴露的指纹相同")
            result = {
                "fingerprint_logs": fp_print,
                "fingerprint_metrics": fp_prom,
                "consistent": True,
                "overall_pass": True
            }
        else:
            print("\n[FAIL] 指纹不一致")
            print(f"  差异: {fp_print} vs {fp_prom}")
            result = {
                "fingerprint_logs": fp_print,
                "fingerprint_metrics": fp_prom,
                "consistent": False,
                "error": f"指纹不一致: {fp_print} vs {fp_prom}",
                "overall_pass": False
            }
    
    # 输出 JSON 报告
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "fingerprint_consistency.json"
    
    try:
        with open(report_path, "w", encoding="utf-8", newline="") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[REPORT] 报告已保存到: {report_path}")
    except Exception as e:
        print(f"\n[WARN] 无法写入报告文件: {e}", file=sys.stderr)
    
    return 0 if result.get("overall_pass", False) else 1


if __name__ == "__main__":
    sys.exit(main())

