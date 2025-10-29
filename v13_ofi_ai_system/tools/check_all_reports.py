#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查所有报告文件是否齐全且通过
"""

import json
from pathlib import Path

reports_dir = Path(__file__).parent.parent / "reports"
required_files = [
    'validate_config_summary.json',
    'runtime_probe_report.json',
    'paper_canary_report.json',
    'fingerprint_consistency.json'
]

print("=" * 60)
print("报告完整性即时检查")
print("=" * 60)

all_exist = True
all_passed = True

for filename in required_files:
    filepath = reports_dir / filename
    if filepath.exists():
        print(f"\n[OK] {filename} 存在")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            pass_status = data.get("overall_pass", False)
            status_str = "PASS" if pass_status else "FAIL"
            print(f"     状态: {status_str}")
            
            if not pass_status:
                all_passed = False
                print(f"     详情: {json.dumps(data, indent=2, ensure_ascii=False)[:200]}...")
        except Exception as e:
            print(f"     [ERROR] 无法解析: {e}")
            all_passed = False
    else:
        print(f"\n[MISSING] {filename}")
        all_exist = False
        all_passed = False

print("\n" + "=" * 60)
if all_exist and all_passed:
    print("[SUCCESS] 所有报告齐全且通过")
    exit(0)
elif all_exist:
    print("[WARNING] 所有报告齐全，但部分未通过")
    exit(1)
else:
    print("[ERROR] 部分报告缺失")
    exit(1)

