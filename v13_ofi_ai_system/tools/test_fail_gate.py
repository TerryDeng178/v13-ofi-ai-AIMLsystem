#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Fail Gate 功能
验证在未设置 ALLOW_LEGACY_KEYS 时，检测到冲突应该失败
"""

import os
import sys
import subprocess
from pathlib import Path

# 确保使用干净的环境
test_env = os.environ.copy()
if 'ALLOW_LEGACY_KEYS' in test_env:
    del test_env['ALLOW_LEGACY_KEYS']

print("=" * 60)
print("测试 Fail Gate（冲突检测应失败）")
print("=" * 60)
print(f"ALLOW_LEGACY_KEYS: {test_env.get('ALLOW_LEGACY_KEYS', '未设置')}")
print()

# 运行 validate_config
result = subprocess.run(
    [sys.executable, str(Path(__file__).parent / "validate_config.py"), "--strict", "--format", "json"],
    env=test_env,
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    cwd=Path(__file__).parent.parent
)

import json
try:
    output = json.loads(result.stdout)
    print(f"退出码: {result.returncode}")
    print(f"Legacy conflicts: {len(output.get('legacy_conflicts', []))}")
    print(f"Allow legacy: {output.get('allow_legacy_keys', False)}")
    print(f"Valid: {output.get('valid', False)}")
    print(f"Type errors: {len(output.get('type_errors', []))}")
    
    if output.get('legacy_conflicts'):
        print("\n检测到的冲突:")
        for c in output.get('legacy_conflicts', []):
            print(f"  - {c.get('description', 'N/A')}")
    
    if output.get('type_errors'):
        print("\n类型错误（包含冲突）:")
        for err in output.get('type_errors', []):
            if 'LEGACY这些_CONFLICT' in err:
                print(f"  ✓ {err}")
            else:
                print(f"  - {err}")
    
    # 验证：如果有冲突且未允许，应该失败
    has_conflicts = len(output.get('legacy_conflicts', [])) > 0
    allow_legacy = output.get('allow_legacy_keys', False)
    should_fail = has_conflicts and not allow_legacy
    actually_failed = result.returncode != 0
    
    print(f"\n验证结果:")
    print(f"  有冲突: {has_conflicts}")
    print(f"  允许旧键: {allow_legacy}")
    print(f"  应该失败: {should_fail}")
    print(f"  实际失败: {actually_failed}")
    
    if should_fail == actually_failed:
        print("\n[PASS] Fail Gate 工作正常")
    else:
        print("\n[FAIL] Fail Gate 行为不符合预期")
        sys.exit(1)
        
except Exception as e:
    print(f"解析输出失败: {e}")
    print(f"原始输出: {result.stdout[:500]}")
    sys.exit(1)

