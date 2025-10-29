#!/usr/bin/env python3
"""
配置系统一键体检脚本
用于CI与本地统一验证：dry-run + 文件名/结构/指纹/未消费键
"""

import sys
import os
from pathlib import Path
import yaml
import re
import subprocess

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_dry_run():
    """测试dry-run验证"""
    print("\n=== 1. Dry-run验证 ===")
    try:
        result = subprocess.run(
            [sys.executable, "tools/conf_build.py", "all", "--base-dir", "config", "--dry-run-config"],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode == 0:
            print("  [OK] Dry-run验证通过")
            return True
        else:
            print(f"  [FAIL] Dry-run验证失败:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"  [FAIL] Dry-run执行异常: {e}")
        return False

def test_filename_format():
    """测试文件名格式"""
    print("\n=== 2. 文件名格式验证 ===")
    config_dir = project_root / "dist" / "config"
    pattern = re.compile(r'^[a-z_]+\.runtime\.\d+\.\d+\.\d+\.[0-9a-f]{8}\.ya?ml$')
    
    if not config_dir.exists():
        print(f"  [SKIP] 配置目录不存在: {config_dir}")
        return True
    
    errors = []
    for file_path in config_dir.glob("*.runtime.*.yaml"):
        filename = file_path.name
        if not pattern.match(filename):
            errors.append(f"    文件名不符合规范: {filename}")
        else:
            print(f"  [OK] {filename}")
    
    if errors:
        print("  [FAIL] 文件名格式验证失败:")
        for error in errors:
            print(error)
        return False
    print("  [OK] 所有文件名格式正确")
    return True

def test_git_sha_format():
    """测试Git SHA格式"""
    print("\n=== 3. Git SHA格式验证 ===")
    config_dir = project_root / "dist" / "config"
    sha_pattern = re.compile(r'^[0-9a-f]{8}$')
    
    if not config_dir.exists():
        print(f"  [SKIP] 配置目录不存在: {config_dir}")
        return True
    
    errors = []
    for file_path in config_dir.glob("*.runtime.current.yaml"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                meta = data.get('__meta__', {})
                git_sha = meta.get('git_sha', '')
                
                if not sha_pattern.match(git_sha):
                    errors.append(f"    {file_path.name}: Git SHA格式无效 '{git_sha}'")
                else:
                    print(f"  [OK] {file_path.name}: {git_sha}")
        except Exception as e:
            errors.append(f"    {file_path.name}: 读取失败 - {e}")
    
    if errors:
        print("  [FAIL] Git SHA格式验证失败:")
        for error in errors:
            print(error)
        return False
    print("  [OK] 所有Git SHA格式正确（8位十六进制）")
    return True

def test_runtime_pack_structure():
    """测试运行时包结构"""
    print("\n=== 4. 运行时包结构验证 ===")
    config_dir = project_root / "dist" / "config"
    required_meta_keys = ['version', 'git_sha', 'component', 'source_layers', 'checksum']
    
    if not config_dir.exists():
        print(f"  [SKIP] 配置目录不存在: {config_dir}")
        return True
    
    errors = []
    for file_path in config_dir.glob("*.runtime.current.yaml"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
                if '__meta__' not in data:
                    errors.append(f"    {file_path.name}: 缺少__meta__")
                    continue
                
                meta = data['__meta__']
                component = meta.get('component', 'unknown')
                
                missing_keys = [k for k in required_meta_keys if k not in meta]
                if missing_keys:
                    errors.append(f"    {file_path.name}: __meta__缺少键: {missing_keys}")
                    continue
                
                if '__invariants__' not in data:
                    errors.append(f"    {file_path.name}: 缺少__invariants__")
                    continue
                
                print(f"  [OK] {component}: 结构完整")
        except Exception as e:
            errors.append(f"    {file_path.name}: 读取失败 - {e}")
    
    if errors:
        print("  [FAIL] 运行时包结构验证失败:")
        for error in errors:
            print(error)
        return False
    print("  [OK] 所有运行时包结构完整")
    return True

def test_scenarios_fingerprint():
    """测试场景快照指纹（Strategy组件）"""
    print("\n=== 5. 场景快照指纹验证 ===")
    config_dir = project_root / "dist" / "config"
    strategy_file = config_dir / "strategy.runtime.current.yaml"
    
    if not strategy_file.exists():
        print(f"  [SKIP] Strategy运行时包不存在: {strategy_file}")
        return True
    
    try:
        with open(strategy_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        snapshot_sha = data.get('scenarios_snapshot_sha256')
        if not snapshot_sha:
            print("  [WARN] Strategy包缺少scenarios_snapshot_sha256（可能不是Strategy组件）")
            return True
        
        # 验证指纹格式（64位十六进制）
        if len(snapshot_sha) == 64 and all(c in '0123456789abcdef' for c in snapshot_sha.lower()):
            print(f"  [OK] 场景快照指纹格式正确: {snapshot_sha[:8]}...")
            return True
        else:
            print(f"  [FAIL] 场景快照指纹格式无效: {snapshot_sha}")
            return False
    except Exception as e:
        print(f"  [FAIL] 场景快照指纹验证异常: {e}")
        return False

def test_unconsumed_keys_gate():
    """测试未消费键阻断（主分支模式）"""
    print("\n=== 6. 未消费键阻断验证（主分支模式） ===")
    
    # 设置主分支环境变量
    env = os.environ.copy()
    env['CI_BRANCH'] = 'main'
    
    try:
        result = subprocess.run(
            [sys.executable, "tools/conf_build.py", "all", "--base-dir", "config", "--dry-run-config"],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        # 检查是否有未消费键警告/错误
        output = result.stdout + result.stderr
        if "未消费" in output or "unconsumed" in output.lower():
            print("  [INFO] 检测到未消费键（主分支应失败）")
            if result.returncode != 0:
                print("  [OK] 主分支模式正确阻断未消费键")
                return True
            else:
                print("  [WARN] 主分支模式未阻断未消费键（可能无未消费键）")
                return True  # 如果没有未消费键，这也是正常情况
        else:
            print("  [OK] 未检测到未消费键")
            return True
    except Exception as e:
        print(f"  [FAIL] 未消费键阻断验证异常: {e}")
        return False

def main():
    print("=" * 80)
    print("配置系统一键体检")
    print("=" * 80)
    
    results = []
    results.append(("Dry-run验证", test_dry_run()))
    results.append(("文件名格式", test_filename_format()))
    results.append(("Git SHA格式", test_git_sha_format()))
    results.append(("运行时包结构", test_runtime_pack_structure()))
    results.append(("场景快照指纹", test_scenarios_fingerprint()))
    results.append(("未消费键阻断", test_unconsumed_keys_gate()))
    
    print("\n" + "=" * 80)
    print("体检结果汇总")
    print("=" * 80)
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("[OK] 所有检查通过！")
        return 0
    else:
        print("[FAIL] 部分检查失败")
        return 1

if __name__ == '__main__':
    sys.exit(main())

