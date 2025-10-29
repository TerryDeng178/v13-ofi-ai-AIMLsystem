#!/usr/bin/env python3
"""
配置系统验收测试脚本
"""

import sys
import os
from pathlib import Path
import yaml
import re

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_filename_format():
    """测试文件名格式"""
    print("\n=== 测试1: 文件名格式验证 ===")
    config_dir = project_root / "dist" / "config"
    pattern = re.compile(r'^[a-z_]+\.runtime\.\d+\.\d+\.\d+\.[0-9a-f]{8}\.ya?ml$')
    
    errors = []
    for file_path in config_dir.glob("*.runtime.*.yaml"):
        filename = file_path.name
        if not pattern.match(filename):
            errors.append(f"  文件名不符合规范: {filename}")
        else:
            print(f"  ✓ {filename}")
    
    if errors:
        print("\n[失败] 文件名格式验证失败:")
        for error in errors:
            print(error)
        return False
    print("[通过] 所有文件名格式正确")
    return True

def test_git_sha_format():
    """测试Git SHA格式"""
    print("\n=== 测试2: Git SHA格式验证 ===")
    config_dir = project_root / "dist" / "config"
    sha_pattern = re.compile(r'^[0-9a-f]{8}$')
    
    errors = []
    for file_path in config_dir.glob("*.runtime.current.yaml"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                meta = data.get('__meta__', {})
                git_sha = meta.get('git_sha', '')
                
                if not sha_pattern.match(git_sha):
                    errors.append(f"  {file_path.name}: Git SHA格式无效 '{git_sha}'")
                else:
                    print(f"  ✓ {file_path.name}: {git_sha}")
        except Exception as e:
            errors.append(f"  {file_path.name}: 读取失败 - {e}")
    
    if errors:
        print("\n[失败] Git SHA格式验证失败:")
        for error in errors:
            print(error)
        return False
    print("[通过] 所有Git SHA格式正确（8位十六进制）")
    return True

def test_runtime_pack_structure():
    """测试运行时包结构"""
    print("\n=== 测试3: 运行时包结构验证 ===")
    config_dir = project_root / "dist" / "config"
    required_meta_keys = ['version', 'git_sha', 'component', 'source_layers', 'checksum']
    
    errors = []
    for file_path in config_dir.glob("*.runtime.current.yaml"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
                # 检查__meta__存在
                if '__meta__' not in data:
                    errors.append(f"  {file_path.name}: 缺少__meta__")
                    continue
                
                meta = data['__meta__']
                component = meta.get('component', 'unknown')
                
                # 检查必需的meta键
                missing_keys = [k for k in required_meta_keys if k not in meta]
                if missing_keys:
                    errors.append(f"  {file_path.name}: __meta__缺少键: {missing_keys}")
                    continue
                
                # 检查__invariants__存在
                if '__invariants__' not in data:
                    errors.append(f"  {file_path.name}: 缺少__invariants__")
                    continue
                
                print(f"  ✓ {component}: 结构完整")
        except Exception as e:
            errors.append(f"  {file_path.name}: 读取失败 - {e}")
    
    if errors:
        print("\n[失败] 运行时包结构验证失败:")
        for error in errors:
            print(error)
        return False
    print("[通过] 所有运行时包结构完整")
    return True

def test_path_format_display():
    """测试路径展示格式（POSIX分隔符）"""
    print("\n=== 测试4: 路径展示格式验证 ===")
    # 这个测试主要验证代码中是否正确使用了POSIX分隔符
    # 检查conf_build.py中的路径打印
    conf_build_path = project_root / "tools" / "conf_build.py"
    try:
        with open(conf_build_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if '.replace("\\\\", "/")' in content or ".replace('\\\\', '/')" in content or "print_path" in content:
                print("  ✓ conf_build.py已使用POSIX分隔符转换")
                return True
            else:
                print("  ⚠ conf_build.py可能未使用POSIX分隔符转换")
                return False
    except Exception as e:
        print(f"  ✗ 无法读取conf_build.py: {e}")
        return False

def main():
    print("=" * 80)
    print("配置系统验收测试")
    print("=" * 80)
    
    results = []
    results.append(("文件名格式", test_filename_format()))
    results.append(("Git SHA格式", test_git_sha_format()))
    results.append(("运行时包结构", test_runtime_pack_structure()))
    results.append(("路径展示格式", test_path_format_display()))
    
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有测试通过！")
        return 0
    else:
        print("✗ 部分测试失败")
        return 1

if __name__ == '__main__':
    sys.exit(main())

