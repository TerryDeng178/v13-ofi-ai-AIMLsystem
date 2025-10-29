#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局配置到位检查 (GCC: Global Config Check)
根据 globletest.md 文档执行完整检查
"""

import os
import sys
import re
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple


def find_env_direct_reads(src_dir: str = "src") -> List[Tuple[str, int, str]]:
    """
    扫描环境变量直读（必须为0条）
    
    Returns:
        List of (file_path, line_num, line_content)
    """
    issues = []
    src_path = Path(src_dir)
    
    if not src_path.exists():
        return issues
    
    # 匹配模式
    patterns = [
        r'os\.getenv\s*\(',
        r'os\.environ\[',
        r'from dotenv import',
        r'import dotenv',
        r'load_dotenv\s*\(',
    ]
    
    for py_file in src_path.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    for pattern in patterns:
                        if re.search(pattern, line):
                            # 排除注释行
                            stripped = line.strip()
                            if not stripped.startswith("#"):
                                issues.append((str(py_file), line_num, line.strip()))
                                break
        except Exception as e:
            print(f"[WARN] 无法读取文件 {py_file}: {e}", file=sys.stderr)
    
    return issues


def check_constructor_injection(src_dir: str = "src") -> Dict[str, List[Dict]]:
    """
    验证构造函数注入模式
    
    Returns:
        Dict with component names and their injection status
    """
    components = {
        "harvest": [],
        "ofi": [],
        "cvd": [],
        "fusion": [],
        "divergence": [],
        "strategymode": [],
        "core_algo": []
    }
    
    src_path = Path(src_dir)
    if not src_path.exists():
        return components
    
    # 搜索组件文件
    component_patterns = {
        "harvest": ["*harvest*.py", "*harvester*.py"],
        "ofi": ["*ofi*.py"],
        "cvd": ["*cvd*.py"],
        "fusion": ["*fusion*.py"],
        "divergence": ["*divergence*.py"],
        "strategymode": ["*strategy*.py", "*mode*.py"],
        "core_algo": ["*core*.py", "*algo*.py"]
    }
    
    for comp_name, patterns in component_patterns.items():
        for pattern in patterns:
            for py_file in src_path.rglob(pattern):
                if "test" in str(py_file).lower():
                    continue
                
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                        # 查找 __init__ 方法
                        init_match = re.search(r'def __init__\(self[^)]*\):', content)
                        if init_match:
                            # 检查是否包含 cfg 或 config_loader 参数
                            has_cfg = "cfg" in content[init_match.start():init_match.start()+500]
                            has_config_loader = "config_loader" in content[init_match.start():init_match.start()+500]
                            
                            components[comp_name].append({
                                "file": str(py_file),
                                "has_cfg": has_cfg,
                                "has_config_loader": has_config_loader
                            })
                except Exception as e:
                    print(f"[WARN] 无法分析文件 {py_file}: {e}", file=sys.stderr)
    
    return components


def check_unified_config_loader() -> Dict[str, any]:
    """
    检查统一配置加载器是否支持 system.yaml
    """
    config_loader_file = Path("config/unified_config_loader.py")
    result = {
        "exists": config_loader_file.exists(),
        "supports_system_yaml": False,
        "supports_defaults_yaml": False,
        "supports_env_override": False
    }
    
    if not result["exists"]:
        return result
    
    try:
        with open(config_loader_file, "r", encoding="utf-8") as f:
            content = f.read()
            result["supports_defaults_yaml"] = "defaults.yaml" in content
            result["supports_system_yaml"] = "system.yaml" in content or '"system.yaml"' in content
            result["supports_env_override"] = "env_prefix" in content and "V13" in content
    except Exception as e:
        print(f"[WARN] 无法分析配置加载器: {e}", file=sys.stderr)
    
    return result


def run_config_validation() -> Dict:
    """
    运行配置验证脚本
    """
    validate_script = Path("tools/validate_config.py")
    if not validate_script.exists():
        return {"error": "validate_config.py not found"}
    
    try:
        result = subprocess.run(
            [sys.executable, str(validate_script), "--format", "json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            try:
                return json.loads(result.stdout)
            except:
                return {"error": result.stderr}
    except Exception as e:
        return {"error": str(e)}


def main():
    """主函数"""
    print("=" * 60)
    print("全局配置到位检查 (GCC: Global Config Check)")
    print("=" * 60)
    print()
    
    # 切换到项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    all_results = {
        "env_direct_reads": [],
        "constructor_injection": {},
        "config_loader": {},
        "config_validation": {},
        "summary": {}
    }
    
    # 1. 检查环境变量直读
    print("[1/4] 检查环境变量直读...")
    env_issues = find_env_direct_reads("src")
    all_results["env_direct_reads"] = [
        {"file": f, "line": l, "content": c} for f, l, c in env_issues
    ]
    print(f"  发现 {len(env_issues)} 个环境变量直读")
    if env_issues:
        for f, l, c in env_issues[:10]:  # 只显示前10个
            print(f"    {f}:{l} - {c[:60]}")
        if len(env_issues) > 10:
            print(f"    ... 还有 {len(env_issues) - 10} 个")
    print()
    
    # 2. 检查构造函数注入
    print("[2/4] 检查构造函数注入...")
    injection_status = check_constructor_injection("src")
    all_results["constructor_injection"] = injection_status
    
    for comp_name, files in injection_status.items():
        if files:
            print(f"  {comp_name}: {len(files)} 个文件")
            cfg_count = sum(1 for f in files if f.get("has_cfg") or f.get("has_config_loader"))
            print(f"    其中 {cfg_count} 个支持配置注入")
    print()
    
    # 3. 检查配置加载器
    print("[3/4] 检查统一配置加载器...")
    loader_status = check_unified_config_loader()
    all_results["config_loader"] = loader_status
    print(f"  存在: {loader_status['exists']}")
    print(f"  支持 defaults.yaml: {loader_status['supports_defaults_yaml']}")
    print(f"  支持 system.yaml: {loader_status['supports_system_yaml']}")
    print(f"  支持环境变量覆盖: {loader_status['supports_env_override']}")
    print()
    
    # 4. 运行配置验证
    print("[4/4] 运行配置验证...")
    validation_result = run_config_validation()
    all_results["config_validation"] = validation_result
    
    if "error" in validation_result:
        print(f"  错误: {validation_result['error']}")
    else:
        type_errors = len(validation_result.get("type_errors", []))
        unknown_keys = len(validation_result.get("unknown_keys", []))
        print(f"  类型错误: {type_errors}")
        print(f"  未知键: {unknown_keys}")
        print(f"  验证通过: {validation_result.get('valid', False)}")
    print()
    
    # 生成摘要
    print("=" * 60)
    print("检查摘要")
    print("=" * 60)
    
    env_pass = len(env_issues) == 0
    injection_good = any(
        sum(1 for f in files if f.get("has_cfg") or f.get("has_config_loader")) > 0
        for files in injection_status.values()
    )
    loader_pass = (
        loader_status["exists"] and
        loader_status["supports_defaults_yaml"] and
        loader_status["supports_env_override"]
    )
    validation_pass = validation_result.get("valid", False) if "error" not in validation_result else False
    
    all_results["summary"] = {
        "env_direct_reads_pass": env_pass,
        "constructor_injection_pass": injection_good,
        "config_loader_pass": loader_pass,
        "config_validation_pass": validation_pass,
        "overall_pass": env_pass and injection_good and loader_pass
    }
    
    print(f"环境变量直读检查: {'[PASS]' if env_pass else '[FAIL]'} ({len(env_issues)} 个问题)")
    print(f"构造函数注入检查: {'[PASS]' if injection_good else '[WARN]'}")
    print(f"配置加载器检查: {'[PASS]' if loader_pass else '[FAIL]'}")
    print(f"配置验证检查: {'[PASS]' if validation_pass else '[FAIL]'}")
    print()
    print(f"总体状态: {'[GO]' if all_results['summary']['overall_pass'] else '[NO-GO]'}")
    print()
    
    # 保存结果到 JSON
    output_file = project_root / "reports" / "gcc_check_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"详细结果已保存到: {output_file}")
    
    return 0 if all_results["summary"]["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

