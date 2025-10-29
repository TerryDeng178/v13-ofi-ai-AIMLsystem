#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
负向回归测试：验证 Fail Gate 和 ALLOW_LEGACY_KEYS 行为
"""

import sys
import os
import subprocess
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_conflict_keys_fail_by_default():
    """
    场景 A：并存新旧键 → 退出码=1（默认失败）
    """
    fixture_path = Path(__file__).parent / "fixtures" / "both_keys.yml"
    tools_dir = Path(__file__).parent.parent / "tools"
    validate_script = tools_dir / "validate_config.py"
    
    # 清除 ALLOW_LEGACY_KEYS 环境变量（如果存在）
    env = os.environ.copy()
    env.pop("ALLOW_LEGACY_KEYS", None)
    
    result = subprocess.run(
        [sys.executable, str(validate_script), "--config", str(fixture_path), "--strict", "--format", "json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=str(Path(__file__).parent.parent)
    )
    
    # 应该失败（退出码=1）
    assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}. Output: {result.stdout}\nError: {result.stderr}"
    
    # 解析 JSON 输出
    try:
        output = json.loads(result.stdout)
        assert output["conflicts_count"] > 0, "Should detect conflicts"
        assert not output["overall_pass"], "Should not pass with conflicts"
    except json.JSONDecodeError:
        # 如果不是 JSON 格式，检查 stderr
        assert "LEGACY_CONFLICT" in result.stderr or "conflict" in result.stderr.lower()


def test_conflict_keys_pass_with_allow_legacy():
    """
    场景 B：ALLOW_LEGACY_KEYS=1 → 退出码=0 但记录告警
    """
    fixture_path = Path(__file__).parent / "fixtures" / "both_keys.yml"
    tools_dir = Path(__file__).parent.parent / "tools"
    validate_script = tools_dir / "validate_config.py"
    
    env = os.environ.copy()
    env["ALLOW_LEGACY_KEYS"] = "1"
    
    result = subprocess.run(
        [sys.executable, str(validate_script), "--config", str(fixture_path), "--strict", "--format", "json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=str(Path(__file__).parent.parent)
    )
    
    # 应该通过（退出码=0）
    assert result.returncode == 0, f"Expected exit code 0 with ALLOW_LEGACY_KEYS=1, got {result.returncode}. Output: {result.stdout}\nError: {result.stderr}"
    
    # 解析 JSON 输出
    try:
        output = json.loads(result.stdout)
        assert output["allow_legacy_keys"] is True, "Should have allow_legacy_keys=True"
        assert len(output.get("warnings", [])) > 0, "Should have warnings about legacy keys"
        # 即使有冲突，如果允许旧键，应该 overall_pass=True（只要没有其他错误）
        if output["errors_count"] == 0 and output["unknown_count"] == 0:
            assert output["overall_pass"] is True, "Should pass when allowing legacy keys"
    except json.JSONDecodeError:
        # 如果输出不是 JSON，至少检查退出码
        pass


def test_type_error_fail():
    """
    场景 C：类型错（fusion_metrics.thresholds.* 注入字符串）→ Schema 失败
    """
    # 创建一个临时配置文件，包含类型错误
    import tempfile
    import yaml
    
    bad_config = {
        "fusion_metrics": {
            "thresholds": {
                "fuse_buy": "not a number",  # 应该是 float
                "fuse_sell": -0.95,
                "fuse_strong_buy": 1.70,
                "fuse_strong_sell": -1.70
            }
        },
        "logging": {
            "level": "INFO"
        },
        "monitoring": {
            "enabled": True
        }
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8") as f:
        yaml.dump(bad_config, f, allow_unicode=True)
        temp_path = f.name
    
    try:
        tools_dir = Path(__file__).parent.parent / "tools"
        validate_script = tools_dir / "validate_config.py"
        
        result = subprocess.run(
            [sys.executable, str(validate_script), "--config", temp_path, "--strict", "--format", "json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(Path(__file__).parent.parent)
        )
        
        # 应该失败（退出码=1）
        assert result.returncode == 1, f"Expected exit code 1 for type error, got {result.returncode}"
        
        # 解析 JSON 输出
        try:
            output = json.loads(result.stdout)
            assert output["errors_count"] > 0 or len(output.get("type_errors", [])) > 0, "Should detect type errors"
            assert not output["overall_pass"], "Should not pass with type errors"
        except json.JSONDecodeError:
            pass
    
    finally:
        os.unlink(temp_path)


def test_write_summary_json():
    """
    验证 validate_config 输出摘要 JSON 到 reports/
    """
    fixture_path = Path(__file__).parent / "fixtures" / "both_keys.yml"
    tools_dir = Path(__file__).parent.parent / "tools"
    validate_script = tools_dir / "validate_script.py"
    reports_dir = Path(__file__).parent.parent / "reports"
    summary_path = reports_dir / "validate_config_summary.json"
    
    # 确保 reports 目录存在
    reports_dir.mkdir(exist_ok=True)
    
    # 运行验证（即使失败也应该生成摘要）
    env = os.environ.copy()
    env.pop("ALLOW_LEGACY_KEYS", None)
    
    result = subprocess.run(
        [sys.executable, str(validate_script), "--config", str(fixture_path), "--strict"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=str(Path(__file__).parent.parent)
    )
    
    # 检查是否生成了摘要文件
    assert summary_path.exists(), f"Summary file should exist at {summary_path}"
    
    # 读取并验证摘要内容
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    assert "conflicts_count" in summary
    assert "overall_pass" in summary
    assert "allow_legacy_keys" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

