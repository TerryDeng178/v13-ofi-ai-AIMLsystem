#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置来源链打印工具
打印关键配置键的来源链和配置指纹
"""

import sys
import hashlib
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.unified_config_loader import UnifiedConfigLoader


def get_config_fingerprint(config: dict) -> str:
    """计算配置指纹（带 hex 校验）"""
    import re
    
    key_fields = {
        "logging.level": config.get("logging", {}).get("level"),
        "data_source.default_symbol": config.get("data_source", {}).get("default_symbol"),
        "monitoring.enabled": config.get("monitoring", {}).get("enabled"),
        "system.version": config.get("system", {}).get("version"),
        "fusion_metrics.thresholds.fuse_buy": config.get("fusion_metrics", {}).get("thresholds", {}).get("fuse_buy"),
        "strategy_mode.triggers.market.min_trades_per_min": config.get("strategy_mode", {}).get("triggers", {}).get("market", {}).get("min_trades_per_min"),
    }
    config_str = json.dumps(key_fields, sort_keys=True)
    fingerprint = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]
    
    # Hex 校验：确保只包含 [0-9a-f] 字符
    hex_pattern = re.compile(r'^[0-9a-f]*$')
    if not hex_pattern.match(fingerprint):
        print(f"[WARN] Invalid hex fingerprint detected: {fingerprint}, cleaning...", file=sys.stderr)
        fingerprint_cleaned = ''.join(c for c in fingerprint if c in '0123456789abcdef')
        if len(fingerprint_cleaned) < 16:
            fingerprint_cleaned = fingerprint_cleaned.ljust(16, '0')
        fingerprint = fingerprint_cleaned
    
    return fingerprint


def print_config_origin():
    """打印关键配置键的来源和指纹"""
    loader = UnifiedConfigLoader()
    config = loader.get()
    
    print("=" * 60)
    print("配置来源链与指纹")
    print("=" * 60)
    
    # 关键配置键
    key_configs = [
        ("logging.level", "日志级别"),
        ("data_source.default_symbol", "默认交易对"),
        ("fusion_metrics.thresholds.fuse_buy", "Fusion买入阈值"),
        ("strategy_mode.triggers.market.min_trades_per_min", "策略最小交易数阈值"),
    ]
    
    print("\n[关键配置键来源]")
    for key_path, description in key_configs:
        value = loader.get(key_path, "NOT_FOUND")
        print(f"  {description}:")
        print(f"    路径: {key_path}")
        print(f"    值: {value}")
        print(f"    来源: system.yaml (通过配置加载器合并后)")
    
    # 配置指纹
    fingerprint = get_config_fingerprint(config)
    print("\n[配置指纹]")
    print(f"  指纹: {fingerprint}")
    print(f"  用途: 用于跨进程/跨组件一致性验证")
    # 输出单独一行便于 grep
    print(f"\nCONFIG_FINGERPRINT={fingerprint}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_config_origin()

