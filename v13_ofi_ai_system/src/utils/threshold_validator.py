#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值范围断言（业务逻辑层）
用于在配置加载后、使用前验证阈值范围
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def assert_fusion_thresholds(thresholds: Dict[str, Any]) -> bool:
    """
    断言融合阈值范围（业务逻辑层验证）
    
    Args:
        thresholds: 阈值字典，包含 fuse_buy, fuse_sell, fuse_strong_buy, fuse_strong_sell
    
    Returns:
        True if valid
    
    Raises:
        AssertionError: if thresholds fail validation
    """
    # 提取阈值
    fuse_buy = thresholds.get("fuse_buy")
    fuse_sell = thresholds.get("fuse_sell")
    fuse_strong_buy = thresholds.get("fuse_strong_buy")
    fuse_strong_sell = thresholds.get("fuse_strong_sell")
    
    # 断言1: 买入阈值必须为正
    assert fuse_buy > 0, f"fuse_buy must be positive, got {fuse_buy}"
    
    # 断言2: 卖出阈值必须为负
    assert fuse_sell < 0, f"fuse_sell must be negative, got {fuse_sell}"
    
    # 断言3: 强买入阈值必须大于普通买入阈值
    assert fuse_strong_buy > fuse_buy, f"fuse_strong_buy ({fuse_strong_buy}) must be > fuse_buy ({fuse_buy})"
    
    # 断言4: 强卖出阈值必须小于普通卖出阈值（绝对值更大）
    assert fuse_strong_sell < fuse_sell, f"fuse_strong_sell ({fuse_strong_sell}) must be < fuse_sell ({fuse_sell})"
    
    # 断言5: 阈值范围合理性（防止过大/过小）
    assert 0.1 <= fuse_buy <= 10.0, f"fuse_buy ({fuse_buy}) should be in [0.1, 10.0]"
    assert -10.0 <= fuse_sell <= -0.1, f"fuse_sell ({fuse_sell}) should be in [-10.0, -0.1]"
    
    logger.info(f"[THRESHOLD_VALIDATION] Fusion thresholds validated successfully")
    return True


def assert_strategy_thresholds(thresholds: Dict[str, Any]) -> bool:
    """
    断言策略阈值范围（业务逻辑层验证）
    
    Args:
        thresholds: 阈值字典
    
    Returns:
        True if valid
    
    Raises:
        AssertionError: if thresholds fail validation
    """
    # 如果存在 min_trades_per_min
    if "min_trades_per_min" in thresholds:
        min_trades = thresholds["min_trades_per_min"]
        assert min_trades > 0, f"min_trades_per_min must be positive, got {min_trades}"
        assert min_trades <= 10000, f"min_trades_per_min ({min_trades}) should be <= 10000"
    
    # 如果存在 min_quote_updates_per_sec
    if "min_quote_updates_per_sec" in thresholds:
        min_updates = thresholds["min_quote_updates_per_sec"]
        assert min_updates > 0, f"min_quote_updates_per_sec must be positive, got {min_updates}"
        assert min_updates <= 1000, f"min_quote_updates_per_sec ({min_updates}) should be <= 1000"
    
    logger.info(f"[THRESHOLD_VALIDATION] Strategy thresholds validated successfully")
    return True


def validate_config_thresholds(config: Dict[str, Any]) -> bool:
    """
    验证配置中的所有阈值
    
    Args:
        config: 完整配置字典
    
    Returns:
        True if all thresholds are valid
    
    Raises:
        AssertionError: if any threshold fails validation
    """
    # 验证 fusion_metrics 阈值
    fusion_metrics = config.get("fusion_metrics", {})
    if "thresholds" in fusion_metrics:
        assert_fusion_thresholds(fusion_metrics["thresholds"])
    
    # 验证 strategy_mode 阈值
    strategy_mode = config.get("strategy_mode", {})
    if "triggers" in strategy_mode and "market" in strategy_mode["triggers"]:
        market_trigger = strategy_mode["triggers"]["market"]
        assert_strategy_thresholds(market_trigger)
    
    logger.info("[THRESHOLD_VALIDATION] All thresholds validated successfully")
    return True

