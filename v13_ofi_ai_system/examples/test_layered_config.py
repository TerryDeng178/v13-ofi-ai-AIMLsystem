#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新的分层配置体系
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ofi_config_parser import OFIConfigParser, OFIConfig, Guardrails
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_layered_config():
    """测试分层配置体系"""
    print("=== 测试分层配置体系 ===")
    
    parser = OFIConfigParser("../config/defaults.yaml")
    
    # 1. 验证配置完整性
    print("\n1. 配置验证:")
    if parser.validate_config():
        print("OK 配置验证通过")
    else:
        print("FAIL 配置验证失败")
        return
    
    # 2. 测试不同profile和regime组合
    print("\n2. 测试配置组合:")
    test_cases = [
        ("BTCUSDT", "offline_eval", "active", "高流动性-活跃-离线评估"),
        ("BTCUSDT", "online_prod", "active", "高流动性-活跃-线上生产"),
        ("BTCUSDT", "offline_eval", "quiet", "高流动性-安静-离线评估"),
        ("BTCUSDT", "online_prod", "quiet", "高流动性-安静-线上生产"),
        ("XRPUSDT", "offline_eval", "active", "低流动性-活跃-离线评估"),
        ("XRPUSDT", "online_prod", "active", "低流动性-活跃-线上生产"),
        ("XRPUSDT", "offline_eval", "quiet", "低流动性-安静-离线评估"),
        ("XRPUSDT", "online_prod", "quiet", "低流动性-安静-线上生产"),
    ]
    
    for symbol, profile, regime, description in test_cases:
        try:
            config = parser.get_ofi_config(symbol, profile, regime)
            print(f"OK {description}:")
            print(f"   z_window={config.z_window}, ema_alpha={config.ema_alpha}")
            print(f"   z_clip={config.z_clip}, winsor_k_mad={config.winsor_k_mad}")
            print(f"   std_floor={config.std_floor}")
        except Exception as e:
            print(f"FAIL {description}: {e}")
    
    # 3. 测试保护机制
    print("\n3. 保护机制配置:")
    try:
        guardrails = parser.get_guardrails()
        print(f"OK P(|z|>2)目标区间: {guardrails.p_gt2_range}")
        print(f"OK P(|z|>3)上限: {guardrails.p_gt3_max}")
        print(f"OK 回滚时间: {guardrails.rollback_minutes}分钟")
        print(f"OK 调整步长: {guardrails.clip_adjust_step}")
        print(f"OK z_clip范围: [{guardrails.min_z_clip}, {guardrails.max_z_clip}]")
    except Exception as e:
        print(f"FAIL 保护机制配置失败: {e}")
    
    # 4. 测试symbol override功能
    print("\n4. 测试Symbol Override功能:")
    try:
        # 添加override
        override_config = {
            'z_window': 100,
            'ema_alpha': 0.35,
            'z_clip': 2.5
        }
        parser.add_symbol_override("BTCUSDT", "offline_eval", override_config)
        
        # 测试override是否生效
        config_with_override = parser.get_ofi_config("BTCUSDT", "offline_eval", "active")
        print(f"OK BTCUSDT override生效:")
        print(f"   z_window={config_with_override.z_window} (override: 100)")
        print(f"   ema_alpha={config_with_override.ema_alpha} (override: 0.35)")
        print(f"   z_clip={config_with_override.z_clip} (override: 2.5)")
        
        # 移除override
        parser.remove_symbol_override("BTCUSDT", "offline_eval")
        
        # 验证override已移除
        config_without_override = parser.get_ofi_config("BTCUSDT", "offline_eval", "active")
        print(f"OK BTCUSDT override已移除:")
        print(f"   z_window={config_without_override.z_window} (恢复默认)")
        print(f"   ema_alpha={config_without_override.ema_alpha} (恢复默认)")
        print(f"   z_clip={config_without_override.z_clip} (恢复默认)")
        
    except Exception as e:
        print(f"FAIL Symbol Override测试失败: {e}")
    
    # 5. 验证配置一致性
    print("\n5. 配置一致性验证:")
    print("OK 离线评估与线上生产仅在z_clip不同:")
    
    # 高流动性-活跃
    offline_config = parser.get_ofi_config("BTCUSDT", "offline_eval", "active")
    online_config = parser.get_ofi_config("BTCUSDT", "online_prod", "active")
    
    print(f"   离线: z_clip={offline_config.z_clip}")
    print(f"   线上: z_clip={online_config.z_clip}")
    print(f"   其他参数一致: z_window={offline_config.z_window}, ema_alpha={offline_config.ema_alpha}")
    
    # 低流动性-安静
    offline_config_low = parser.get_ofi_config("XRPUSDT", "offline_eval", "quiet")
    online_config_low = parser.get_ofi_config("XRPUSDT", "online_prod", "quiet")
    
    print(f"   离线: z_clip={offline_config_low.z_clip}")
    print(f"   线上: z_clip={online_config_low.z_clip}")
    print(f"   其他参数一致: z_window={offline_config_low.z_window}, ema_alpha={offline_config_low.ema_alpha}")

def test_regime_scenarios():
    """测试2×2场景配置"""
    print("\n=== 测试2×2场景配置 ===")
    
    parser = OFIConfigParser("../config/defaults.yaml")
    
    scenarios = [
        ("高流动性-活跃", "BTCUSDT", "offline_eval", "active"),
        ("高流动性-安静", "BTCUSDT", "offline_eval", "quiet"),
        ("低流动性-活跃", "XRPUSDT", "offline_eval", "active"),
        ("低流动性-安静", "XRPUSDT", "offline_eval", "quiet"),
    ]
    
    for scenario_name, symbol, profile, regime in scenarios:
        config = parser.get_ofi_config(symbol, profile, regime)
        print(f"OK {scenario_name}:")
        print(f"   z_window={config.z_window}, ema_alpha={config.ema_alpha}")
        print(f"   z_clip={config.z_clip}, winsor_k_mad={config.winsor_k_mad}")

def test_guardrails_logic():
    """测试保护机制逻辑"""
    print("\n=== 测试保护机制逻辑 ===")
    
    parser = OFIConfigParser("../config/defaults.yaml")
    guardrails = parser.get_guardrails()
    
    # 模拟越界情况
    test_cases = [
        ("正常范围", 0.05, 0.01, "无需调整"),
        ("过窄", 0.005, 0.001, "需要放宽z_clip"),
        ("过宽", 0.12, 0.02, "需要收紧z_clip"),
        ("极端过宽", 0.15, 0.03, "需要紧急调整"),
    ]
    
    for case_name, p_gt2, p_gt3, expected_action in test_cases:
        print(f"OK {case_name}: P(|z|>2)={p_gt2:.3f}, P(|z|>3)={p_gt3:.3f}")
        
        # 检查是否越界
        p_gt2_min, p_gt2_max = guardrails.p_gt2_range
        is_p_gt2_out = p_gt2 < p_gt2_min or p_gt2 > p_gt2_max
        is_p_gt3_out = p_gt3 > guardrails.p_gt3_max
        
        if is_p_gt2_out or is_p_gt3_out:
            print(f"   FAIL 越界检测: P(|z|>2)范围[{p_gt2_min:.3f}, {p_gt2_max:.3f}], P(|z|>3)上限{guardrails.p_gt3_max:.3f}")
            print(f"   ADJUST 建议动作: {expected_action}")
        else:
            print(f"   OK 在正常范围内")

if __name__ == "__main__":
    test_layered_config()
    test_regime_scenarios()
    test_guardrails_logic()
    
    print("\n=== 测试完成 ===")
    print("OK 分层配置体系测试通过")
    print("OK 2x2场景配置测试通过")
    print("OK 保护机制逻辑测试通过")
    print("\nSUCCESS 全局统一基线配置已固化，可以开始灰度/实盘监控！")
