#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OFI配置解析器 - 支持分层配置：Global → Profile → Regime → Symbol override
"""

import yaml
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class OFIConfig:
    """OFI配置参数"""
    z_window: int
    ema_alpha: float
    z_clip: Optional[float]
    winsor_k_mad: float
    std_floor: float
    
    def __post_init__(self):
        """验证配置参数"""
        if self.z_window <= 0:
            raise ValueError(f"z_window must be positive, got {self.z_window}")
        if not 0 < self.ema_alpha < 1:
            raise ValueError(f"ema_alpha must be in (0,1), got {self.ema_alpha}")
        if self.winsor_k_mad <= 0:
            raise ValueError(f"winsor_k_mad must be positive, got {self.winsor_k_mad}")
        if self.std_floor <= 0:
            raise ValueError(f"std_floor must be positive, got {self.std_floor}")

@dataclass
class Guardrails:
    """保护机制配置"""
    p_gt2_range: Tuple[float, float]
    p_gt3_max: float
    rollback_minutes: int
    clip_adjust_step: float
    min_z_clip: float
    max_z_clip: Optional[float]

class OFIConfigParser:
    """OFI配置解析器"""
    
    def __init__(self, config_path: str = "config/defaults.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def get_ofi_config(self, 
                      symbol: str, 
                      profile: str = "offline_eval",
                      regime: str = "active") -> OFIConfig:
        """
        获取OFI配置参数
        
        Args:
            symbol: 交易对符号
            profile: 配置profile (offline_eval/online_prod)
            regime: 市场regime (active/quiet)
            
        Returns:
            OFIConfig: OFI配置参数
        """
        # 1. 获取symbol的流动性分类
        symbol_class = self._get_symbol_class(symbol)
        
        # 2. 获取profile配置
        profile_config = self.config['ofi']['profiles'][profile]
        
        # 3. 获取regime配置
        regime_config = profile_config['regimes'][symbol_class][regime]
        
        # 4. 检查是否有symbol override
        override_config = self._get_symbol_override(symbol, profile)
        
        # 5. 合并配置
        config = {
            'z_window': int(regime_config['z_window']),
            'ema_alpha': float(regime_config['ema_alpha']),
            'z_clip': profile_config['z_clip'],
            'winsor_k_mad': float(profile_config['winsor_k_mad']),
            'std_floor': float(profile_config['std_floor'])
        }
        
        # 应用override
        if override_config:
            # 确保override配置的类型正确
            for key, value in override_config.items():
                if key == 'z_window':
                    config[key] = int(value)
                elif key in ['ema_alpha', 'winsor_k_mad', 'std_floor']:
                    config[key] = float(value)
                elif key == 'z_clip':
                    config[key] = value  # 保持原类型（可能是None）
            logger.info(f"Applied symbol override for {symbol}: {override_config}")
        
        return OFIConfig(**config)
    
    def _get_symbol_class(self, symbol: str) -> str:
        """获取symbol的流动性分类"""
        symbol_info = self.config['symbols'].get(symbol)
        if not symbol_info:
            logger.warning(f"Symbol {symbol} not found in config, using high liquidity")
            return "high_liquidity"
        
        symbol_class = symbol_info['class']
        liquidity = self.config['symbol_classes'][symbol_class]['liquidity']
        
        return f"{liquidity}_liquidity"
    
    def _get_symbol_override(self, symbol: str, profile: str) -> Optional[Dict[str, Any]]:
        """获取symbol override配置"""
        overrides = self.config['ofi']['symbol_overrides']
        return overrides.get(symbol, {}).get(profile)
    
    def get_guardrails(self) -> Guardrails:
        """获取保护机制配置"""
        guardrails_config = self.config['ofi']['guardrails']
        return Guardrails(
            p_gt2_range=tuple(guardrails_config['p_gt2_range']),
            p_gt3_max=guardrails_config['p_gt3_max'],
            rollback_minutes=guardrails_config['rollback_minutes'],
            clip_adjust_step=guardrails_config['clip_adjust_step'],
            min_z_clip=guardrails_config['min_z_clip'],
            max_z_clip=guardrails_config['max_z_clip']
        )
    
    def add_symbol_override(self, 
                           symbol: str, 
                           profile: str, 
                           override_config: Dict[str, Any]) -> None:
        """
        添加symbol override配置
        
        Args:
            symbol: 交易对符号
            profile: 配置profile
            override_config: override配置参数
        """
        if 'symbol_overrides' not in self.config['ofi']:
            self.config['ofi']['symbol_overrides'] = {}
        
        if symbol not in self.config['ofi']['symbol_overrides']:
            self.config['ofi']['symbol_overrides'][symbol] = {}
        
        self.config['ofi']['symbol_overrides'][symbol][profile] = override_config
        logger.info(f"Added symbol override for {symbol}/{profile}: {override_config}")
    
    def remove_symbol_override(self, symbol: str, profile: str) -> None:
        """移除symbol override配置"""
        overrides = self.config['ofi']['symbol_overrides']
        if symbol in overrides and profile in overrides[symbol]:
            del overrides[symbol][profile]
            if not overrides[symbol]:  # 如果symbol下没有其他profile，删除整个symbol
                del overrides[symbol]
            logger.info(f"Removed symbol override for {symbol}/{profile}")
    
    def save_config(self) -> None:
        """保存配置文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Saved config to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
            raise
    
    def validate_config(self) -> bool:
        """验证配置完整性"""
        try:
            # 检查必要的配置结构
            required_keys = ['ofi', 'symbols', 'symbol_classes']
            for key in required_keys:
                if key not in self.config:
                    logger.error(f"Missing required config key: {key}")
                    return False
            
            # 检查ofi配置
            ofi_config = self.config['ofi']
            if 'profiles' not in ofi_config:
                logger.error("Missing ofi.profiles config")
                return False
            
            # 检查每个profile
            for profile_name, profile_config in ofi_config['profiles'].items():
                if not self._validate_profile(profile_name, profile_config):
                    return False
            
            # 检查symbols配置
            for symbol, symbol_info in self.config['symbols'].items():
                if 'class' not in symbol_info:
                    logger.error(f"Symbol {symbol} missing class")
                    return False
                
                symbol_class = symbol_info['class']
                if symbol_class not in self.config['symbol_classes']:
                    logger.error(f"Symbol {symbol} class {symbol_class} not found")
                    return False
            
            logger.info("Config validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            return False
    
    def _validate_profile(self, profile_name: str, profile_config: Dict[str, Any]) -> bool:
        """验证profile配置"""
        required_keys = ['z_clip', 'winsor_k_mad', 'std_floor', 'regimes']
        for key in required_keys:
            if key not in profile_config:
                logger.error(f"Profile {profile_name} missing {key}")
                return False
        
        # 检查regimes配置
        regimes = profile_config['regimes']
        for liquidity in ['high_liquidity', 'low_liquidity']:
            if liquidity not in regimes:
                logger.error(f"Profile {profile_name} missing {liquidity} regime")
                return False
            
            for regime_type in ['active', 'quiet']:
                if regime_type not in regimes[liquidity]:
                    logger.error(f"Profile {profile_name} missing {liquidity}.{regime_type}")
                    return False
                
                regime_config = regimes[liquidity][regime_type]
                if 'z_window' not in regime_config or 'ema_alpha' not in regime_config:
                    logger.error(f"Profile {profile_name}.{liquidity}.{regime_type} missing z_window or ema_alpha")
                    return False
        
        return True

def test_config_parser():
    """测试配置解析器"""
    parser = OFIConfigParser()
    
    # 验证配置
    if not parser.validate_config():
        print("Config validation failed!")
        return
    
    print("Config validation passed!")
    
    # 测试获取配置
    test_cases = [
        ("BTCUSDT", "offline_eval", "active"),
        ("BTCUSDT", "online_prod", "active"),
        ("XRPUSDT", "offline_eval", "quiet"),
        ("XRPUSDT", "online_prod", "quiet"),
    ]
    
    for symbol, profile, regime in test_cases:
        try:
            config = parser.get_ofi_config(symbol, profile, regime)
            print(f"{symbol}/{profile}/{regime}: z_window={config.z_window}, ema_alpha={config.ema_alpha}, z_clip={config.z_clip}")
        except Exception as e:
            print(f"Failed to get config for {symbol}/{profile}/{regime}: {e}")
    
    # 测试保护机制
    guardrails = parser.get_guardrails()
    print(f"Guardrails: p_gt2_range={guardrails.p_gt2_range}, rollback_minutes={guardrails.rollback_minutes}")

if __name__ == "__main__":
    test_config_parser()
