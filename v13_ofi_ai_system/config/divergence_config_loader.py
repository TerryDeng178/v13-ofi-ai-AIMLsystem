
# -*- coding: utf-8 -*-
"""
DivergenceConfigLoader
把统一配置（divergence_detection）解析为背离检测的强类型配置对象。
"""

from dataclasses import dataclass
from typing import Any

@dataclass
class HysteresisConfig:
    cooldown_secs: float = 1.0

@dataclass
class DivergenceDetectionConfig:
    swing_L: int = 12
    ema_k: int = 5
    z_hi: float = 1.5
    z_mid: float = 0.7
    min_separation: int = 6
    cooldown_secs: float = 1.0
    warmup_min: int = 100
    max_lag: float = 0.30
    use_fusion: bool = True

class DivergenceConfigLoader:
    def __init__(self, unified_loader):
        self.cfg = unified_loader

    def load_config(self) -> DivergenceDetectionConfig:
        raw = self.cfg.get("divergence_detection.default", {}) or {}
        return DivergenceDetectionConfig(
            swing_L=int(raw.get("swing_L", 12)),
            ema_k=int(raw.get("ema_k", 5)),
            z_hi=float(raw.get("z_hi", 1.5)),
            z_mid=float(raw.get("z_mid", 0.7)),
            min_separation=int(raw.get("min_separation", 6)),
            cooldown_secs=float(raw.get("cooldown_secs", 1.0)),
            warmup_min=int(raw.get("warmup_min", 100)),
            max_lag=float(raw.get("max_lag", 0.30)),
            use_fusion=bool(raw.get("use_fusion", True)),
        )
