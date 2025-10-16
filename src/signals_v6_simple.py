import numpy as np
import pandas as pd

def gen_signals_v6_ultra_simple(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v6 超极简版 - 基于v5成功逻辑，只做最小调整
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    
    # 完全基于v5成功逻辑：只使用最强的OFI信号
    ofi_threshold = m.get("ofi_z_min", 1.5)  # 使用v5成功阈值
    
    # 仅基于OFI方向和强度
    strong_long = out["ofi_z"] >= ofi_threshold
    strong_short = out["ofi_z"] <= -ofi_threshold
    
    # 应用信号并记录信号强度
    out.loc[strong_long, "sig_type"] = "momentum"
    out.loc[strong_long, "sig_side"] = 1
    out.loc[strong_long, "signal_strength"] = abs(out["ofi_z"][strong_long])
    
    out.loc[strong_short, "sig_type"] = "momentum"
    out.loc[strong_short, "sig_side"] = -1
    out.loc[strong_short, "signal_strength"] = abs(out["ofi_z"][strong_short])
    
    return out

def gen_signals_v6_minimal_enhancement(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v6 最小增强版 - 在v5基础上添加最小的价格动量确认
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    
    # 基础OFI信号（v5成功逻辑）
    ofi_threshold = m.get("ofi_z_min", 1.5)
    ofi_signal = abs(out["ofi_z"]) >= ofi_threshold
    
    # 最小价格动量确认（可选）
    price_momentum_threshold = m.get("price_momentum_threshold", 0.00001)
    price_momentum_long = out["ret_1s"] > price_momentum_threshold
    price_momentum_short = out["ret_1s"] < -price_momentum_threshold
    
    # 方向一致性（最小要求）
    direction_consistent_long = (out["ofi_z"] > 0) & price_momentum_long
    direction_consistent_short = (out["ofi_z"] < 0) & price_momentum_short
    
    # 组合信号（最小增强）
    long_mask = ofi_signal & direction_consistent_long
    short_mask = ofi_signal & direction_consistent_short
    
    # 应用信号
    out.loc[long_mask, "sig_type"] = "momentum"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[long_mask, "signal_strength"] = abs(out["ofi_z"][long_mask])
    
    out.loc[short_mask, "sig_type"] = "momentum"
    out.loc[short_mask, "sig_side"] = -1
    out.loc[short_mask, "signal_strength"] = abs(out["ofi_z"][short_mask])
    
    return out
