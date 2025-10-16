import numpy as np
import pandas as pd

def gen_signals_v5(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v5 极简信号逻辑 - 基于数据质量诊断结果
    关键发现：所有信号收益都远低于手续费成本，需要极简逻辑
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    
    # 极简逻辑1: 仅使用OFI信号，去掉CVD要求
    ofi_threshold = m.get("ofi_z_min", 0.5)
    ofi_signal = abs(out["ofi_z"]) >= ofi_threshold
    
    # 极简逻辑2: 仅使用价格动量确认
    price_momentum_threshold = m.get("min_ret", 0.0)
    price_momentum_long = out["ret_1s"] > price_momentum_threshold
    price_momentum_short = out["ret_1s"] < -price_momentum_threshold
    
    # 极简逻辑3: 方向一致性检查
    direction_consistent_long = (out["ofi_z"] > 0) & price_momentum_long
    direction_consistent_short = (out["ofi_z"] < 0) & price_momentum_short
    
    # 极简组合信号
    long_mask = ofi_signal & direction_consistent_long
    short_mask = ofi_signal & direction_consistent_short
    
    # 应用信号
    out.loc[long_mask, "sig_type"] = "momentum"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[short_mask, "sig_type"] = "momentum"
    out.loc[short_mask, "sig_side"] = -1
    
    return out

def gen_signals_v5_ultra_simple(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v5 超极简信号逻辑 - 只关注最强的OFI信号
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    
    # 超极简逻辑: 只使用最强的OFI信号
    ofi_threshold = m.get("ofi_z_min", 1.5)  # 使用较高阈值
    
    # 仅基于OFI方向和强度
    strong_long = out["ofi_z"] >= ofi_threshold
    strong_short = out["ofi_z"] <= -ofi_threshold
    
    # 应用信号
    out.loc[strong_long, "sig_type"] = "momentum"
    out.loc[strong_long, "sig_side"] = 1
    out.loc[strong_short, "sig_type"] = "momentum"
    out.loc[strong_short, "sig_side"] = -1
    
    return out

def gen_signals_v5_reversal(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v5 反转信号逻辑 - 基于OFI反转
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    
    # 反转逻辑: OFI与价格方向相反
    ofi_threshold = m.get("ofi_z_min", 0.8)
    
    # 反转信号：OFI与价格动量相反
    reversal_long = (out["ofi_z"] >= ofi_threshold) & (out["ret_1s"] < 0)  # 强买但价格下跌
    reversal_short = (out["ofi_z"] <= -ofi_threshold) & (out["ret_1s"] > 0)  # 强卖但价格上涨
    
    # 应用信号
    out.loc[reversal_long, "sig_type"] = "divergence"
    out.loc[reversal_long, "sig_side"] = 1
    out.loc[reversal_short, "sig_type"] = "divergence"
    out.loc[reversal_short, "sig_side"] = -1
    
    return out

def gen_signals_v5_momentum_breakout(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v5 动量突破信号逻辑 - 基于价格突破 + OFI确认
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    
    # 动量突破逻辑
    price_threshold = m.get("price_break_threshold", 0.0001)  # 0.01%价格突破
    ofi_threshold = m.get("ofi_z_min", 0.5)
    
    # 价格突破 + OFI确认
    price_up = out["ret_1s"] > price_threshold
    price_down = out["ret_1s"] < -price_threshold
    
    ofi_confirm_long = out["ofi_z"] > ofi_threshold
    ofi_confirm_short = out["ofi_z"] < -ofi_threshold
    
    # 突破信号
    breakout_long = price_up & ofi_confirm_long
    breakout_short = price_down & ofi_confirm_short
    
    # 应用信号
    out.loc[breakout_long, "sig_type"] = "momentum"
    out.loc[breakout_long, "sig_side"] = 1
    out.loc[breakout_short, "sig_type"] = "momentum"
    out.loc[breakout_short, "sig_side"] = -1
    
    return out
