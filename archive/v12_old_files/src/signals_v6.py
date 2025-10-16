import numpy as np
import pandas as pd

def gen_signals_v6_quality(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v6 信号质量提升版 - 基于v5突破结果进一步优化
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    
    # 1. 增加信号强度要求
    signal_strength = abs(out["ofi_z"])
    min_signal_strength = m.get("min_signal_strength", 2.0)
    strong_signal = signal_strength >= min_signal_strength
    
    # 2. 增加价格动量确认
    price_momentum_threshold = m.get("price_momentum_threshold", 0.00005)  # 0.005%最小价格变动
    price_momentum_long = out["ret_1s"] > price_momentum_threshold
    price_momentum_short = out["ret_1s"] < -price_momentum_threshold
    
    # 3. 方向一致性检查
    direction_consistent_long = (out["ofi_z"] > 0) & price_momentum_long
    direction_consistent_short = (out["ofi_z"] < 0) & price_momentum_short
    
    # 4. 组合信号 - 更严格的条件
    ofi_threshold = m.get("ofi_z_min", 2.0)
    long_mask = (out["ofi_z"] >= ofi_threshold) & strong_signal & direction_consistent_long
    short_mask = (out["ofi_z"] <= -ofi_threshold) & strong_signal & direction_consistent_short
    
    # 应用信号并记录信号强度
    out.loc[long_mask, "sig_type"] = "momentum"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[long_mask, "signal_strength"] = signal_strength[long_mask]
    
    out.loc[short_mask, "sig_type"] = "momentum"
    out.loc[short_mask, "sig_side"] = -1
    out.loc[short_mask, "signal_strength"] = signal_strength[short_mask]
    
    return out

def gen_signals_v6_momentum_enhanced(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v6 动量增强版 - 基于价格动量 + OFI确认
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    
    # 1. 价格动量要求
    price_break_threshold = m.get("price_break_threshold", 0.0001)  # 0.01%价格突破
    strong_price_up = out["ret_1s"] > price_break_threshold
    strong_price_down = out["ret_1s"] < -price_break_threshold
    
    # 2. OFI确认要求
    ofi_confirm_threshold = m.get("ofi_z_min", 1.0)
    ofi_confirm_long = out["ofi_z"] > ofi_confirm_threshold
    ofi_confirm_short = out["ofi_z"] < -ofi_confirm_threshold
    
    # 3. 信号强度计算
    signal_strength = (abs(out["ofi_z"]) + abs(out["ret_1s"]) * 10000) / 2
    min_signal_strength = m.get("min_signal_strength", 1.5)
    strong_signal = signal_strength >= min_signal_strength
    
    # 4. 组合信号
    long_mask = strong_price_up & ofi_confirm_long & strong_signal
    short_mask = strong_price_down & ofi_confirm_short & strong_signal
    
    # 应用信号
    out.loc[long_mask, "sig_type"] = "momentum"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[long_mask, "signal_strength"] = signal_strength[long_mask]
    
    out.loc[short_mask, "sig_type"] = "momentum"
    out.loc[short_mask, "sig_side"] = -1
    out.loc[short_mask, "signal_strength"] = signal_strength[short_mask]
    
    return out

def gen_signals_v6_reversal_enhanced(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v6 反转增强版 - 基于OFI与价格方向相反的反转信号
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    
    # 1. OFI强度要求
    ofi_threshold = m.get("ofi_z_min", 1.2)
    strong_ofi_long = out["ofi_z"] >= ofi_threshold
    strong_ofi_short = out["ofi_z"] <= -ofi_threshold
    
    # 2. 价格方向相反（反转条件）
    price_reversal_long = out["ret_1s"] < 0  # 强买但价格下跌
    price_reversal_short = out["ret_1s"] > 0  # 强卖但价格上涨
    
    # 3. 反转强度计算
    reversal_strength = abs(out["ofi_z"]) * abs(out["ret_1s"]) * 10000
    min_reversal_strength = m.get("min_reversal_strength", 0.5)
    strong_reversal = reversal_strength >= min_reversal_strength
    
    # 4. 组合信号
    long_mask = strong_ofi_long & price_reversal_long & strong_reversal
    short_mask = strong_ofi_short & price_reversal_short & strong_reversal
    
    # 应用信号
    out.loc[long_mask, "sig_type"] = "divergence"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[long_mask, "signal_strength"] = reversal_strength[long_mask]
    
    out.loc[short_mask, "sig_type"] = "divergence"
    out.loc[short_mask, "sig_side"] = -1
    out.loc[short_mask, "signal_strength"] = reversal_strength[short_mask]
    
    return out

def gen_signals_v6_adaptive(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v6 自适应版 - 根据市场条件动态调整信号逻辑
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    
    # 1. 市场状态识别
    price_volatility = out["ret_1s"].rolling(20).std()
    ofi_volatility = out["ofi_z"].rolling(20).std()
    
    # 高波动率市场
    high_vol = (price_volatility > price_volatility.quantile(0.7)) | \
               (ofi_volatility > ofi_volatility.quantile(0.7))
    
    # 2. 自适应阈值
    base_ofi_threshold = m.get("ofi_z_min", 1.5)
    # 使用numpy.where处理Series布尔值
    adaptive_threshold = base_ofi_threshold * np.where(high_vol, 1.5, 1.0)
    
    # 3. 信号生成
    strong_long = out["ofi_z"] >= adaptive_threshold
    strong_short = out["ofi_z"] <= -adaptive_threshold
    
    # 4. 在高波动率市场增加价格动量确认
    price_momentum = abs(out["ret_1s"]) > 0.00005
    # 在高波动率时增加价格动量确认
    strong_long = strong_long & (~high_vol | (price_momentum & (out["ret_1s"] > 0)))
    strong_short = strong_short & (~high_vol | (price_momentum & (out["ret_1s"] < 0)))
    
    # 5. 信号强度
    signal_strength = abs(out["ofi_z"])
    
    # 应用信号
    out.loc[strong_long, "sig_type"] = "momentum"
    out.loc[strong_long, "sig_side"] = 1
    out.loc[strong_long, "signal_strength"] = signal_strength[strong_long]
    
    out.loc[strong_short, "sig_type"] = "momentum"
    out.loc[strong_short, "sig_side"] = -1
    out.loc[strong_short, "signal_strength"] = signal_strength[strong_short]
    
    return out
