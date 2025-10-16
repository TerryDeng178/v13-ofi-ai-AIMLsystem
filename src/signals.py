
import numpy as np
import pandas as pd

def thin_book(dfrow) -> bool:
    # placeholder: thin if spread in bps < threshold & volume reasonably high
    price = dfrow["price"]
    spread = (dfrow["ask1"] - dfrow["bid1"]) / price * 1e4  # bps
    vol_ratio = dfrow["size"] / (dfrow["bid1_size"] + dfrow["ask1_size"] + 1e-9)
    # treat "thin" as small spread but decent matching volume
    return (spread <= dfrow.get("thin_spread_bps_max", 2.0)) and (vol_ratio > 0.01)

def compute_adaptive_thresholds(df: pd.DataFrame, window: int, quantile_hi: float, quantile_lo: float) -> tuple:
    """
    Compute adaptive thresholds using rolling quantiles.
    Returns (threshold_hi, threshold_lo) for the given quantiles.
    """
    threshold_hi = df.rolling(window, min_periods=window//4).quantile(quantile_hi)
    threshold_lo = df.rolling(window, min_periods=window//4).quantile(quantile_lo)
    return threshold_hi.ffill(), threshold_lo.ffill()

def gen_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    p = params["signals"]
    m = p["momentum"]
    d = p["divergence"]

    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0  # +1 long, -1 short

    # Momentum: 降低连续确认要求，从2根改为1根 (v4 信号重构)
    # 增加价格动量确认
    price_momentum_long = out["ret_1s"] > m.get("min_ret", 0.0)
    price_momentum_short = out["ret_1s"] < -m.get("min_ret", 0.0)
    
    # 引入信号强度评分
    signal_strength = (abs(out["ofi_z"]) + abs(out["cvd_z"])) / 2
    min_signal_strength = m.get("min_signal_strength", 1.0)
    strong_signal = signal_strength >= min_signal_strength
    
    # 简化动量信号逻辑：单根确认 + 信号强度 + 价格动量
    long_mask = (out["ofi_z"] >= m["ofi_z_min"]) & (out["cvd_z"] >= m["cvd_z_min"]) & price_momentum_long & strong_signal
    short_mask = (out["ofi_z"] <= -m["ofi_z_min"]) & (out["cvd_z"] <= -m["cvd_z_min"]) & price_momentum_short & strong_signal

    out.loc[long_mask, "sig_type"] = "momentum"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[short_mask, "sig_type"] = "momentum"
    out.loc[short_mask, "sig_side"] = -1

    # Divergence: 两段式收复确认 (v3)
    hh = out["price"].rolling(60, min_periods=30).max()
    ll = out["price"].rolling(60, min_periods=30).min()
    new_high = out["price"] >= hh
    new_low  = out["price"] <= ll

    reclaim_bars = d.get("reclaim_bars", 1)
    reclaim_high = new_high & (out["price"].shift(-reclaim_bars) < hh)
    reclaim_low  = new_low  & (out["price"].shift(-reclaim_bars) > ll)

    div_short = reclaim_high & ((out["cvd_z"] <= 0.0) | (out["ofi_z"] <= 0.0))
    div_long  = reclaim_low  & ((out["cvd_z"] <= 0.0) | (out["ofi_z"] <= 0.0))

    out.loc[div_long, "sig_type"] = "divergence"
    out.loc[div_long, "sig_side"] = 1
    out.loc[div_short, "sig_type"] = "divergence"
    out.loc[div_short, "sig_side"] = -1

    return out
