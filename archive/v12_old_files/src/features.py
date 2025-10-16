
import numpy as np
import pandas as pd
try:
    from .utils import rolling_z
except ImportError:
    # For testing when running as standalone
    from utils import rolling_z

def compute_mid(df: pd.DataFrame):
    return (df["bid1"] + df["ask1"]) / 2.0

def compute_returns(mid: pd.Series, window:int=1):
    return mid.pct_change(periods=window).fillna(0.0)

def compute_cvd(df: pd.DataFrame, mid_prev: pd.Series) -> pd.Series:
    # side: 1 = aggressive buy, -1 = aggressive sell, 0 unknown
    delta = np.sign(df["price"].values - mid_prev.values) * df["size"].values
    cvd = pd.Series(delta, index=df.index).cumsum()
    return cvd

def compute_ofi(df: pd.DataFrame, window_seconds:int=1, levels:int=5) -> pd.Series:
    """
    Compute Order Flow Imbalance using top N levels with weighted approach.
    OFI = Σ(Δbid_size * I_bid_improve) - Σ(Δask_size * I_ask_improve)
    where weights decrease with level depth.
    """
    ofi_total = pd.Series(0.0, index=df.index)
    
    for level in range(1, levels + 1):
        # Get level columns (assuming they exist in the data)
        bid_col = f"bid{level}" if f"bid{level}" in df.columns else "bid1"
        ask_col = f"ask{level}" if f"ask{level}" in df.columns else "ask1"
        bid_size_col = f"bid{level}_size" if f"bid{level}_size" in df.columns else "bid1_size"
        ask_size_col = f"ask{level}_size" if f"ask{level}_size" in df.columns else "ask1_size"
        
        # Weight decreases with level depth (more weight to top levels)
        weight = 1.0 / level
        
        # Check for bid/ask improvements
        bid_up = (df[bid_col] > df[bid_col].shift(1)).fillna(False)
        ask_up = (df[ask_col] > df[ask_col].shift(1)).fillna(False)
        
        # Compute size changes
        delta_bid = (df[bid_size_col] - df[bid_size_col].shift(1)).fillna(0.0)
        delta_ask = (df[ask_size_col] - df[ask_size_col].shift(1)).fillna(0.0)
        
        # OFI contribution for this level
        ofi_level = weight * (delta_bid.where(bid_up, 0.0) - delta_ask.where(ask_up, 0.0))
        ofi_total += ofi_level
    
    # Rolling sum over window_seconds
    ofi_roll = ofi_total.rolling(window_seconds, min_periods=1).sum()
    return ofi_roll.fillna(0.0)

def compute_bp(df: pd.DataFrame) -> pd.Series:
    num = df["bid1_size"] - df["ask1_size"]
    den = (df["bid1_size"] + df["ask1_size"]).replace(0, np.nan)
    return (num / den).fillna(0.0)

def compute_vwap(df: pd.DataFrame, window_seconds:int=900) -> pd.Series:
    pv = (df["price"] * df["size"]).rolling(window_seconds, min_periods=10).sum()
    vv = (df["size"]).rolling(window_seconds, min_periods=10).sum()
    return (pv / vv.replace(0, np.nan)).ffill().fillna(df["price"])

def compute_atr(df: pd.DataFrame, window:int=14) -> pd.Series:
    # approximate 1s ATR using synthetic high/low from best quotes
    high = df[["price","ask1"]].max(axis=1)
    low  = df[["price","bid1"]].min(axis=1)
    prev_close = df["price"].shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=max(2,window//2)).mean()
    return atr.bfill()

def add_feature_block(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    mid = compute_mid(out)
    mid_prev = mid.shift(1).bfill()
    out["ret_1s"] = compute_returns(mid, 1)
    out["cvd"] = compute_cvd(out, mid_prev)
    out["ofi"] = compute_ofi(out, params["features"]["ofi_window_seconds"], params["features"]["ofi_levels"])
    out["bp"]  = compute_bp(out)
    out["vwap"] = compute_vwap(out, params["features"]["vwap_window_seconds"])
    out["atr"] = compute_atr(out, params["features"]["atr_window"])
    # z-scores
    zwin = params["features"]["z_window"]
    out["cvd_z"] = rolling_z(out["cvd"], zwin)
    out["ofi_z"] = rolling_z(out["ofi"], zwin)
    return out
