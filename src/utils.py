
import numpy as np
import pandas as pd

def rolling_z(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window, min_periods=window//2).mean()
    s = x.rolling(window, min_periods=window//2).std(ddof=0)
    z = (x - m) / (s.replace(0, np.nan))
    return z.fillna(0.0)

def ewma(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def vwap(price, volume):
    pv = (price * volume).cumsum()
    vv = volume.cumsum()
    return pv / vv.replace(0, np.nan)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))
