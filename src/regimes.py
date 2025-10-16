
import pandas as pd
import numpy as np

def classify_regime(df: pd.DataFrame) -> pd.Series:
    # simple regime: trend if |ret_1s| rolling mean > threshold, else chop
    vol = df["ret_1s"].rolling(120, min_periods=60).std(ddof=0).fillna(0.0)
    mom = df["ret_1s"].rolling(120, min_periods=60).mean().fillna(0.0)
    trend = (abs(mom) > (vol * 0.1)) & (vol > vol.quantile(0.3))
    highv = vol > vol.quantile(0.7)
    regime = pd.Series("chop", index=df.index)
    regime[trend] = "trend"
    regime[highv & ~trend] = "highvol"
    return regime
