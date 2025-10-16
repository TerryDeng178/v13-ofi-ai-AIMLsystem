
import pandas as pd
from datetime import datetime, timezone

REQUIRED_COLS = [
    "ts","price","size","side",
    "bid1","bid1_size","ask1","ask1_size",
    "oi"
]

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    # parse ts to datetime
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df

def resample_to_seconds(df: pd.DataFrame, seconds:int=1) -> pd.DataFrame:
    # Assumes df contains 1s data already; placeholder for completeness.
    return df
