
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    trades, summary = run(args.config)
    outdir = "examples/out"
    os.makedirs(outdir, exist_ok=True)
    trades_path = os.path.join(outdir, "trades.csv")
    trades.to_csv(trades_path, index=False)
    print("Summary:", summary)
    print("Trades saved to:", trades_path)

if __name__ == "__main__":
    main()
