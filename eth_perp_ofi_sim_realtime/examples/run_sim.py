import argparse, yaml, os
import pandas as pd
from src.sim import MarketSimulator

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); args=ap.parse_args()
    with open(args.config,'r') as f: params=yaml.safe_load(f)
    sim=MarketSimulator(params)
    events=[]
    for evts in sim.stream(realtime=False, dt_ms=10):
        events.extend(evts)
    df=pd.DataFrame(events)
    out='examples/out'; os.makedirs(out, exist_ok=True)
    df.to_csv(f'{out}/events.csv', index=False)
    print('Saved', len(df), 'events to', out)

if __name__=='__main__': main()
