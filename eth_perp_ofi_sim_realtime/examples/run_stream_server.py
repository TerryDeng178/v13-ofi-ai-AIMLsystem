import argparse, yaml, asyncio
from src.sim import MarketSimulator
from src.stream import WSHub

async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--realtime', default='true')
    ap.add_argument('--dt_ms', type=int, default=10)
    args=ap.parse_args()
    with open(args.config, 'r') as f: params=yaml.safe_load(f)
    hub=WSHub(); sim=MarketSimulator(params)
    async def producer():
        for evts in sim.stream(realtime=(args.realtime.lower()=='true'), dt_ms=args.dt_ms):
            for e in evts: await hub.broadcast(e)
        await hub.broadcast({"type":"end"})
    await asyncio.gather(hub.serve(), producer())

if __name__=='__main__':
    asyncio.run(main())
