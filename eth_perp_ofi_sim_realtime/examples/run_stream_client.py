import asyncio, json, websockets
from src.ofi import OnlineOFI

async def main():
    uri='ws://127.0.0.1:8765'
    ofi=OnlineOFI(micro_window_ms=100, z_window_seconds=900)
    async with websockets.connect(uri) as ws:
        print('Connected to', uri)
        async for msg in ws:
            e=json.loads(msg)
            if e.get('type')=='best':
                ofi.on_best(e['t'], e['bid'], e['bid_sz'], e['ask'], e['ask_sz'])
                res=ofi.read()
                if res and abs(res['ofi_z'])>=2.0:
                    print('[OFI]', res)
            elif e.get('type') in ('l2_add','l2_cancel'):
                ofi.on_l2(e['t'], e['type'], e['side'], e['price'], e['qty'])
            elif e.get('type')=='end':
                print('Stream ended.'); break

if __name__=='__main__':
    asyncio.run(main())
