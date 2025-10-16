import asyncio, json, websockets

class WSHub:
    def __init__(self, host='127.0.0.1', port=8765):
        self.host=host; self.port=port; self.clients=set()

    async def handler(self, ws):
        self.clients.add(ws)
        try:
            async for _ in ws: pass
        finally:
            self.clients.remove(ws)

    async def broadcast(self, msg: dict):
        if not self.clients: return
        data=json.dumps(msg)
        await asyncio.gather(*[c.send(data) for c in list(self.clients)], return_exceptions=True)

    async def serve(self):
        async with websockets.serve(self.handler, self.host, self.port):
            print(f'WS server on ws://{self.host}:{self.port}')
            await asyncio.Future()
