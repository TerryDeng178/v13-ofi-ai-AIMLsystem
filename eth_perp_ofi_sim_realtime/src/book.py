from dataclasses import dataclass
import numpy as np

@dataclass
class L2Level:
    price: float
    size: float

class OrderBookTopN:
    def __init__(self, levels=5, tick=0.1):
        self.N = levels; self.tick = tick
        self.bids = []; self.asks = []

    def snapshot(self):
        def to_arrays(levels):
            return [lvl.price for lvl in levels], [lvl.size for lvl in levels]
        bp, bs = to_arrays(self.bids); ap, az = to_arrays(self.asks)
        return bp, bs, ap, az

    def init_from_mid(self, mid, spread_ticks, base_depth, jitter):
        bid1 = mid - spread_ticks * self.tick / 2.0
        ask1 = mid + spread_ticks * self.tick / 2.0
        self.bids = [L2Level(bid1 - i*self.tick, base_depth * max(0.2, np.random.lognormal(0, jitter))) for i in range(self.N)]
        self.asks = [L2Level(ask1 + i*self.tick, base_depth * max(0.2, np.random.lognormal(0, jitter))) for i in range(self.N)]

    def _trim(self):
        self.bids = sorted(self.bids, key=lambda x: (-x.price, -x.size))[:self.N]
        self.asks = sorted(self.asks, key=lambda x: (x.price, -x.size))[:self.N]

    def limit_add(self, side, px, qty):
        levels = self.bids if side=='bid' else self.asks
        for lvl in levels:
            if abs(lvl.price - px) < 1e-9:
                lvl.size += qty; break
        else:
            levels.append(L2Level(px, qty))
        self._trim()

    def limit_cancel(self, side, px, qty):
        levels = self.bids if side=='bid' else self.asks
        for lvl in levels:
            if abs(lvl.price - px) < 1e-9:
                lvl.size = max(0.0, lvl.size - qty)
        self._trim()

    def market_sweep(self, side, qty):
        notional = 0.0
        lvls = self.asks if side=='buy' else self.bids
        while qty > 0 and lvls:
            lvl = lvls[0]
            hit = min(qty, lvl.size)
            notional += hit * lvl.price
            qty -= hit; lvl.size -= hit
            if lvl.size <= 1e-9: del lvls[0]
        self._trim()
        last_px = (self.asks[0].price if side=='buy' and self.asks else self.bids[0].price if self.bids else None)
        vwap = last_px
        return vwap, last_px

    def mid(self):
        if not self.bids or not self.asks: return None
        return 0.5*(self.bids[0].price + self.asks[0].price)
