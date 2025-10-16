import numpy as np
from collections import deque

class OnlineOFI:
    def __init__(self, micro_window_ms=100, z_window_seconds=900):
        self.w = micro_window_ms
        self.zn = int(max(10, z_window_seconds*1000 // self.w))
        self.cur_bucket = None; self.bucket_sum = 0.0
        self.history = deque(maxlen=self.zn); self.t_series = deque(maxlen=self.zn)
        self.last_best = None

    def on_best(self, t, bid, bid_sz, ask, ask_sz):
        self.last_best = (t, bid, bid_sz, ask, ask_sz)

    def on_l2(self, t, typ, side, price, qty):
        if not self.last_best: return
        _, bid, _, ask, _ = self.last_best
        is_add = (typ == "l2_add"); is_bid1 = abs(price - bid) < 1e-9; is_ask1 = abs(price - ask) < 1e-9
        contrib = 0.0
        if is_add and is_bid1: contrib += qty
        if is_add and is_ask1: contrib -= qty
        if (not is_add) and is_bid1: contrib -= qty
        if (not is_add) and is_ask1: contrib += qty
        bucket = (t // self.w) * self.w
        if self.cur_bucket is None: self.cur_bucket = bucket
        if bucket != self.cur_bucket:
            self.history.append(self.bucket_sum); self.t_series.append(self.cur_bucket)
            self.bucket_sum = 0.0; self.cur_bucket = bucket
        self.bucket_sum += contrib

    def read(self):
        if len(self.history) < max(10, self.zn//10): return None
        arr = np.array(self.history, dtype=float)
        z = (arr[-1] - arr.mean()) / (arr.std(ddof=0) + 1e-9)
        return {"t": self.t_series[-1], "ofi": float(arr[-1]), "ofi_z": float(z)}
