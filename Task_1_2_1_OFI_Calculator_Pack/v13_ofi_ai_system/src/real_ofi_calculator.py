# -*- coding: utf-8 -*-
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

@dataclass
class OFIConfig:
    levels: int = 5
    weights: Optional[List[float]] = None
    z_window: int = 300
    ema_alpha: float = 0.2

def _is_finite_number(x: float) -> bool:
    try:
        y = float(x)
        return y == y and y not in (float('inf'), float('-inf'))
    except Exception:
        return False

class RealOFICalculator:
    __slots__ = (
        "symbol","K","w","z_window","ema_alpha",
        "bids","asks","prev_bids","prev_asks",
        "ofi_hist","ema_ofi","bad_points"
    )
    def __init__(self, symbol: str, cfg: OFIConfig = OFIConfig()):
        self.symbol = (symbol or "").upper()
        self.K = int(cfg.levels) if cfg.levels and cfg.levels > 0 else 5
        default_w = [0.4,0.25,0.2,0.1,0.05]
        if cfg.weights is None:
            w_raw = default_w[:self.K] if len(default_w) >= self.K else (default_w + [0.0]*max(0,self.K-len(default_w)))
        else:
            w_raw = [float(x) for x in cfg.weights[:self.K]] + [0.0]*max(0, self.K - len(cfg.weights))
        total = sum(max(0.0,x) for x in w_raw)
        if total <= 0.0:
            raise ValueError("weights must have positive sum")
        self.w = [max(0.0,x)/total for x in w_raw]
        self.z_window = int(cfg.z_window) if cfg.z_window and cfg.z_window>0 else 300
        self.ema_alpha = float(cfg.ema_alpha)
        self.bids = [[0.0,0.0] for _ in range(self.K)]
        self.asks = [[0.0,0.0] for _ in range(self.K)]
        self.prev_bids = [[0.0,0.0] for _ in range(self.K)]
        self.prev_asks = [[0.0,0.0] for _ in range(self.K)]
        self.ofi_hist = deque(maxlen=self.z_window)
        self.ema_ofi: Optional[float] = None
        self.bad_points = 0

    def _pad_snapshot(self, arr: List[Tuple[float, float]]):
        out = [[0.0,0.0] for _ in range(self.K)]
        n = min(len(arr or []), self.K)
        bad = False
        for i in range(n):
            p,q = arr[i]
            if not _is_finite_number(p): bad=True; p=0.0
            if not _is_finite_number(q) or float(q)<0: bad=True; q=0.0
            out[i][0]=float(p); out[i][1]=float(q)
        if bad: self.bad_points += 1
        return out

    @staticmethod
    def _mean_std(values: List[float]):
        n=len(values)
        if n==0: return 0.0,0.0
        m=sum(values)/n
        if n==1: return m,0.0
        var=sum((x-m)*(x-m) for x in values)/(n-1)
        return m, var**0.5

    def reset(self)->None:
        for i in range(self.K):
            self.bids[i][0]=self.bids[i][1]=0.0
            self.asks[i][0]=self.asks[i][1]=0.0
            self.prev_bids[i][0]=self.prev_bids[i][1]=0.0
            self.prev_asks[i][0]=self.prev_asks[i][1]=0.0
        self.ofi_hist.clear(); self.ema_ofi=None; self.bad_points=0

    def get_state(self)->Dict:
        return {
            "symbol": self.symbol,
            "levels": self.K,
            "weights": list(self.w),
            "bids": [list(x) for x in self.bids],
            "asks": [list(x) for x in self.asks],
            "bad_points": self.bad_points,
            "ema_ofi": self.ema_ofi,
            "ofi_hist_len": len(self.ofi_hist),
        }

    def update_with_snapshot(self, bids: List[Tuple[float,float]], asks: List[Tuple[float,float]], event_time_ms: Optional[int]=None)->Dict:
        for i in range(self.K):
            self.prev_bids[i][0]=self.bids[i][0]; self.prev_bids[i][1]=self.bids[i][1]
            self.prev_asks[i][0]=self.asks[i][0]; self.prev_asks[i][1]=self.asks[i][1]
        self.bids = self._pad_snapshot(bids)
        self.asks = self._pad_snapshot(asks)

        k_components=[]; ofi_val=0.0
        for i in range(self.K):
            delta_b = self.bids[i][1] - self.prev_bids[i][1]
            delta_a = self.asks[i][1] - self.prev_asks[i][1]
            comp = self.w[i]*(delta_b - delta_a)
            k_components.append(comp); ofi_val += comp

        self.ofi_hist.append(ofi_val)
        z_ofi=None; warmup=False
        if len(self.ofi_hist) < max(5, self.z_window//5):
            warmup=True
        else:
            arr=list(self.ofi_hist); m,s=self._mean_std(arr)
            z_ofi = 0.0 if s<=1e-9 else (ofi_val - m)/s

        if self.ema_ofi is None:
            self.ema_ofi = ofi_val
        else:
            a=self.ema_alpha; self.ema_ofi = a*ofi_val + (1.0-a)*self.ema_ofi

        return {
            "symbol": self.symbol,
            "event_time_ms": event_time_ms,
            "ofi": ofi_val,
            "k_components": k_components,
            "z_ofi": z_ofi,
            "ema_ofi": self.ema_ofi,
            "meta": {
                "levels": self.K,
                "weights": list(self.w),
                "bad_points": self.bad_points,
                "warmup": warmup,
            },
        }

    def update_with_l2_delta(self, deltas, event_time_ms: Optional[int]=None):
        raise NotImplementedError("Task 1.2.1 implements snapshot mode only.")
