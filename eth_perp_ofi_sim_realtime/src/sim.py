import numpy as np, time
from .book import OrderBookTopN
from dataclasses import dataclass

@dataclass
class Regime:
    name: str; mu: float; sigma: float; dur_mean_s: float; prob: float

class MarketSimulator:
    def __init__(self, params: dict):
        self.p = params["sim"]
        self.rng = np.random.default_rng(self.p["seed"])
        self.tick = self.p["tick_size"]; self.levels = self.p["levels"]
        self.regimes = [Regime(r["name"], r["mu"], r["sigma"], r["dur_mean_s"], r["prob"]) for r in self.p["regimes"]]
        probs = np.array([r.prob for r in self.regimes]); probs/=probs.sum()
        self.regime = self.rng.choice(self.regimes, p=probs); self.regime_left = self._dur(self.regime)
        self.book = OrderBookTopN(self.levels, self.tick)
        self.mid = self.p["init_mid"]; self.book.init_from_mid(self.mid, self.p["base_spread_ticks"], self.p["base_depth"], self.p["depth_jitter"])
        self.time_ms = 0

    def _dur(self, reg): return int(max(1, self.rng.exponential(reg.dur_mean_s))*1000)
    def _switch(self, dt): self.regime_left -= dt
    def _maybe_switch(self):
        if self.regime_left<=0:
            probs=np.array([r.prob for r in self.regimes]); probs/=probs.sum()
            self.regime=self.rng.choice(self.regimes,p=probs); self.regime_left=self._dur(self.regime)

    def _latent(self, dt_s):
        mu=self.regime.mu*dt_s; sigma=self.regime.sigma*np.sqrt(dt_s)
        d=self.rng.normal(mu,sigma); self.mid = max(1.0, self.mid*(1.0 + d/1e3))

    def _step_once(self, dt_ms=10):
        self.time_ms += dt_ms; self._switch(dt_ms); self._latent(dt_ms/1000); self._maybe_switch()
        evts=[]; rates=self.p["rates"]; step=lambda r: 1-np.exp(-r*(dt_ms/1000))
        if self.rng.random()<step(rates["limit_add"]):
            side='bid' if self.rng.random()<0.5 else 'ask'
            lvl=max(0,min(int(self.rng.geometric(0.5))-1,self.levels-1))
            bp,bs,ap,az=self.book.snapshot()
            ref=(bp[0]-lvl*self.tick) if side=='bid' and bp else (ap[0]+lvl*self.tick) if ap else self.mid
            qty=max(0.5, self.rng.lognormal(2.5, self.p["depth_jitter"])); self.book.limit_add(side,ref,qty)
            evts.append({"t":self.time_ms,"type":"l2_add","side":side,"price":float(ref),"qty":float(qty)})
        if self.rng.random()<step(rates["limit_cancel"]):
            side='bid' if self.rng.random()<0.5 else 'ask'
            bp,bs,ap,az=self.book.snapshot(); ref=(bp[0] if side=='bid' and bp else ap[0] if ap else self.mid)
            qty=max(0.5, self.rng.lognormal(2.2, self.p["depth_jitter"])); self.book.limit_cancel(side,ref,qty)
            evts.append({"t":self.time_ms,"type":"l2_cancel","side":side,"price":float(ref),"qty":float(qty)})
        if self.rng.random()<step(rates["market_sweep"]):
            side='buy' if self.rng.random()<0.5 else 'sell'
            levels=max(1,int(self.rng.exponential(self.p["sweep_levels_mean"])))
            bp,bs,ap,az=self.book.snapshot()
            depth=sum(az[:levels]) if side=='buy' and az else sum(bs[:levels]) if bs else 0.0
            qty=max(1.0, depth*self.rng.uniform(0.2,0.8)); vwap,last_px=self.book.market_sweep(side,qty)
            if vwap: evts.append({"t":self.time_ms,"type":"trade","side":side,"qty":float(qty),"vwap":float(vwap),"last_px":float(last_px)})
        if self.time_ms%50==0:
            bp,bs,ap,az=self.book.snapshot()
            if bp and ap: evts.append({"t":self.time_ms,"type":"best","bid":float(bp[0]),"bid_sz":float(bs[0]),"ask":float(ap[0]),"ask_sz":float(az[0])})
        return evts

    def run(self):
        import pandas as pd
        events=[]; steps=int(self.p["seconds"]*1000/10)
        for _ in range(steps):
            events.extend(self._step_once(10))
        return pd.DataFrame(events)

    def stream(self, realtime=True, dt_ms=10):
        steps=int(self.p["seconds"]*1000/dt_ms)
        start=time.time()
        for i in range(steps):
            evts=self._step_once(dt_ms)
            yield evts
            if realtime:
                target=(i+1)*dt_ms/1000.0; elapsed=time.time()-start; sleep=target-elapsed
                if sleep>0: time.sleep(sleep)
