# -*- coding: utf-8 -*-
"""
binance_trade_stream.py - Minimal runnable Binance @aggTrade stream + CVD integration
- websockets client with heartbeat timeout and capped backoff
- backpressure: keep latest (drop stale) using bounded asyncio.Queue
- logging with simple rate limiting for noisy warnings
- integration with RealCVDCalculator (standard-library only)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

# ---- optional import path helper (project-style) ----
def _ensure_project_path():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "v13_ofi_ai_system", "src"),
        os.path.join(os.path.dirname(here), "v13_ofi_ai_system", "src"),
        os.path.join(here),
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

_ensure_project_path()

try:
    from real_cvd_calculator import RealCVDCalculator, CVDConfig
except Exception as e:  # pragma: no cover
    raise RuntimeError("Cannot import real_cvd_calculator. Ensure it exists under v13_ofi_ai_system/src") from e

try:
    import websockets
except ImportError as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: websockets>=10,<13") from e

# ---- logging ----
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("binance_trade_stream")

# ---- small rate limiter for noisy warnings ----
class RateLimiter:
    def __init__(self, max_per_sec: int = 5):
        self.max_per_sec = max_per_sec
        self._sec = int(time.time())
        self._count = 0
        self._suppressed = 0

    def allow(self) -> bool:
        now = int(time.time())
        if now != self._sec:
            if self._suppressed > 0:
                log.warning("...suppressed=%d", self._suppressed)
            self._sec = now
            self._count = 0
            self._suppressed = 0
        self._count += 1
        if self._count <= self.max_per_sec:
            return True
        self._suppressed += 1
        return False

# ---- perf stats ----
@dataclass
class PerfStats:
    count: int = 0
    total: float = 0.0
    p50: float = 0.0
    p95: float = 0.0

    def feed(self, dt: float):
        self.count += 1
        self.total += dt

    def snapshot_and_reset(self) -> Dict[str, Any]:
        avg = (self.total / self.count) if self.count else 0.0
        out = {"count": self.count, "avg_ms": avg * 1000.0, "p50_ms": self.p50, "p95_ms": self.p95}
        self.count = 0
        self.total = 0.0
        return out

# ---- monitoring metrics ----
@dataclass
class MonitoringMetrics:
    """全局监控指标"""
    reconnect_count: int = 0
    queue_dropped: int = 0
    total_messages: int = 0
    parse_errors: int = 0
    
    def queue_dropped_rate(self) -> float:
        return (self.queue_dropped / self.total_messages) if self.total_messages > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconnect_count": self.reconnect_count,
            "queue_dropped": self.queue_dropped,
            "total_messages": self.total_messages,
            "parse_errors": self.parse_errors,
            "queue_dropped_rate": self.queue_dropped_rate(),
        }

# ---- parsing ----
def parse_aggtrade_message(text: str) -> Optional[Tuple[float, float, bool, Optional[int]]]:
    """Support both stream-wrapped and raw aggTrade payloads."""
    try:
        obj = json.loads(text)
    except Exception:
        return None
    payload = obj.get("data", obj)
    try:
        p = float(payload["p"])
        q = float(payload["q"])
        m = bool(payload["m"])
        is_buy = (not m)  # m=isBuyerMaker: True means seller was aggressive -> is_buy=False
        et = payload.get("E") or payload.get("T")
        return p, q, is_buy, int(et) if et is not None else None
    except Exception:
        return None

# ---- websocket consumer ----
async def ws_consume(url: str, queue: asyncio.Queue, stop_evt: asyncio.Event, metrics: MonitoringMetrics):
    heartbeat_timeout = int(os.getenv("HEARTBEAT_TIMEOUT", "60"))
    backoff_max = int(os.getenv("BACKOFF_MAX", "30"))
    warn_rl = RateLimiter(max_per_sec=5)

    backoff = 1.0
    first_connect = True
    while not stop_evt.is_set():
        try:
            async with websockets.connect(url, ping_interval=None, close_timeout=5) as ws:
                log.info("Connected: %s", url)
                if not first_connect:
                    metrics.reconnect_count += 1
                    log.info("[METRICS] reconnect_count=%d", metrics.reconnect_count)
                first_connect = False
                backoff = 1.0
                while not stop_evt.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=heartbeat_timeout)
                    except asyncio.TimeoutError:
                        if warn_rl.allow():
                            log.warning("Heartbeat timeout (>%ss). Reconnecting...", heartbeat_timeout)
                        raise
                    except websockets.ConnectionClosed as e:
                        if warn_rl.allow():
                            log.warning("Connection closed: %s", e)
                        raise
                    
                    metrics.total_messages += 1
                    if queue.full():  # backpressure: keep latest
                        metrics.queue_dropped += 1
                        try:
                            _ = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await queue.put((time.time(), msg))
        except Exception as e:
            if warn_rl.allow():
                log.warning("Reconnect due to error: %s", e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, float(backoff_max))

# ---- processor ----
async def processor(symbol: str, queue: asyncio.Queue, stop_evt: asyncio.Event, metrics: MonitoringMetrics):
    cfg = CVDConfig()
    calc = RealCVDCalculator(symbol=symbol, cfg=cfg)

    perf = PerfStats()
    print_every = int(os.getenv("PRINT_EVERY", "100"))  # 每100条打印一次
    processed = 0
    stats_interval = 60.0
    last_stats = time.time()

    while not stop_evt.is_set():
        try:
            ts_recv, raw = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            if (time.time() - last_stats) >= stats_interval:
                snap = perf.snapshot_and_reset()
                log.info("[STAT] trades=%d avg_proc=%.3fms | %s", 
                         snap["count"], snap["avg_ms"], metrics.to_dict())
                last_stats = time.time()
            continue

        t0 = time.time()
        parsed = parse_aggtrade_message(raw)
        if parsed is None:
            metrics.parse_errors += 1
            log.warning("Parse error on message (truncated): %s", raw[:160])
            continue

        price, qty, is_buy, event_ms = parsed
        ret = calc.update_with_trade(price=price, qty=qty, is_buy=is_buy, event_time_ms=event_ms)
        processed += 1
        dt = time.time() - t0
        perf.feed(dt)
        
        # 计算延迟（从交易所事件时间到现在）
        latency_ms = (time.time() * 1000 - event_ms) if event_ms else 0.0

        if processed % print_every == 0 or processed <= 5:
            log.info(
                "CVD %s | cvd=%.6f z=%s ema=%.6f | warmup=%s std_zero=%s bad=%d | latency=%.1fms",
                symbol,
                ret["cvd"],
                "None" if ret["z_cvd"] is None else f"{ret['z_cvd']:.3f}",
                ret["ema_cvd"] if ret["ema_cvd"] is not None else float('nan'),
                ret["meta"]["warmup"],
                ret["meta"]["std_zero"],
                ret["meta"]["bad_points"],
                latency_ms,
            )

# ---- main ----
async def main(symbol: Optional[str] = None, url: Optional[str] = None):
    sym = (symbol or os.getenv("SYMBOL", "ETHUSDT")).upper()
    url = url or os.getenv(
        "WS_URL",
        f"wss://fstream.binancefuture.com/stream?streams={sym.lower()}@aggTrade",
    )

    queue_size = int(os.getenv("QUEUE_SIZE", "1024"))
    q: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    stop_evt = asyncio.Event()
    metrics = MonitoringMetrics()

    loop = asyncio.get_running_loop()
    def _set_stop(*_a):
        if not stop_evt.is_set():
            stop_evt.set()
    try:
        loop.add_signal_handler(signal.SIGINT, _set_stop)
    except NotImplementedError:
        pass
    try:
        loop.add_signal_handler(signal.SIGTERM, _set_stop)
    except (NotImplementedError, AttributeError):
        pass

    prod = asyncio.create_task(ws_consume(url, q, stop_evt, metrics))
    cons = asyncio.create_task(processor(sym, q, stop_evt, metrics))

    log.info("Starting trade stream for %s | url=%s", sym, url)
    try:
        await stop_evt.wait()
    except KeyboardInterrupt:
        _set_stop()
    finally:
        prod.cancel(); cons.cancel()
        await asyncio.gather(prod, cons, return_exceptions=True)
        log.info("Graceful shutdown. Final metrics: %s", metrics.to_dict())

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default=None, help="symbol, e.g. ETHUSDT (default from ENV SYMBOL)")
    ap.add_argument("--url", type=str, default=None, help="override websocket URL (default from ENV WS_URL)")
    args = ap.parse_args()
    asyncio.run(main(symbol=args.symbol, url=args.url))
