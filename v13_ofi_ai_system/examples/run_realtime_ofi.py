# -*- coding: utf-8 -*-
"""
run_realtime_ofi.py — Realtime OFI pipeline (WebSocket + Calculator)

Features:
- Works with a real WebSocket stream OR a local `--demo` generator
- Parses incoming messages into (bids, asks) top-K, then feeds RealOFICalculator
- Logs OFI / Z / EMA and meta flags (warmup, std_zero) with rate limiting
- Includes reconnection, heartbeat timeout, backpressure protection, perf stats, graceful shutdown

Usage:
  Demo (no external deps needed):
    python run_realtime_ofi.py --demo

  Real stream:
    export WS_URL="wss://your-provider/your-stream"
    export SYMBOL="BTCUSDT"
    python run_realtime_ofi.py
    
  Control log level:
    LOG_LEVEL=DEBUG python run_realtime_ofi.py --demo
"""
import asyncio
import json
import os
import signal
import time
import math
import random
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

# === Import RealOFICalculator from project or local ===
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_PATHS = [
    os.path.abspath(os.path.join(THIS_DIR, "..", "src")),       # v13_ofi_ai_system/examples -> v13_ofi_ai_system/src
    os.path.abspath(os.path.join(THIS_DIR, "..", "..", "src")), # ofi_cvd_framework/v13_ofi_ai_system/examples -> src
    THIS_DIR,                                                     # same folder
    os.getcwd(),                                                  # current working dir
]
for p in CANDIDATE_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from real_ofi_calculator import RealOFICalculator, OFIConfig  # type: ignore
except Exception as e:
    raise SystemExit(
        "Cannot import real_ofi_calculator. Ensure it exists in project src or same directory.\n"
        f"sys.path searched: {CANDIDATE_PATHS}\nError: {e}"
    )

# === Config ===
K_LEVELS = int(os.getenv("K_LEVELS", "5"))
SYMBOL = os.getenv("SYMBOL", "DEMO-USD")
WS_URL = os.getenv("WS_URL", "")  # if empty -> demo mode
Z_WINDOW = int(os.getenv("Z_WINDOW", "300"))
EMA_ALPHA = float(os.getenv("EMA_ALPHA", "0.2"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# === Setup logging ===
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class PerfStats:
    last_reset: float = time.time()
    samples: List[float] = None

    def __post_init__(self):
        self.samples = []

    def add(self, dt_ms: float):
        self.samples.append(dt_ms)

    def snapshot_and_reset(self):
        now = time.time()
        win = now - self.last_reset
        arr = sorted(self.samples)
        def pct(p):
            if not arr: return None
            i = max(0, min(len(arr)-1, int(len(arr)*p/100)-1))
            return round(arr[i], 3)
        snap = {"window_s": round(win,1), "n": len(arr), "p50_ms": pct(50), "p95_ms": pct(95)}
        self.last_reset, self.samples = now, []
        return snap

@dataclass
class RateLimiter:
    """Rate limiter for log messages (per-key throttling)"""
    window_sec: float = 1.0  # Time window for rate limiting
    max_per_window: int = 5  # Max messages per window per key
    counters: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_reset: Dict[str, float] = field(default_factory=dict)
    suppressed: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def should_log(self, key: str) -> bool:
        """Check if a log message should be emitted for this key"""
        now = time.time()
        
        # Reset window if expired
        if key not in self.last_reset or (now - self.last_reset[key]) >= self.window_sec:
            if self.suppressed[key] > 0:
                # Log suppression summary at window end
                logger.warning(f"[{key}] Suppressed {self.suppressed[key]} messages in last {self.window_sec}s")
                self.suppressed[key] = 0
            self.counters[key] = 0
            self.last_reset[key] = now
        
        # Check rate limit
        if self.counters[key] < self.max_per_window:
            self.counters[key] += 1
            return True
        else:
            self.suppressed[key] += 1
            return False

def topk_pad(levels: List[Tuple[float,float]], k: int, reverse: bool) -> List[Tuple[float,float]]:
    lv = []
    for x in levels:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            continue
        try:
            px = float(x[0])
            qx = float(x[1])
        except Exception:
            continue
        lv.append((px, max(0.0, qx)))
    lv.sort(key=lambda x: x[0], reverse=reverse)
    lv = lv[:k]
    if len(lv) < k:
        lv = lv + [(0.0, 0.0)]*(k - len(lv))
    return lv

# === Adapter: incoming msg -> (bids, asks) ===
def parse_message(msg: str) -> Optional[Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]]:
    """
    Expected JSON (Binance):
      {"stream": "ethusdt@depth@100ms", "data": {"e": "depthUpdate", "b": [[price, qty], ...], "a": [[price, qty], ...]}}
    Or for DEMO mode:
      {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}
    """
    try:
        raw = json.loads(msg)
        
        # Check if it's Binance format (nested in 'data')
        if "data" in raw:
            data = raw["data"]
            bids = topk_pad(data.get("b", []), K_LEVELS, reverse=True)
            asks = topk_pad(data.get("a", []), K_LEVELS, reverse=False)
        else:
            # DEMO format (direct bids/asks)
            bids = topk_pad(raw.get("bids", []), K_LEVELS, reverse=True)
            asks = topk_pad(raw.get("asks", []), K_LEVELS, reverse=False)
        
        return bids, asks
    except Exception:
        return None

# === DEMO source: local synthetic orderbook ===
async def demo_source(queue: asyncio.Queue, hz: int = 50):
    """
    Local synthetic order book generator
    
    Args:
        queue: asyncio.Queue to put messages into
        hz: Frequency in Hz (messages per second)
            - Default: 50 Hz (50 msgs/s) - normal testing
            - Can set to 100 Hz (100 msgs/s) - high load testing
    """
    base_p = 100.0
    t = 0.0
    while True:
        t += 1.0/hz
        mid = base_p + 0.5*math.sin(t/3.0)
        spread = 0.02
        bids = [[mid - i*spread, max(0.0, 10 + 3*math.sin(t+i) + random.uniform(-1,1))] for i in range(K_LEVELS)]
        asks = [[mid + i*spread, max(0.0, 10 + 3*math.cos(t+i) + random.uniform(-1,1))] for i in range(K_LEVELS)]
        await queue.put(json.dumps({"bids": bids, "asks": asks}))
        await asyncio.sleep(1.0/hz)

# === WebSocket consumer ===
async def ws_consume(url: str, queue: asyncio.Queue, stop: asyncio.Event):
    try:
        import websockets  # external lightweight dependency
    except ModuleNotFoundError:
        logger.error("websockets not installed. Use: pip install websockets  (or run with --demo)")
        stop.set()
        return

    backoff = 1
    reconnect_count = 0
    while not stop.is_set():
        try:
            logger.info(f"Connecting to WebSocket: {url} (attempt #{reconnect_count+1})")
            async with websockets.connect(url, ping_interval=20, close_timeout=5) as ws:
                backoff = 1
                if reconnect_count > 0:
                    logger.info(f"Reconnected successfully after {reconnect_count} attempts")
                reconnect_count = 0
                # TODO: send subscription if needed
                while not stop.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=60)  # heartbeat timeout
                        await queue.put(msg)
                    except asyncio.TimeoutError:
                        logger.warning("No data for 60s, triggering reconnect (heartbeat timeout)")
                        raise  # Force reconnection
        except asyncio.TimeoutError:
            reconnect_count += 1
            logger.warning(f"WS heartbeat timeout; reconnect in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff*2, 30)
        except Exception as e:
            reconnect_count += 1
            logger.warning(f"WS disconnected: {type(e).__name__}: {e}; reconnect in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff*2, 30)

async def printer(oci: RealOFICalculator, queue: asyncio.Queue, stop: asyncio.Event):
    perf = PerfStats()
    rate_limiter = RateLimiter(window_sec=1.0, max_per_window=5)  # Max 5 WARN/ERROR per second per type
    last_stat = time.time()
    dropped = 0
    parse_errors = 0
    processed = 0
    ofi_print_interval = 10  # Print OFI every N messages (reduce spam in high-frequency mode)
    
    while not stop.is_set():
        try:
            msg = await queue.get()
            # Backpressure: keep latest frame only
            skip_count = 0
            while not queue.empty():
                _ = queue.get_nowait()
                skip_count += 1
            if skip_count > 0:
                dropped += skip_count
                # Rate-limited backpressure warning
                if rate_limiter.should_log("backpressure"):
                    logger.warning(f"Backpressure: skipped {skip_count} stale frames (total dropped: {dropped})")
            
            t0 = time.perf_counter()
            parsed = parse_message(msg)
            if parsed is None:
                parse_errors += 1
                # Rate-limited parse error
                if rate_limiter.should_log("parse_error"):
                    logger.error(f"Failed to parse message (total errors: {parse_errors})")
                continue
            bids, asks = parsed
            ret = oci.update_with_snapshot(bids=bids, asks=asks)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            perf.add(dt_ms)
            processed += 1

            # Print OFI at reduced frequency (every Nth message)
            if processed % ofi_print_interval == 0 or processed <= 10:  # Always print first 10
                z = ret.get("z_ofi")
                warm = ret.get("meta",{}).get("warmup")
                stdz = ret.get("meta",{}).get("std_zero")
                logger.info(f"{ret.get('symbol', 'N/A')} OFI={ret['ofi']:+.5f}  Z={('None' if z is None else f'{z:+.3f}')}  "
                          f"EMA={ret['ema_ofi']:+.5f}  warmup={warm}  std_zero={stdz}")

            # Periodic stats (every 60s)
            if time.time() - last_stat > 60:
                stat = perf.snapshot_and_reset()
                queue_depth = queue.qsize()
                logger.info(f"STATS | window={stat['window_s']}s processed={processed} p50={stat['p50_ms']}ms p95={stat['p95_ms']}ms "
                          f"dropped={dropped} parse_errors={parse_errors} queue_depth={queue_depth}")
                last_stat = time.time()
        except asyncio.TimeoutError:
            logger.warning("consume timeout; continue")
        except Exception as e:
            logger.error(f"consume loop exception: {type(e).__name__}: {e}", exc_info=True)

async def main(demo: bool = False):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    
    # Cross-platform signal handling
    if os.name == 'nt':  # Windows
        # Windows doesn't support loop.add_signal_handler for SIGTERM
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            stop.set()
        signal.signal(signal.SIGINT, signal_handler)
        logger.info("Signal handlers configured (Windows mode: SIGINT only)")
    else:  # Unix-like
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: (
                logger.info(f"Received signal {s}, initiating graceful shutdown..."),
                stop.set()
            )[1])
        logger.info("Signal handlers configured (Unix mode: SIGINT + SIGTERM)")

    q: asyncio.Queue = asyncio.Queue(maxsize=1024)
    cfg = OFIConfig(levels=K_LEVELS, z_window=Z_WINDOW, ema_alpha=EMA_ALPHA)
    oci = RealOFICalculator(symbol=SYMBOL, cfg=cfg)
    
    logger.info(f"OFI Calculator initialized: symbol={SYMBOL}, K={K_LEVELS}, z_window={Z_WINDOW}, ema_alpha={EMA_ALPHA}")

    if demo or not WS_URL:
        demo_hz = 50  # Change to 100 for high-load testing (100 msgs/s)
        prod = asyncio.create_task(demo_source(q, hz=demo_hz))
        logger.info(f"Running in DEMO mode (local synthetic orderbook, {demo_hz} Hz = {demo_hz} msgs/s)")
    else:
        prod = asyncio.create_task(ws_consume(WS_URL, q, stop))
        logger.info(f"Connecting to real WebSocket: {WS_URL}")
    cons = asyncio.create_task(printer(oci, q, stop))

    try:
        await stop.wait()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
        stop.set()
    
    logger.info("Cancelling tasks...")
    prod.cancel()
    cons.cancel()
    results = await asyncio.gather(prod, cons, return_exceptions=True)
    
    # Check for pending tasks
    pending = [t for t in asyncio.all_tasks() if not t.done()]
    if pending:
        logger.warning(f"{len(pending)} pending tasks at shutdown: {pending}")
    else:
        logger.info("All tasks completed cleanly")
    
    logger.info("Graceful shutdown completed")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use local demo source instead of WebSocket")
    args = ap.parse_args()
    asyncio.run(main(demo=args.demo))
