# -*- coding: utf-8 -*-
"""
run_realtime_ofi.py — Realtime OFI pipeline (WebSocket + Calculator)

Features:
- Works with a real WebSocket stream OR a local `--demo` generator
- Parses incoming messages into (bids, asks) top-K, then feeds RealOFICalculator
- Prints OFI / Z / EMA and meta flags (warmup, std_zero)
- Includes reconnection, heartbeat timeout, backpressure protection, perf stats, graceful shutdown

Usage:
  Demo (no external deps needed):
    python run_realtime_ofi.py --demo

  Real stream:
    export WS_URL="wss://your-provider/your-stream"
    export SYMBOL="BTCUSDT"
    python run_realtime_ofi.py
"""
import asyncio
import json
import os
import signal
import time
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
        print("[ERROR] websockets not installed. Use: pip install websockets  (or run with --demo)")
        stop.set()
        return

    backoff = 1
    reconnect_count = 0
    while not stop.is_set():
        try:
            print(f"[INFO] Connecting to WebSocket: {url} (attempt #{reconnect_count+1})")
            async with websockets.connect(url, ping_interval=20, close_timeout=5) as ws:
                backoff = 1
                if reconnect_count > 0:
                    print(f"[INFO] Reconnected successfully after {reconnect_count} attempts")
                reconnect_count = 0
                # TODO: send subscription if needed
                while not stop.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=60)  # heartbeat timeout
                        await queue.put(msg)
                    except asyncio.TimeoutError:
                        print("[WARN] No data for 60s, triggering reconnect (heartbeat timeout)")
                        raise  # Force reconnection
        except asyncio.TimeoutError:
            reconnect_count += 1
            print(f"[WARN] WS heartbeat timeout; reconnect in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff*2, 30)
        except Exception as e:
            reconnect_count += 1
            print(f"[WARN] WS disconnected: {type(e).__name__}: {e}; reconnect in {backoff}s (backoff={backoff}s, max=30s)")
            await asyncio.sleep(backoff)
            backoff = min(backoff*2, 30)

async def printer(oci: RealOFICalculator, queue: asyncio.Queue, stop: asyncio.Event):
    perf = PerfStats()
    last_stat = time.time()
    dropped = 0
    parse_errors = 0
    processed = 0
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
                print(f"[WARN] Backpressure: skipped {skip_count} stale frames (queue depth was {skip_count+1})")
            
            t0 = time.perf_counter()
            parsed = parse_message(msg)
            if parsed is None:
                parse_errors += 1
                print(f"[ERROR] Failed to parse message (total parse errors: {parse_errors})")
                continue
            bids, asks = parsed
            ret = oci.update_with_snapshot(bids=bids, asks=asks)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            perf.add(dt_ms)
            processed += 1

            z = ret.get("z_ofi")
            warm = ret.get("meta",{}).get("warmup")
            stdz = ret.get("meta",{}).get("std_zero")
            print(f"{ret.get('symbol', 'N/A')} OFI={ret['ofi']:+.5f}  Z={('None' if z is None else f'{z:+.3f}')}  "
                  f"EMA={ret['ema_ofi']:+.5f}  warmup={warm}  std_zero={stdz}")

            if time.time() - last_stat > 60:
                stat = perf.snapshot_and_reset()
                queue_depth = queue.qsize()
                print(f"[STAT] window={stat['window_s']}s processed={processed} p50={stat['p50_ms']}ms p95={stat['p95_ms']}ms "
                      f"dropped={dropped} parse_errors={parse_errors} queue_depth={queue_depth}")
                last_stat = time.time()
        except asyncio.TimeoutError:
            print("[WARN] consume timeout; continue")
        except Exception as e:
            print(f"[ERROR] consume loop exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

async def main(demo: bool = False):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    
    # Cross-platform signal handling
    if os.name == 'nt':  # Windows
        # Windows doesn't support loop.add_signal_handler for SIGTERM
        def signal_handler(sig, frame):
            print(f"\n[INFO] Received signal {sig}, initiating graceful shutdown...")
            stop.set()
        signal.signal(signal.SIGINT, signal_handler)
        print("[INFO] Signal handlers configured (Windows mode: SIGINT only)")
    else:  # Unix-like
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: (
                print(f"\n[INFO] Received signal {s}, initiating graceful shutdown..."),
                stop.set()
            )[1])
        print("[INFO] Signal handlers configured (Unix mode: SIGINT + SIGTERM)")

    q: asyncio.Queue = asyncio.Queue(maxsize=1024)
    cfg = OFIConfig(levels=K_LEVELS, z_window=Z_WINDOW, ema_alpha=EMA_ALPHA)
    oci = RealOFICalculator(symbol=SYMBOL, cfg=cfg)
    
    print(f"[INFO] OFI Calculator initialized: symbol={SYMBOL}, K={K_LEVELS}, z_window={Z_WINDOW}, ema_alpha={EMA_ALPHA}")

    if demo or not WS_URL:
        prod = asyncio.create_task(demo_source(q))
        print("[INFO] Running in DEMO mode (local synthetic orderbook, 50 Hz)")
    else:
        prod = asyncio.create_task(ws_consume(WS_URL, q, stop))
        print(f"[INFO] Connecting to real WebSocket: {WS_URL}")
    cons = asyncio.create_task(printer(oci, q, stop))

    try:
        await stop.wait()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received, shutting down...")
        stop.set()
    
    print("[INFO] Cancelling tasks...")
    prod.cancel()
    cons.cancel()
    results = await asyncio.gather(prod, cons, return_exceptions=True)
    
    # Check for pending tasks
    pending = [t for t in asyncio.all_tasks() if not t.done()]
    if pending:
        print(f"[WARN] {len(pending)} pending tasks at shutdown: {pending}")
    else:
        print("[INFO] All tasks completed cleanly")
    
    print("[INFO] Graceful shutdown completed")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use local demo source instead of WebSocket")
    args = ap.parse_args()
    asyncio.run(main(demo=args.demo))
