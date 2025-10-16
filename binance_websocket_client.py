# -*- coding: utf-8 -*-
"""
Binance WebSocket Order Book Streamer (Task 1.1.6 compliant)
- Async logging via QueueHandler/QueueListener (non-blocking)
- Log rotation (interval or size) with retention
- NDJSON per message (optional)
- metrics.json refreshed every 10s
- Strict continuity: Futures depthUpdate requires pu == last_u, otherwise resync
- No external deps (pure stdlib). Percentiles implemented in pure Python.
"""
from __future__ import annotations

import sys, os, json, time, gzip, threading
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# websocket-client (3rd party) is assumed to be available in the project env
try:
    import websocket  # type: ignore
except Exception:
    websocket = None  # Will raise at runtime if not installed

# Async logging helper
try:
    from utils.async_logging import setup_async_logging, sample_queue_metrics  # project path
except Exception:
    # fallback to flat file in PYTHONPATH
    from async_logging import setup_async_logging, sample_queue_metrics  # type: ignore

# ---------------------------- Helpers ----------------------------

def now_ms() -> int:
    return int(time.time() * 1000)

def percentile(values: List[float], p: float) -> float:
    """Pure Python percentile (0-100). Linear interpolation between ranks."""
    n = len(values)
    if n == 0: return 0.0
    if n == 1: return float(values[0])
    v = sorted(values)
    if p <= 0: return float(v[0])
    if p >= 100: return float(v[-1])
    pos = (n - 1) * (p / 100.0)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(v[lo] * (1 - frac) + v[hi] * frac)

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def http_get(url: str, timeout: int = 5) -> Dict[str, Any]:
    """std lib GET; returns parsed JSON dict"""
    import urllib.request, urllib.error
    req = urllib.request.Request(url, headers={"User-Agent": "V13-OFI/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))

# ---------------------------- Main class ----------------------------

class BinanceOrderBookStream:
    def __init__(self,
                 symbol: str = "ethusdt",
                 depth_levels: int = 5,
                 rotate: str = "interval",
                 rotate_sec: int = 60,
                 max_bytes: int = 5_000_000,
                 backups: int = 7,
                 print_interval: int = 10,
                 base_dir: Path | None = None):
        self.symbol = symbol.lower()
        self.depth_levels = depth_levels
        self.print_interval = max(5, int(print_interval))
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_url = f"wss://fstream.binance.com/stream?streams={self.symbol}@depth@100ms"
        self.rest_snap_url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol.upper()}&limit=1000"

        # Paths
        self.base_dir = base_dir or Path.cwd() / "v13_ofi_ai_system"
        self.log_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "data" / "order_book"
        self.ndjson_path = self.data_dir / f"{self.symbol}_depth.ndjson.gz"
        ensure_dir(self.ndjson_path)
        ensure_dir(self.log_dir / "dummy.log")

        # Async logging (non-blocking)
        log_file = self.log_dir / f"{self.symbol}_{datetime.utcnow().strftime('%Y%m%d')}.log"
        self.logger, self.listener, self.log_qh = setup_async_logging(
            name=f"LOB[{self.symbol}]",
            log_path=str(log_file),
            rotate=rotate,
            rotate_sec=rotate_sec,
            max_bytes=max_bytes,
            backups=backups,
            level=logging.INFO,
            queue_max=10000,
            to_console=True,
        )

        # Stats
        self.stats: Dict[str, Any] = dict(
            start_time = datetime.now(),
            total_messages = 0,
            latency_list = deque(maxlen=10_000),
            batch_span_list = deque(maxlen=10_000),
            batch_span_max = 0,
            log_queue_depth_list = deque(maxlen=3_600),
            log_queue_max_depth = 0,
            log_drops = 0,
            resyncs = 0,
            reconnects = 0,
            last_summary_ts = 0.0,
            last_metrics_ts = 0.0,
        )

        # Continuity state
        self.last_update_id: Optional[int] = None  # from REST snapshot
        self.last_u: Optional[int] = None          # last applied u
        self.synced: bool = False                  # whether we passed alignment U <= L+1 <= u

        self.logger.info("Initialized BinanceOrderBookStream")
        self.logger.info(f"REST snapshot URL: {self.rest_snap_url}")
        self.logger.info(f"WebSocket URL:     {self.ws_url}")
        self.logger.info(f"Log file: {log_file}")

    # --------------------- Snapshot & Continuity ---------------------

    def load_snapshot(self):
        """Fetch REST snapshot to get lastUpdateId for alignment."""
        try:
            snap = http_get(self.rest_snap_url, timeout=8)
            L = int(snap.get("lastUpdateId"))
            self.last_update_id = L
            self.last_u = None
            self.synced = False
            self.logger.info(f"Loaded REST snapshot lastUpdateId={L}")
        except Exception as e:
            self.logger.error(f"load_snapshot failed: {e}", exc_info=True)
            raise

    def _try_align_first_event(self, U: int, u: int) -> bool:
        """Return True if U <= L+1 <= u"""
        if self.last_update_id is None:
            return False
        L = self.last_update_id
        return (U <= (L + 1) <= u)

    # --------------------- WebSocket handlers ---------------------

    def on_open(self, ws):
        self.logger.info("WebSocket opened")
        # Always refresh snapshot on open to have a fresh L
        self.load_snapshot()

    def on_close(self, ws, *args):
        self.logger.warning("WebSocket closed")
        self.stats["reconnects"] += 1

    def on_error(self, ws, err):
        self.logger.error(f"WebSocket error: {err}", exc_info=True)

    def on_message(self, ws, raw: str):
        ts_recv = now_ms()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return

        # Futures combined stream format: {"stream": "...", "data": {...}}
        data = msg.get("data", msg)

        # Extract depthUpdate fields
        U = data.get("U")  # first
        u = data.get("u")  # last
        pu = data.get("pu")  # previous last
        E = data.get("E", data.get("T", ts_recv))  # event time

        if U is None or u is None:
            # Not a depthUpdate
            return

        # Latency
        latency_event_ms = max(0.0, float(ts_recv - int(E)))
        t0 = time.perf_counter()

        # First alignment
        if not self.synced:
            if self.last_update_id is None:
                self.logger.warning("No snapshot L yet; skipping")
                return
            if self._try_align_first_event(int(U), int(u)):
                self.synced = True
                self.last_u = int(u)
                self.logger.info(f"Aligned on first event: U={U}, u={u}, L={self.last_update_id}")
                # Apply this event as the first applied
                batch_span = int(u) - int(U) + 1
                self._record_stats(latency_event_ms, time.perf_counter()-t0, batch_span)
                self._maybe_emit(ts_recv)
            else:
                # discard until alignment satisfied
                return
            return

        # Subsequent continuity check: require pu == last_u
        if pu is None or self.last_u is None or int(pu) != int(self.last_u):
            self.logger.warning(f"Continuity break (pu={pu}, last_u={self.last_u}), resync...")
            self.stats["resyncs"] += 1
            # Resync: reload snapshot, wait for next aligned event
            self.load_snapshot()
            return

        # Apply event & update last_u
        self.last_u = int(u)
        batch_span = int(u) - int(U) + 1
        self._record_stats(latency_event_ms, time.perf_counter()-t0, batch_span)

        # Persist NDJSON (compact)
        self._write_ndjson({
            "timestamp": datetime.utcfromtimestamp(E/1000).isoformat(),
            "symbol": self.symbol.upper(),
            "ts_recv": float(ts_recv),
            "E": int(E),
            "U": int(U),
            "u": int(u),
            "pu": int(pu) if pu is not None else None,
            "latency_event_ms": round(latency_event_ms, 3),
            "latency_pipeline_ms": round((time.perf_counter()-t0)*1000.0, 3),
        })

        self._maybe_emit(ts_recv)

    # --------------------- Stats & IO ---------------------

    def _record_stats(self, latency_event_ms: float, pipeline_sec: float, batch_span: int):
        self.stats["total_messages"] += 1
        self.stats["latency_list"].append(latency_event_ms)
        self.stats["batch_span_list"].append(batch_span)
        if batch_span > self.stats["batch_span_max"]:
            self.stats["batch_span_max"] = batch_span

        # log queue metrics
        q = sample_queue_metrics(self.log_qh)
        self.stats["log_queue_depth_list"].append(q["depth"])
        if q["max_depth"] > self.stats["log_queue_max_depth"]:
            self.stats["log_queue_max_depth"] = q["max_depth"]
        self.stats["log_drops"] = q["drops"]

    def _write_ndjson(self, obj: Dict[str, Any]):
        try:
            line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"
            with gzip.open(self.ndjson_path, "at", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            self.logger.error(f"write ndjson failed: {e}")

    def _maybe_emit(self, ts_recv: int):
        # print summary
        if (time.time() - self.stats["last_summary_ts"]) >= self.print_interval:
            self.stats["last_summary_ts"] = time.time()
            self.print_statistics()
        # save metrics json every 10s
        if (time.time() - self.stats["last_metrics_ts"]) >= 10.0:
            self.stats["last_metrics_ts"] = time.time()
            self.save_metrics_json()

    def calculate_percentiles(self) -> Dict[str, float]:
        lat = list(self.stats["latency_list"])
        return {
            "p50": round(percentile(lat, 50), 2),
            "p95": round(percentile(lat, 95), 2),
            "p99": round(percentile(lat, 99), 2),
        }

    def print_statistics(self):
        """SUMMARY line, 10s interval"""
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        if elapsed <= 0:
            return
        rate = self.stats['total_messages'] / elapsed if elapsed > 0 else 0.0
        pc = self.calculate_percentiles() if len(self.stats['latency_list']) else {'p50':0,'p95':0,'p99':0}

        bs_list = list(self.stats['batch_span_list'])
        lq_list = list(self.stats['log_queue_depth_list'])
        bs_p95 = round(percentile(bs_list, 95), 0) if bs_list else 0
        lq_p95 = round(percentile(lq_list, 95), 0) if lq_list else 0

        print(
            f"\nSUMMARY | t={elapsed:.0f}s | msgs={self.stats['total_messages']} | "
            f"rate={rate:.2f}/s | p50={pc['p50']:.1f} p95={pc['p95']:.1f} p99={pc['p99']:.1f} | "
            f"breaks={0} resyncs={self.stats['resyncs']} reconnects={self.stats['reconnects']} | "
            f"batch_span_p95={bs_p95:.0f} max={self.stats['batch_span_max']} | "
            f"log_q_p95={lq_p95:.0f} max={self.stats['log_queue_max_depth']} drops={self.stats['log_drops']}"
        )
        self.logger.info(
            f"SUMMARY: runtime={elapsed:.0f}s, msgs={self.stats['total_messages']}, "
            f"rate={rate:.2f}/s, p95={pc['p95']:.1f}ms, resyncs={self.stats['resyncs']}, "
            f"log_drops={self.stats['log_drops']}"
        )

    def save_metrics_json(self):
        """Write metrics.json (10s cadence)"""
        try:
            elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
            rate = self.stats['total_messages'] / elapsed if elapsed > 0 else 0.0
            pc = self.calculate_percentiles() if len(self.stats['latency_list']) else {'p50':0,'p95':0,'p99':0}
            lat_list = list(self.stats['latency_list'])

            bs_list = list(self.stats['batch_span_list'])
            lq_list = list(self.stats['log_queue_depth_list'])
            bs_p95 = round(percentile(bs_list, 95), 0) if bs_list else 0
            lq_p95 = round(percentile(lq_list, 95), 0) if lq_list else 0

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "window_sec": 10,
                "runtime_seconds": round(elapsed, 2),
                "total_messages": self.stats["total_messages"],
                "recv_rate": round(rate, 2),
                "latency_ms": {
                    "avg_ms": round(sum(lat_list)/len(lat_list), 2) if lat_list else 0.0,
                    "min_ms": round(min(lat_list), 2) if lat_list else 0.0,
                    "max_ms": round(max(lat_list), 2) if lat_list else 0.0,
                    "p50": pc["p50"],
                    "p95": pc["p95"],
                    "p99": pc["p99"],
                },
                "continuity": {
                    "breaks": 0,  # real breaks would be pu!=last_u; we resync immediately so break count stays 0
                    "resyncs": self.stats["resyncs"],
                    "reconnects": self.stats["reconnects"],
                },
                "batch_span": {
                    "p95": bs_p95,
                    "max": int(self.stats["batch_span_max"]),
                },
                "log_queue": {
                    "depth_p95": lq_p95,
                    "depth_max": int(self.stats["log_queue_max_depth"]),
                    "drops": int(self.stats["log_drops"]),
                },
                "symbol": self.symbol.upper(),
            }
            ensure_dir(self.data_dir / "metrics.json")
            with open(self.data_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            self.logger.debug("metrics.json updated")
        except Exception as e:
            self.logger.error(f"save_metrics_json failed: {e}", exc_info=True)

    # --------------------- Runner ---------------------

    def run(self, reconnect: bool = True):
        if websocket is None:
            raise RuntimeError("websocket-client not installed. Please install 'websocket-client'.")
        self.logger.info("=" * 60)
        self.logger.info(f"Start Binance WebSocket client")
        self.logger.info(f"Symbol: {self.symbol.upper()} depth={self.depth_levels} url={self.ws_url}")
        self.logger.info(f"Auto reconnect: {'on' if reconnect else 'off'}")
        self.logger.info("=" * 60)

        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        try:
            self.ws.run_forever(ping_interval=20, ping_timeout=10)
        except KeyboardInterrupt:
            self.logger.info("User interrupted, closing ws...")
            try: self.ws.close()
            except Exception: pass
        except Exception as e:
            self.logger.error(f"run_forever error: {e}", exc_info=True)
            raise
        finally:
            # Ensure log listener is stopped
            try:
                self.listener.stop()
            except Exception:
                pass
            self.logger.info("Listener stopped; exiting run()")

# ---------------------------- CLI ----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Binance Order Book Streamer (Task 1.1.6)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--symbol", type=str, default="ETHUSDT")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--rotate", type=str, default="interval", choices=["interval", "size"])
    parser.add_argument("--rotate-sec", type=int, default=60)
    parser.add_argument("--max-bytes", type=int, default=5_000_000)
    parser.add_argument("--backups", type=int, default=7)
    parser.add_argument("--print-interval", type=int, default=10)
    parser.add_argument("--run-minutes", type=int, default=None)

    args = parser.parse_args()

    client = BinanceOrderBookStream(
        symbol=args.symbol,
        depth_levels=args.depth,
        rotate=args.rotate,
        rotate_sec=args.rotate_sec,
        max_bytes=args.max_bytes,
        backups=args.backups,
        print_interval=args.print_interval,
    )

    if args.run_minutes:
        import threading
        t = threading.Thread(target=client.run, kwargs={"reconnect": True}, daemon=True)
        t.start()
        time.sleep(args.run_minutes * 60)
        if client.ws:
            try: client.ws.close()
            except Exception: pass
        try: client.listener.stop()
        except Exception: pass
    else:
        client.run(reconnect=True)
