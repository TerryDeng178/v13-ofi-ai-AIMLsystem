# -*- coding: utf-8 -*-
"""
run_realtime_cvd.py - Binance Trade Stream CVD实时计算与数据落盘

功能：
- 实时连接Binance aggTrade流
- 计算CVD指标（使用RealCVDCalculator）
- 落盘到Parquet文件（包含完整监控指标）
- 支持10-15分钟验收测试

使用方法：
    # 默认ETHUSDT，运行10分钟
    python run_realtime_cvd.py
    
    # 指定交易对和时长
    python run_realtime_cvd.py --symbol BTCUSDT --duration 900
    
    # 指定输出目录
    python run_realtime_cvd.py --output-dir ./data/cvd_test

环境变量：
    SYMBOL: 交易对（默认ETHUSDT）
    DURATION: 运行时长（秒，默认600）
    DATA_OUTPUT_DIR: 输出目录（默认./data/CVDTEST）
    PRINT_EVERY: 打印间隔（默认100条）
    LOG_LEVEL: 日志级别（默认INFO）
"""

from __future__ import annotations

import asyncio
import heapq
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# ---- import path helper ----
def _ensure_project_path():
    here = Path(__file__).resolve().parent.parent
    src_dir = here / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

_ensure_project_path()

try:
    from real_cvd_calculator import RealCVDCalculator, CVDConfig
except ImportError as e:
    raise RuntimeError("Cannot import real_cvd_calculator") from e

try:
    import websockets
except ImportError as e:
    raise RuntimeError("Missing dependency: websockets>=10,<13") from e

try:
    import pandas as pd
except ImportError as e:
    raise RuntimeError("Missing dependency: pandas (for Parquet export)") from e

# ---- logging ----
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("run_realtime_cvd")

# ---- data record ----
@dataclass
class CVDRecord:
    """单条CVD数据记录"""
    timestamp: float           # 接收时间戳（Unix秒）
    event_time_ms: Optional[int]  # 交易所事件时间（毫秒）
    agg_trade_id: Optional[int]   # Binance aggTradeId（真正的唯一标识）
    price: float              # 成交价格
    qty: float                # 成交数量
    is_buy: bool              # 买卖方向
    cvd: float                # CVD值
    z_cvd: Optional[float]    # Z-score
    ema_cvd: Optional[float]  # EMA
    warmup: bool              # warmup状态
    std_zero: bool            # 标准差为0标记
    bad_points: int           # 坏数据点计数
    queue_dropped: int        # 队列丢弃计数
    reconnect_count: int      # 重连计数
    latency_ms: float         # 延迟（毫秒）

# ---- monitoring metrics ----
@dataclass
class MonitoringMetrics:
    """监控指标 - P0-B增强版"""
    reconnect_count: int = 0
    # 指标口径分离：区分通道层面丢弃 vs 水位线迟到丢弃
    queue_dropped: int = 0        # 通道层面丢弃（队列满时丢弃）
    late_event_dropped: int = 0   # 水位线后迟到被丢弃的事件
    total_messages: int = 0
    parse_errors: int = 0
    # P0-B新增监控指标
    agg_dup_count: int = 0        # aggTradeId重复次数（a==last_a）
    agg_backward_count: int = 0  # aggTradeId倒序次数（a<last_a）
    buffer_size_samples: List[int] = None  # 缓冲队列大小采样
    
    def __post_init__(self):
        if self.buffer_size_samples is None:
            self.buffer_size_samples = []
    
    def queue_dropped_rate(self) -> float:
        return (self.queue_dropped / self.total_messages) if self.total_messages > 0 else 0.0
    
    def agg_dup_rate(self) -> float:
        return (self.agg_dup_count / self.total_messages) if self.total_messages > 0 else 0.0
    
    def buffer_size_p95(self) -> float:
        if not self.buffer_size_samples:
            return 0.0
        sorted_samples = sorted(self.buffer_size_samples)
        idx = int(len(sorted_samples) * 0.95)
        return float(sorted_samples[min(idx, len(sorted_samples) - 1)])
    
    def buffer_size_max(self) -> int:
        return max(self.buffer_size_samples) if self.buffer_size_samples else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconnect_count": self.reconnect_count,
            "queue_dropped": self.queue_dropped,
            "total_messages": self.total_messages,
            "parse_errors": self.parse_errors,
            "queue_dropped_rate": self.queue_dropped_rate(),
            "agg_dup_count": self.agg_dup_count,
            "agg_dup_rate": self.agg_dup_rate(),
            "agg_backward_count": self.agg_backward_count,
            "late_event_dropped": self.late_event_dropped,
            "buffer_size_p95": self.buffer_size_p95(),
            "buffer_size_max": self.buffer_size_max(),
        }

# ---- P0-B: 2s水位线重排器 ----
class WatermarkBuffer:
    """
    2s水位线重排缓冲：按 (event_time_ms, agg_trade_id) 排序输出
    - 检测倒序ID并resync
    - 统计buffer大小（p95/max）
    - 记录late_write（水位线外写入）
    """
    def __init__(self, watermark_ms: int = 2000):
        self.watermark_ms = watermark_ms
        self.last_a = -1  # 上次输出的aggTradeId
        self.heap: List[Tuple[int, int, Any]] = []  # (event_time_ms, agg_trade_id, parsed_data)
        self.late_writes = 0
    
    def feed(self, event_ms: int, agg_trade_id: int, parsed_data: Tuple, metrics: MonitoringMetrics) -> List[Tuple]:
        """
        输入一条消息，返回可以输出的消息列表（已排序）
        parsed_data = (price, qty, is_buy, event_time_ms, agg_trade_id)
        """
        # 预警：检测倒序/重复ID（仅记录，不阻断）
        if agg_trade_id < self.last_a:
            log.warning(
                "[WATERMARK] Backward agg_trade_id detected: %d < last=%d",
                agg_trade_id, self.last_a
            )
        elif agg_trade_id == self.last_a:
            log.warning(
                "[WATERMARK] Duplicate agg_trade_id detected: %d == last=%d",
                agg_trade_id, self.last_a
            )
        
        # 加入堆
        heapq.heappush(self.heap, (event_ms, agg_trade_id, parsed_data))
        
        # 采样缓冲大小
        if len(self.heap) > 0 and len(metrics.buffer_size_samples) < 100000:  # 限制采样数量
            metrics.buffer_size_samples.append(len(self.heap))
        
        # 水位线逻辑：输出所有 event_time_ms <= (now - watermark_ms) 的消息
        now_ms = int(time.time() * 1000)
        threshold_ms = now_ms - self.watermark_ms
        
        output = []
        while self.heap and self.heap[0][0] <= threshold_ms:
            event_ms_out, agg_id_out, data_out = heapq.heappop(self.heap)
            
            # 去重/去倒序：丢弃 agg_id <= last_a 的事件
            if agg_id_out <= self.last_a:
                if agg_id_out == self.last_a:
                    metrics.agg_dup_count += 1
                    log.warning(
                        "[WATERMARK] Dropped duplicate: agg_id=%d (dup_count=%d)",
                        agg_id_out, metrics.agg_dup_count
                    )
                else:
                    metrics.agg_backward_count += 1
                    log.warning(
                        "[WATERMARK] Dropped backward: agg_id=%d < last_a=%d (backward_count=%d)",
                        agg_id_out, self.last_a, metrics.agg_backward_count
                    )
                metrics.late_event_dropped += 1
                continue  # ✅ 不输出、不更新 last_a
            
            self.last_a = agg_id_out
            output.append(data_out)
        
        return output
    
    def force_flush_timeout(self, metrics: MonitoringMetrics) -> List[Tuple]:
        """强制flush超时消息（定时调用）"""
        now_ms = int(time.time() * 1000)
        threshold_ms = now_ms - self.watermark_ms
        
        output = []
        while self.heap and self.heap[0][0] <= threshold_ms:
            event_ms_out, agg_id_out, data_out = heapq.heappop(self.heap)
            
            # 去重/去倒序：丢弃 agg_id <= last_a 的事件
            if agg_id_out <= self.last_a:
                if agg_id_out == self.last_a:
                    metrics.agg_dup_count += 1
                else:
                    metrics.agg_backward_count += 1
                metrics.late_event_dropped += 1
                continue
            
            self.last_a = agg_id_out
            output.append(data_out)
        
        return output
    
    def flush_all(self, metrics: MonitoringMetrics) -> List[Tuple]:
        """强制输出所有剩余消息（程序退出时），同样应用去重逻辑"""
        output = []
        while self.heap:
            event_ms_out, agg_id_out, data_out = heapq.heappop(self.heap)
            
            # flush阶段同样去重/去倒序
            if agg_id_out <= self.last_a:
                if agg_id_out == self.last_a:
                    metrics.agg_dup_count += 1
                    log.warning(
                        "[FLUSH] Dropped duplicate: agg_id=%d (dup_count=%d)",
                        agg_id_out, metrics.agg_dup_count
                    )
                else:
                    metrics.agg_backward_count += 1
                    log.warning(
                        "[FLUSH] Dropped backward: agg_id=%d < last_a=%d (backward_count=%d)",
                        agg_id_out, self.last_a, metrics.agg_backward_count
                    )
                metrics.late_event_dropped += 1
                continue  # ✅ 不输出、不更新 last_a
            
            self.last_a = agg_id_out
            output.append(data_out)
        return output

# ---- parsing ----
def parse_aggtrade_message(text: str) -> Optional[tuple]:
    """Parse aggTrade message.
    Returns: (price, qty, is_buy, event_time_ms, agg_trade_id)
    """
    try:
        obj = json.loads(text)
    except Exception:
        return None
    payload = obj.get("data", obj)
    try:
        p = float(payload["p"])
        q = float(payload["q"])
        m = bool(payload["m"])
        is_buy = (not m)
        et = payload.get("E") or payload.get("T")
        a = payload.get("a")  # aggTradeId - 真正的唯一标识符
        return p, q, is_buy, int(et) if et is not None else None, int(a) if a is not None else None
    except Exception:
        return None

# ---- websocket consumer ----
async def ws_consume(url: str, queue: asyncio.Queue, stop_evt: asyncio.Event, metrics: MonitoringMetrics):
    heartbeat_timeout = 60
    backoff_max = 30
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
                        log.warning("Heartbeat timeout (>%ss). Reconnecting...", heartbeat_timeout)
                        raise
                    except websockets.ConnectionClosed as e:
                        log.warning("Connection closed: %s", e)
                        raise
                    
                    metrics.total_messages += 1
                    # 分析模式：阻塞不丢（实时灰度可通过DROP_OLD=true启用丢旧策略）
                    DROP_OLD = os.getenv("DROP_OLD", "false").lower() == "true"
                    if DROP_OLD and queue.full():
                        metrics.queue_dropped += 1
                        try:
                            _ = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await queue.put((time.time(), msg))  # 分析模式默认阻塞
        except Exception as e:
            log.warning("Reconnect due to error: %s", e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, float(backoff_max))

# ---- processor with data collection ----
async def processor(
    symbol: str,
    queue: asyncio.Queue,
    stop_evt: asyncio.Event,
    metrics: MonitoringMetrics,
    records: List[CVDRecord],
):
    # P1.1: 支持环境变量配置CVD计算器
    # Step 1: 支持稳健尺度地板配置
    # Step 1.6 基线配置（默认值）
    cfg = CVDConfig(
        z_mode=os.getenv("CVD_Z_MODE", "delta"),  # Step 1.6: delta模式
        half_life_trades=int(os.getenv("HALF_LIFE_TRADES", "300")),
        winsor_limit=float(os.getenv("WINSOR_LIMIT", "8.0")),
        freeze_min=int(os.getenv("FREEZE_MIN", "80")),  # Step 1.6: 80
        stale_threshold_ms=int(os.getenv("STALE_THRESHOLD_MS", "5000")),
        # Step 1.6 稳健尺度地板参数
        scale_mode=os.getenv("SCALE_MODE", "hybrid"),  # Step 1.6: hybrid模式
        ewma_fast_hl=int(os.getenv("EWMA_FAST_HL", "80")),
        mad_window_trades=int(os.getenv("MAD_WINDOW_TRADES", "300")),
        mad_scale_factor=float(os.getenv("MAD_SCALE_FACTOR", "1.4826")),
        # Step 1.6 微调参数
        scale_fast_weight=float(os.getenv("SCALE_FAST_WEIGHT", "0.35")),  # Step 1.6: 0.35
        scale_slow_weight=float(os.getenv("SCALE_SLOW_WEIGHT", "0.65")),  # Step 1.6: 0.65
        mad_multiplier=float(os.getenv("MAD_MULTIPLIER", "1.45")),  # Step 1.6: 1.45
        post_stale_freeze=int(os.getenv("POST_STALE_FREEZE", "2")),
    )
    calc = RealCVDCalculator(symbol=symbol, cfg=cfg)
    
    # P0-B: 初始化水位线缓冲
    watermark_ms = int(os.getenv("WATERMARK_MS", "2000"))
    watermark = WatermarkBuffer(watermark_ms=watermark_ms)
    log.info("[P0-B] Watermark buffer initialized: watermark_ms=%d", watermark_ms)
    
    print_every = int(os.getenv("PRINT_EVERY", "1000"))  # 分析模式：每1000条打印一次
    processed = 0
    last_metrics_time = time.time()
    last_watermark_flush = time.time()
    METRICS_FLUSH_INTERVAL_MS = 10000  # 每10秒刷新指标
    WATERMARK_FLUSH_INTERVAL_MS = 200  # 每200ms强制flush水位线
    
    while not stop_evt.is_set():
        try:
            ts_recv, raw = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        
        parsed = parse_aggtrade_message(raw)
        if parsed is None:
            metrics.parse_errors += 1
            log.warning("Parse error on message")
            continue
        
        price, qty, is_buy, event_ms, agg_trade_id = parsed
        
        # P0-B: 送入水位线重排，返回排序后可输出的消息
        ready_list = watermark.feed(event_ms, agg_trade_id, parsed, metrics)
        
        # 处理所有ready的消息
        for ready_parsed in ready_list:
            price_r, qty_r, is_buy_r, event_ms_r, agg_trade_id_r = ready_parsed
            ret = calc.update_with_trade(price=price_r, qty=qty_r, is_buy=is_buy_r, event_time_ms=event_ms_r)
            processed += 1
            
            # 计算延迟
            latency_ms = (time.time() * 1000 - event_ms_r) if event_ms_r else 0.0
            
            # 记录数据
            record = CVDRecord(
                timestamp=ts_recv,
                event_time_ms=event_ms_r,
                agg_trade_id=agg_trade_id_r,
                price=price_r,
                qty=qty_r,
                is_buy=is_buy_r,
                cvd=ret["cvd"],
                z_cvd=ret["z_cvd"],
                ema_cvd=ret["ema_cvd"],
                warmup=ret["meta"]["warmup"],
                std_zero=ret["meta"]["std_zero"],
                bad_points=ret["meta"]["bad_points"],
                queue_dropped=metrics.queue_dropped,
                reconnect_count=metrics.reconnect_count,
                latency_ms=latency_ms,
            )
            records.append(record)
            
            if processed % print_every == 0 or processed <= 5:
                log.info(
                    "CVD %s | count=%d cvd=%.6f z=%s ema=%.6f | bad=%d dropped=%d reconnect=%d latency=%.1fms",
                    symbol, processed,
                    ret["cvd"],
                    "None" if ret["z_cvd"] is None else f"{ret['z_cvd']:.3f}",
                    ret["ema_cvd"] if ret["ema_cvd"] is not None else float('nan'),
                    ret["meta"]["bad_points"],
                    metrics.queue_dropped,
                    metrics.reconnect_count,
                    latency_ms,
                )
        
        # 定时flush水位线（每200ms）
        current_time = time.time()
        if (current_time - last_watermark_flush) * 1000 >= WATERMARK_FLUSH_INTERVAL_MS:
            # 强制flush水位线中的超时消息
            ready_list = watermark.force_flush_timeout(metrics)
            for ready_parsed in ready_list:
                price_r, qty_r, is_buy_r, event_ms_r, agg_trade_id_r = ready_parsed
                ret = calc.update_with_trade(price=price_r, qty=qty_r, is_buy=is_buy_r, event_time_ms=event_ms_r)
                processed += 1
                
                latency_ms = (time.time() * 1000 - event_ms_r) if event_ms_r else 0.0
                
                record = CVDRecord(
                    timestamp=time.time(),  # 使用当前时间，不复用旧的ts_recv
                    event_time_ms=event_ms_r,
                    agg_trade_id=agg_trade_id_r,
                    price=price_r,
                    qty=qty_r,
                    is_buy=is_buy_r,
                    cvd=ret["cvd"],
                    z_cvd=ret["z_cvd"],
                    ema_cvd=ret["ema_cvd"],
                    warmup=ret["meta"]["warmup"],
                    std_zero=ret["meta"]["std_zero"],
                    bad_points=ret["meta"]["bad_points"],
                    queue_dropped=metrics.queue_dropped,
                    reconnect_count=metrics.reconnect_count,
                    latency_ms=latency_ms,
                )
                records.append(record)
            
            last_watermark_flush = current_time
    
    # P0-B: 程序退出时，flush所有剩余消息（应用去重逻辑）
    log.info("[P0-B] Flushing watermark buffer: %d messages remaining", len(watermark.heap))
    remaining = watermark.flush_all(metrics)
    for ready_parsed in remaining:
        price_r, qty_r, is_buy_r, event_ms_r, agg_trade_id_r = ready_parsed
        ret = calc.update_with_trade(price=price_r, qty=qty_r, is_buy=is_buy_r, event_time_ms=event_ms_r)
        processed += 1
        
        # 计算延迟
        latency_ms = (time.time() * 1000 - event_ms_r) if event_ms_r else 0.0
        
        # 记录数据
        record = CVDRecord(
            timestamp=time.time(),
            event_time_ms=event_ms_r,
            agg_trade_id=agg_trade_id_r,
            price=price_r,
            qty=qty_r,
            is_buy=is_buy_r,
            cvd=ret["cvd"],
            z_cvd=ret["z_cvd"],
            ema_cvd=ret["ema_cvd"],
            warmup=ret["meta"]["warmup"],
            std_zero=ret["meta"]["std_zero"],
            bad_points=ret["meta"]["bad_points"],
            queue_dropped=metrics.queue_dropped,
            reconnect_count=metrics.reconnect_count,
            latency_ms=latency_ms,
        )
        records.append(record)
    log.info("[P0-B] Flushed %d messages, total processed=%d", len(remaining), processed)

# ---- main test ----
async def main():
    import argparse
    ap = argparse.ArgumentParser(description="Real-time CVD calculation with data export")
    ap.add_argument("--symbol", type=str, default=None, help="Trading symbol (default: ETHUSDT)")
    ap.add_argument("--duration", type=int, default=None, help="Test duration in seconds (default: 600)")
    ap.add_argument("--output-dir", type=str, default=None, help="Output directory (default: ./data/CVDTEST)")
    args = ap.parse_args()
    
    # 配置
    symbol = (args.symbol or os.getenv("SYMBOL", "ETHUSDT")).upper()
    duration = args.duration or int(os.getenv("DURATION", "600"))
    output_dir = Path(args.output_dir or os.getenv("DATA_OUTPUT_DIR", "./data/CVDTEST"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = f"wss://fstream.binancefuture.com/stream?streams={symbol.lower()}@aggTrade"
    
    # 初始化 - 分析模式：阻塞不丢，队列足够大
    queue_size = 50000  # 分析模式：50k队列，阻塞不丢
    q: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    stop_evt = asyncio.Event()
    metrics = MonitoringMetrics()
    records: List[CVDRecord] = []
    
    # 批量写盘配置（暂时保持内存，最后统一导出）
    BATCH_SIZE = 200  # 每200条批量写盘
    BATCH_MS = 200    # 每200ms批量写盘
    
    # 信号处理
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
    
    # 启动任务
    prod = asyncio.create_task(ws_consume(url, q, stop_evt, metrics))
    cons = asyncio.create_task(processor(symbol, q, stop_evt, metrics, records))
    
    start_time = time.time()
    log.info("=" * 60)
    log.info("Starting CVD real-time test")
    log.info("Symbol: %s", symbol)
    log.info("Duration: %d seconds", duration)
    log.info("Output: %s", output_dir)
    log.info("=" * 60)
    
    try:
        # 等待指定时长或手动中断
        await asyncio.wait_for(stop_evt.wait(), timeout=duration)
    except asyncio.TimeoutError:
        log.info("Test duration reached (%ds). Stopping...", duration)
        _set_stop()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt detected. Stopping...")
        _set_stop()
    finally:
        prod.cancel()
        cons.cancel()
        await asyncio.gather(prod, cons, return_exceptions=True)
        
        elapsed = time.time() - start_time
        log.info("=" * 60)
        log.info("Test completed")
        log.info("Elapsed: %.1f seconds", elapsed)
        log.info("Records collected: %d", len(records))
        log.info("Final metrics: %s", metrics.to_dict())
        log.info("=" * 60)
        
        # 导出数据
        if records:
            # 转换为DataFrame
            df = pd.DataFrame([asdict(r) for r in records])
            
            # 导出Parquet
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            parquet_file = output_dir / f"cvd_{symbol.lower()}_{timestamp_str}.parquet"
            df.to_parquet(parquet_file, index=False)
            log.info("✅ Exported %d records to: %s", len(records), parquet_file)
            
            # 生成报告
            report = {
                "test_info": {
                    "symbol": symbol,
                    "duration_planned": duration,
                    "duration_actual": elapsed,
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "end_time": datetime.fromtimestamp(time.time()).isoformat(),
                },
                "data_stats": {
                    "total_records": len(records),
                    "avg_rate_per_sec": len(records) / elapsed if elapsed > 0 else 0,
                    "cvd_range": [float(df["cvd"].min()), float(df["cvd"].max())],
                    "z_cvd_stats": {
                        "p50": float(df[df["z_cvd"].notna()]["z_cvd"].quantile(0.5)) if not df[df["z_cvd"].notna()].empty else None,
                        "p95": float(df[df["z_cvd"].notna()]["z_cvd"].quantile(0.95)) if not df[df["z_cvd"].notna()].empty else None,
                        "p99": float(df[df["z_cvd"].notna()]["z_cvd"].quantile(0.99)) if not df[df["z_cvd"].notna()].empty else None,
                    },
                    "latency_stats": {
                        "p50": float(df["latency_ms"].quantile(0.5)),
                        "p95": float(df["latency_ms"].quantile(0.95)),
                        "p99": float(df["latency_ms"].quantile(0.99)),
                    },
                },
                "metrics": metrics.to_dict(),
                "validation": {
                    "duration_ok": elapsed >= duration * 0.95,  # 允许5%误差
                    "parse_errors_ok": metrics.parse_errors == 0,
                    "queue_dropped_rate_ok": metrics.queue_dropped_rate() <= 0.005,  # ≤0.5%
                    "latency_p95_ok": float(df["latency_ms"].quantile(0.95)) < 5000,  # <5s (宽松)
                    "reconnect_ok": metrics.reconnect_count <= 3,  # ≤3次
                },
            }
            
            report_file = output_dir / f"report_{symbol.lower()}_{timestamp_str}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            log.info("✅ Report saved to: %s", report_file)
            
            # 打印验收结果
            log.info("=" * 60)
            log.info("VALIDATION RESULTS:")
            for key, passed in report["validation"].items():
                log.info("  %s: %s", key, "✅ PASS" if passed else "❌ FAIL")
            log.info("=" * 60)
        else:
            log.warning("⚠️ No records collected!")

if __name__ == "__main__":
    asyncio.run(main())

