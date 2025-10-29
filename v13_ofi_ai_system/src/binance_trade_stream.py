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
import heapq
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict, List

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
# 延迟初始化日志级别（从配置加载器获取，默认INFO）
def _get_log_level(config_loader=None):
    """
    从配置加载器获取日志级别，如果未提供则使用默认值
    
    Args:
        config_loader: 统一配置加载器实例
    
    Returns:
        str: 日志级别（大写）
    """
    if config_loader:
        try:
            level = config_loader.get("logging.level", "INFO")
            return str(level).upper()
        except Exception:
            pass
    # 如果没有配置加载器，使用 defaults.yaml 中的默认值
    # 这些默认值与 defaults.yaml 保持一致，避免硬编码
    return "INFO"

# 初始化日志（使用默认值，实际运行时会通过配置加载器更新）
LOG_LEVEL_DEFAULT = "INFO"
log = logging.getLogger("binance_trade_stream")

def _setup_logging(config_loader=None):
    """设置日志配置"""
    log_level = _get_log_level(config_loader)
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        force=True  # 允许重新配置
    )
    log.setLevel(getattr(logging, log_level, logging.INFO))

# 初始设置（使用默认值）
_setup_logging()

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
    """全局监控指标 - P0-B增强版"""
    reconnect_count: int = 0
    queue_dropped: int = 0
    total_messages: int = 0
    parse_errors: int = 0
    # A2: 强制消息对账
    emitted_messages: int = 0     # 已发出消息数（水位线输出+flush）
    persisted_messages: int = 0   # 已落盘消息数
    # P0-B新增监控指标
    agg_dup_count: int = 0        # aggTradeId重复次数（a==last_a）
    agg_backward_count: int = 0  # aggTradeId倒序次数（a<last_a）
    late_event_dropped: int = 0   # 水位线后到达被丢弃的迟到事件
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
            # A2: 强制消息对账
            "emitted_messages": self.emitted_messages,
            "persisted_messages": self.persisted_messages,
            "persist_rate": (self.persisted_messages / self.emitted_messages) if self.emitted_messages > 0 else 0.0,
            # P0-B监控指标
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
        
        # A2: 计数已发出消息
        metrics.emitted_messages += len(output)
        
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
        
        # A2: 计数已发出消息（flush阶段）
        metrics.emitted_messages += len(output)
        
        return output

# ---- parsing ----
def parse_aggtrade_message(text: str) -> Optional[Tuple[float, float, bool, Optional[int], Optional[int]]]:
    """Support both stream-wrapped and raw aggTrade payloads.
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
        is_buy = (not m)  # m=isBuyerMaker: True means seller was aggressive -> is_buy=False
        et = payload.get("E") or payload.get("T")
        a = payload.get("a")  # aggTradeId - 真正的唯一标识符
        return p, q, is_buy, int(et) if et is not None else None, int(a) if a is not None else None
    except Exception:
        return None

# ---- websocket consumer ----
async def ws_consume(url: str, queue: asyncio.Queue, stop_evt: asyncio.Event, metrics: MonitoringMetrics, 
                     heartbeat_timeout: int = 30, backoff_max: int = 15, ping_interval: int = 20, 
                     close_timeout: int = 10):
    warn_rl = RateLimiter(max_per_sec=5)

    backoff = 1.0
    first_connect = True
    while not stop_evt.is_set():
        try:
            async with websockets.connect(url, ping_interval=ping_interval, close_timeout=close_timeout) as ws:
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

# ---- Trade Stream Processor Class ----
class TradeStreamProcessor:
    """交易流处理器 - 支持统一配置"""
    
    def __init__(self, config_loader=None):
        """
        初始化交易流处理器
        
        Args:
            config_loader: 配置加载器实例，用于从统一配置系统加载参数
        """
        if config_loader:
            # 从统一配置系统加载参数
            from trade_stream_config_loader import TradeStreamConfigLoader
            self.config_loader = TradeStreamConfigLoader(config_loader)
            self.config = self.config_loader.load_config()
        else:
            # 使用默认配置
            self.config_loader = None
            self.config = None
    
    def get_websocket_config(self):
        """获取WebSocket配置"""
        if self.config:
            return self.config.websocket
        else:
            # 默认配置
            from trade_stream_config_loader import WebSocketConfig
            return WebSocketConfig()
    
    def get_queue_config(self):
        """获取队列配置"""
        if self.config:
            return self.config.queue
        else:
            # 默认配置
            from trade_stream_config_loader import QueueConfig
            return QueueConfig()
    
    def get_logging_config(self):
        """获取日志配置"""
        if self.config:
            return self.config.logging
        else:
            # 默认配置
            from trade_stream_config_loader import LoggingConfig
            return LoggingConfig()
    
    def get_performance_config(self):
        """获取性能配置"""
        if self.config:
            return self.config.performance
        else:
            # 默认配置
            from trade_stream_config_loader import PerformanceConfig
            return PerformanceConfig()
    
    def get_monitoring_config(self):
        """获取监控配置"""
        if self.config:
            return self.config.monitoring
        else:
            # 默认配置
            from trade_stream_config_loader import MonitoringConfig
            return MonitoringConfig()
    
    async def start_stream(self, symbol: str, url: str = None, duration: Optional[int] = None):
        """
        启动交易流处理
        
        Args:
            symbol: 交易对符号
            url: WebSocket URL，默认从环境变量获取
            duration: 测试时长（秒），None表示无限运行
        """
        # 获取配置
        websocket_config = self.get_websocket_config()
        queue_config = self.get_queue_config()
        logging_config = self.get_logging_config()
        performance_config = self.get_performance_config()
        
        # 构建URL
        if url is None:
            # 优先从配置加载器获取
            if self.config_loader and self.config_loader._base_config_loader:
                try:
                    # 从 data_source.websocket 获取配置
                    ws_config = self.config_loader._base_config_loader.get("data_source.websocket", {})
                    base_url = ws_config.get("connection", {}).get("base_url")
                    if base_url:
                        url = f"{base_url}stream?streams={symbol.lower()}@aggTrade"
                    else:
                        # 使用默认URL构建
                        url = f"wss://fstream.binance.com/stream?streams={symbol.lower()}@aggTrade"
                except Exception:
                    # 如果获取失败，使用默认URL
                    url = f"wss://fstream.binance.com/stream?streams={symbol.lower()}@aggTrade"
            else:
                # 如果没有配置加载器，使用默认URL构建（不再读取环境变量）
                url = f"wss://fstream.binance.com/stream?streams={symbol.lower()}@aggTrade"
        
        # 创建队列和事件
        q: asyncio.Queue = asyncio.Queue(maxsize=queue_config.size)
        stop_evt = asyncio.Event()
        metrics = MonitoringMetrics()
        
        # 数据收集
        data_records = []
        start_time = time.time()
        
        # 创建数据目录
        from pathlib import Path
        import pandas as pd
        import json
        from datetime import datetime
        
        # 使用绝对路径：从src目录回到项目根目录
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data" / "cvd"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录（按日期和交易对）
        today = datetime.now().strftime("%Y%m%d")
        symbol_dir = data_dir / today / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        run_time = datetime.now().strftime("%H%M%S")
        output_file = symbol_dir / f"run_{run_time}_cvd_data.parquet"
        
        log.info(f"[DATA] 数据将保存到: {output_file}")
        
        # 设置信号处理
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
        prod = asyncio.create_task(ws_consume(
            url, q, stop_evt, metrics,
            heartbeat_timeout=websocket_config.heartbeat_timeout,
            backoff_max=websocket_config.backoff_max,
            ping_interval=websocket_config.ping_interval,
            close_timeout=websocket_config.close_timeout
        ))
        cons = asyncio.create_task(processor(
            symbol, q, stop_evt, metrics,
            watermark_ms=performance_config.watermark_ms,  # 使用性能配置的水位线
            print_every=logging_config.print_every,
            stats_interval=logging_config.stats_interval,
            data_records=data_records
        ))
        
        log.info("Starting trade stream for %s | url=%s | duration=%s", symbol, url, f"{duration}s" if duration else "unlimited")
        
        # 如果设置了duration，启动定时器
        timer_task = None
        if duration:
            async def timer():
                await asyncio.sleep(duration)
                log.info("Duration %ds reached, stopping stream...", duration)
                stop_evt.set()
            timer_task = asyncio.create_task(timer())
        
        try:
            await stop_evt.wait()
        except KeyboardInterrupt:
            _set_stop()
        finally:
            if timer_task:
                timer_task.cancel()
            prod.cancel()
            # 不要立即取消cons，让它自然结束以完成数据收集
            # cons.cancel()
            await asyncio.gather(prod, cons, timer_task, return_exceptions=True)
            log.info("Graceful shutdown. Final metrics: %s", metrics.to_dict())
            
            # 保存数据到parquet文件
            if data_records:
                log.info(f"[DATA] 保存 {len(data_records)} 条记录到 {output_file}")
                df = pd.DataFrame(data_records)
                df.to_parquet(output_file, index=False)
                log.info(f"[DATA] 数据已保存到: {output_file}")
                
                # 生成监控指标JSON
                end_time = time.time()
                duration_seconds = end_time - start_time
                
                metrics_data = {
                    "run_info": {
                        "start_time": datetime.fromtimestamp(start_time).isoformat(),
                        "end_time": datetime.fromtimestamp(end_time).isoformat(),
                        "duration_seconds": duration_seconds,
                        "total_records": len(data_records),
                        "symbol": symbol
                    },
                    "performance": {
                        "queue_dropped_rate": float(metrics.queue_dropped_rate()),
                        "parse_errors": int(metrics.parse_errors),
                        "reconnect_count": int(metrics.reconnect_count),
                        # A2: 对账字段
                        "emitted_messages": int(metrics.emitted_messages),
                        "persisted_messages": int(metrics.persisted_messages),
                        "persist_rate": (metrics.persisted_messages / metrics.emitted_messages) if metrics.emitted_messages > 0 else 0.0
                    },
                    "data_file": str(output_file)
                }
                
                # 保存监控指标
                metrics_file = symbol_dir / f"run_{run_time}_metrics.json"
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics_data, f, ensure_ascii=False, indent=2)
                log.info(f"[DATA] 监控指标已保存到: {metrics_file}")
            else:
                log.warning("[DATA] 没有数据记录需要保存")

# ---- processor ----
async def processor(symbol: str, queue: asyncio.Queue, stop_evt: asyncio.Event, metrics: MonitoringMetrics, 
                   watermark_ms: int = 2000, print_every: int = 100, stats_interval: float = 60.0, data_records: list = None):
    cfg = CVDConfig()
    calc = RealCVDCalculator(symbol=symbol, cfg=cfg)

    # P0-B: 初始化水位线缓冲
    watermark = WatermarkBuffer(watermark_ms=watermark_ms)
    log.info("[P0-B] Watermark buffer initialized: watermark_ms=%d", watermark_ms)

    perf = PerfStats()
    # 使用传入的参数，不再从环境变量读取
    processed = 0
    last_stats = time.time()
    
    # 如果没有提供data_records，创建一个新的
    if data_records is None:
        data_records = []
    
    # P3: 诊断抽样缓存
    last_diag_stats = {}

    while not stop_evt.is_set():
        # P1: 突发出队优化
        burst = []
        try:
            # 先阻塞拿1条，避免空转
            ts_recv, raw = await asyncio.wait_for(queue.get(), timeout=1.0)
            burst.append(raw)
            # 再尽量无阻塞多拿一些，最多拿到当前队列一半
            n_extra = max(0, queue.qsize() // 2)
            for _ in range(n_extra):
                ts_recv, raw_msg = queue.get_nowait()
                burst.append(raw_msg)
        except asyncio.TimeoutError:
            if (time.time() - last_stats) >= stats_interval:
                snap = perf.snapshot_and_reset()
                log.info("[STAT] trades=%d avg_proc=%.3fms | %s", 
                         snap["count"], snap["avg_ms"], metrics.to_dict())
                last_stats = time.time()
            continue
        except asyncio.QueueEmpty:
            pass  # 队列已空，正常继续处理burst中已有的消息

        t0 = time.time()
        
        # P1: 批量parse + 水位线
        ready_all = []
        for raw in burst:
            parsed = parse_aggtrade_message(raw)
            if parsed is None:
                metrics.parse_errors += 1
                log.warning("Parse error on message (truncated): %s", raw[:160])
                continue
            
            price, qty, is_buy, event_ms, agg_trade_id = parsed
            
            # P0-B: 送入水位线重排，返回排序后可输出的消息  
            ready_list = watermark.feed(event_ms, agg_trade_id, parsed, metrics)
            ready_all.extend(ready_list)
        
        # P1: 批量处理所有ready的消息
        for ready_parsed in ready_all:
            price_r, qty_r, is_buy_r, event_ms_r, agg_trade_id_r = ready_parsed
            ret = calc.update_with_trade(price=price_r, qty=qty_r, is_buy=is_buy_r, event_time_ms=event_ms_r)
            processed += 1
            dt = time.time() - t0
            perf.feed(dt)
            
            # 计算延迟（从交易所事件时间到现在）
            latency_ms = (time.time() * 1000 - event_ms_r) if event_ms_r else 0.0
            
            # B1: 获取CVD诊断信息（P3: 每50笔抽样一次）
            if processed % 50 == 0:
                try:
                    stats = calc.get_z_stats()
                    last_diag_stats = stats  # 更新缓存
                except Exception:
                    stats = last_diag_stats if last_diag_stats else {}
            else:
                stats = last_diag_stats if last_diag_stats else {}
            
            # 收集数据记录
            record = {
                'ts': time.time(),
                'event_time_ms': event_ms_r,
                'price': price_r,
                'qty': qty_r,
                'is_buy': is_buy_r,
                'agg_trade_id': agg_trade_id_r,
                'cvd': ret["cvd"],
                'z_cvd': ret["z_cvd"],
                'ema_cvd': ret["ema_cvd"],
                'meta.bad_points': ret["meta"]["bad_points"],
                'meta.warmup': ret["meta"]["warmup"],
                'meta.std_zero': ret["meta"]["std_zero"],
                # B1: CVD诊断字段
                'z_mode': stats.get('z_mode', calc.cfg.z_mode),
                'current_tps': stats.get('current_tps', 0.0),
                'sigma_floor': stats.get('sigma_floor', 0.0),
                'ewma_mix': stats.get('ewma_mix', 0.0),
                'floor_used': stats.get('floor_used', False),
                'reconnect_count': metrics.reconnect_count,
                'queue_dropped': metrics.queue_dropped,
                'total_messages': metrics.total_messages,
                'parse_errors': metrics.parse_errors,
                'latency_ms': latency_ms
            }
            data_records.append(record)
            # A2: 计数已落盘消息
            metrics.persisted_messages += 1

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
    
    # P0-B: 程序退出时，flush所有剩余消息（应用去重逻辑）
    log.info("[P0-B] Flushing watermark buffer: %d messages remaining", len(watermark.heap))
    remaining = watermark.flush_all(metrics)
    for ready_parsed in remaining:
        price_r, qty_r, is_buy_r, event_ms_r, agg_trade_id_r = ready_parsed
        ret = calc.update_with_trade(price=price_r, qty=qty_r, is_buy=is_buy_r, event_time_ms=event_ms_r)
        processed += 1
        
        # 收集flush阶段的数据记录
        latency_ms = (time.time() * 1000 - event_ms_r) if event_ms_r else 0.0
        
        # B1: 获取CVD诊断信息（flush阶段复用缓存）
        stats = last_diag_stats if last_diag_stats else {}
        
        record = {
            'ts': time.time(),
            'event_time_ms': event_ms_r,
            'price': price_r,
            'qty': qty_r,
            'is_buy': is_buy_r,
            'agg_trade_id': agg_trade_id_r,
            'cvd': ret["cvd"],
            'z_cvd': ret["z_cvd"],
            'ema_cvd': ret["ema_cvd"],
            'meta.bad_points': ret["meta"]["bad_points"],
            'meta.warmup': ret["meta"]["warmup"],
            'meta.std_zero': ret["meta"]["std_zero"],
            # B1: CVD诊断字段
            'z_mode': stats.get('z_mode', calc.cfg.z_mode),
            'current_tps': stats.get('current_tps', 0.0),
            'sigma_floor': stats.get('sigma_floor', 0.0),
            'ewma_mix': stats.get('ewma_mix', 0.0),
            'floor_used': stats.get('floor_used', False),
            'reconnect_count': metrics.reconnect_count,
            'queue_dropped': metrics.queue_dropped,
            'total_messages': metrics.total_messages,
            'parse_errors': metrics.parse_errors,
            'latency_ms': latency_ms
        }
        data_records.append(record)
        # A2: 计数已落盘消息
        metrics.persisted_messages += 1
    
    log.info("[P0-B] Flushed %d messages, total processed=%d", len(remaining), processed)

# ---- main ----
async def main(symbol: Optional[str] = None, url: Optional[str] = None, config_loader=None, duration: Optional[int] = None):
    """
    主函数 - 支持统一配置系统
    
    Args:
        symbol: 交易对符号
        url: WebSocket URL
        config_loader: 配置加载器实例
        duration: 测试时长（秒），None表示无限运行
    """
    # 优先从配置加载器获取symbol
    if symbol:
        sym = symbol.upper()
    elif config_loader:
        try:
            sym = config_loader.get("data_source.default_symbol", "ETHUSDT").upper()
        except Exception:
            sym = "ETHUSDT"
    else:
        # 如果没有配置加载器，使用默认symbol（不再读取环境变量）
        sym = "ETHUSDT"
    
    # 使用配置加载器设置日志
    if config_loader:
        _setup_logging(config_loader)
    
    # 使用新的交易流处理器
    processor = TradeStreamProcessor(config_loader=config_loader)
    await processor.start_stream(sym, url, duration)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default=None, help="symbol, e.g. ETHUSDT (default from ENV SYMBOL)")
    ap.add_argument("--url", type=str, default=None, help="override websocket URL (default from ENV WS_URL)")
    ap.add_argument("--duration", type=int, default=None, help="test duration in seconds (default: run indefinitely)")
    args = ap.parse_args()
    asyncio.run(main(symbol=args.symbol, url=args.url, duration=args.duration))
