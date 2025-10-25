#!/usr/bin/env python3
"""
Task 1.5 核心算法v1 - 信号层实现
直接调用成熟组件：OFI计算器、CVD计算器、融合指标、背离检测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import json
from pathlib import Path
import sys
import os
import time
import threading
import queue
import atexit

# 模块级logger
logger = logging.getLogger(__name__)
logger.propagate = True
import sqlite3

# 添加src路径以导入成熟组件
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 导入成熟组件
from real_ofi_calculator import RealOFICalculator, OFIConfig
from real_cvd_calculator import RealCVDCalculator, CVDConfig
from ofi_cvd_fusion import OFI_CVD_Fusion, OFICVDFusionConfig
from ofi_cvd_divergence import DivergenceDetector, DivergenceConfig
from utils.strategy_mode_manager import StrategyModeManager, StrategyMode, MarketActivity
from utils.config_loader import load_config, get_config

# ---- 可插拔 Sink 抽象 ----
class SignalSink:
    def emit(self, entry: dict): ...
    def close(self): ...

# ---- SafeJsonlWriter 安全I/O写入器 ----
class SafeJsonlWriter:
    def __init__(self, base_dir: Path, max_retries: int = 5, retry_sleep: float = 0.05):
        self.base_dir = Path(base_dir)
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

    def write_line(self, rel_path: Path, payload: dict):
        path = self.base_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False) + "\n"

        last_err = None
        for _ in range(self.max_retries):
            try:
                # 逐条追加 + 立刻冲刷，降低数据丢失
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
                return
            except Exception as e:
                last_err = e
                time.sleep(self.retry_sleep)

        # 重试后仍失败：把错误丢到主 logger，但不让核心逻辑崩溃
        raise last_err

# ---- JSONL 分片 Sink：spool/.part 写 → 原子换名到 ready/.jsonl ----
class JsonlSink(SignalSink):
    def __init__(self, base_dir: Path, batch_n: int = 50, fsync_ms: int = 200):
        self.base_dir = Path(base_dir)
        self.spool = self.base_dir / "spool"
        self.ready = self.base_dir / "ready"
        self.spool.mkdir(parents=True, exist_ok=True)
        self.ready.mkdir(parents=True, exist_ok=True)
        self.q = queue.Queue(maxsize=10000)
        self.batch_n = batch_n
        self.fsync_ms = fsync_ms
        self.state = {}  # symbol -> {"minute", "fh", "path", "count", "last_fsync"}
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def emit(self, entry: dict):
        try:
            self.q.put(entry, timeout=0.2)
        except queue.Full:
            # 按需计数告警；这里丢弃以保障交易主路径
            pass

    def _minute_key(self, ts_ms: int) -> str:
        # 用事件时间分片（UTC）；如需本地时间可改为 .fromtimestamp(..., tz=)
        dt = datetime.utcfromtimestamp((ts_ms or int(time.time()*1000)) / 1000.0)
        return dt.strftime("%Y%m%d_%H%M")

    def _open_state(self, symbol: str, minute: str):
        pid = os.getpid()
        subdir = self.spool / "signal" / symbol
        subdir.mkdir(parents=True, exist_ok=True)
        part = subdir / f"signals_{minute}_{pid}.part"
        fh = open(part, "a", encoding="utf-8", buffering=1)
        return {"minute": minute, "fh": fh, "path": part, "count": 0, "last_fsync": time.time()}

    def _rotate(self, symbol: str, st: dict):
        try:
            st["fh"].flush(); os.fsync(st["fh"].fileno()); st["fh"].close()
        except Exception:
            pass
        # 原子换名到 ready 目录
        rel = st["path"].relative_to(self.spool)
        ready_path = (self.ready / rel).with_suffix(".jsonl")
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(st["path"], ready_path)
        except FileNotFoundError:
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                entry = self.q.get(timeout=0.1)
            except queue.Empty:
                # 周期性批量 fsync
                now = time.time()
                for st in list(self.state.values()):
                    if st["fh"] and (now - st["last_fsync"]) * 1000 >= self.fsync_ms:
                        try:
                            st["fh"].flush(); os.fsync(st["fh"].fileno()); st["last_fsync"] = now
                        except Exception:
                            pass
                continue

            if entry is None:
                break

            symbol = entry.get("symbol", "UNKNOWN")
            ts_ms = entry.get("ts_ms", int(time.time() * 1000))
            minute = self._minute_key(ts_ms)
            st = self.state.get(symbol)

            if (st is None) or (st["minute"] != minute):
                if st is not None:
                    self._rotate(symbol, st)
                st = self._open_state(symbol, minute)
                self.state[symbol] = st

            line = json.dumps(entry, ensure_ascii=False) + "\n"
            try:
                st["fh"].write(line)
                st["count"] += 1
                if st["count"] % self.batch_n == 0:
                    st["fh"].flush(); os.fsync(st["fh"].fileno()); st["last_fsync"] = time.time()
            except Exception:
                pass

        # drain：关闭并换名所有打开文件
        for sym, st in list(self.state.items()):
            self._rotate(sym, st)
        self.state.clear()

    def close(self):
        self._stop.set()
        try:
            self.q.put(None, timeout=0.2)
        except queue.Full:
            pass
        self.t.join(timeout=2.0)

# ---- SQLite（WAL）Sink（可选，适合 Windows/并发读） ----
class SqliteSink(SignalSink):
    def __init__(self, base_dir: Path, batch_n: int = 200, flush_ms: int = 300):
        self.db_path = Path(base_dir) / "signals.db"
        self.batch_n = batch_n
        self.flush_ms = flush_ms
        self.q = queue.Queue(maxsize=20000)
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self._init_db()
        self.t.start()

    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            con.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    ts_ms INTEGER,
                    symbol TEXT,
                    score REAL, z_ofi REAL, z_cvd REAL,
                    regime TEXT, div_type TEXT,
                    confirm INTEGER, gating INTEGER
                );
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_signals_sym_ts ON signals(symbol, ts_ms);")
            con.commit()
        finally:
            con.close()

    def emit(self, entry: dict):
        try:
            self.q.put(entry, timeout=0.2)
        except queue.Full:
            pass

    def _run(self):
        con = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            buf, last = [], time.time()
            while not self._stop.is_set():
                timeout = max(0.0, self.flush_ms/1000 - (time.time()-last))
                try:
                    item = self.q.get(timeout=timeout)
                except queue.Empty:
                    item = None
                if item is not None:
                    buf.append((
                        item.get("ts_ms"), item.get("symbol"),
                        item.get("score"), item.get("z_ofi"), item.get("z_cvd"),
                        item.get("regime"), item.get("div_type"),
                        int(bool(item.get("confirm"))), int(bool(item.get("gating")))
                    ))
                if buf and (len(buf) >= self.batch_n or (time.time()-last)*1000 >= self.flush_ms or item is None):
                    con.executemany(
                        "INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?);", buf
                    )
                    con.commit()
                    buf.clear(); last = time.time()
            if buf:
                con.executemany("INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?);", buf)
                con.commit()
        finally:
            con.close()

    def close(self):
        self._stop.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self.t.join(timeout=2.0)

# ---- 空 Sink（S0 诊断用） ----
class NullSink(SignalSink):
    def emit(self, entry: dict): pass
    def close(self): pass

class AsyncJsonlWriter:
    """进程内单线程写 JSONL；不同进程自动写入不同文件（_PID 后缀）"""
    _instances = {}           # base_dir -> writer
    _instances_lock = threading.Lock()

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.q = queue.Queue(maxsize=10000)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    @classmethod
    def get(cls, base_dir: Path):
        with cls._instances_lock:
            key = str(Path(base_dir).resolve())
            inst = cls._instances.get(key)
            if inst is None:
                inst = AsyncJsonlWriter(base_dir)
                cls._instances[key] = inst
            return inst

    def write_line(self, rel_path: Path, payload: dict, timeout: float = 0.2):
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            self.q.put((rel_path, line), timeout=timeout)
        except queue.Full:
            # 可计数告警；为简洁这里直接丢弃
            pass

    def _run(self):
        cur_path = None
        f = None
        pid = os.getpid()
        while True:
            rel_path, line = self.q.get()
            if rel_path is None:
                break
            # 每个进程自己的文件：signals_YYYYMMDD_PID.jsonl
            path = self.base_dir / rel_path
            path = path.with_name(f"{path.stem}_{pid}{path.suffix}")
            path.parent.mkdir(parents=True, exist_ok=True)

            if cur_path != path:
                if f:
                    try: f.flush(); os.fsync(f.fileno()); f.close()
                    except Exception: pass
                f = open(path, "a", encoding="utf-8", buffering=1)
                cur_path = path
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                # 不中断主流程；可扩展成 fallback 文件
                pass

@dataclass
class SignalConfig:
    """信号配置"""
    # 融合权重
    w_ofi: float = 0.6
    w_cvd: float = 0.4
    
    # 裁剪范围
    score_clip_min: float = -5.0
    score_clip_max: float = 5.0
    
    # EMA平滑参数
    ema_alpha: float = 0.2
    
    # 背离检测参数
    swing_L: int = 12
    z_hi: float = 2.0
    z_mid: float = 1.0
    
    # 护栏参数
    spread_bps_cap: float = 15.0
    missing_msgs_rate_cap: float = 0.001
    resync_cooldown_sec: int = 120
    reconnect_cooldown_sec: int = 180
    cooldown_after_exit_sec: int = 60

@dataclass
class SignalData:
    """信号数据"""
    ts_ms: int
    symbol: str
    score: float
    z_ofi: float
    z_cvd: float
    regime: str
    div_type: Optional[str] = None
    confirm: bool = False
    gating: bool = False

class CoreAlgorithm:
    """核心算法类 - 直接调用成熟组件"""
    
    def __init__(self, symbol: str, config: SignalConfig = None, config_loader=None):
        self.symbol = symbol
        self.config = config or SignalConfig()
        self.config_loader = config_loader
        self.logger = logging.getLogger(__name__)
        
        # 加载统一配置
        self.system_config = load_config() if config_loader is None else config_loader
        
        # 直接初始化成熟组件
        self._init_components()
        
        # 状态变量
        self.last_resync_time = 0
        self.last_reconnect_time = 0
        self.last_exit_time = 0
        
        # 护栏状态
        self.guard_active = False
        self.guard_reason = ""
        
        # 统计信息
        self.stats = {
            'total_updates': 0,
            'valid_signals': 0,
            'suppressed_signals': 0,
            'regime_changes': 0,
            'ofi_updates': 0,
            'cvd_updates': 0,
            'fusion_updates': 0,
            'divergence_events': 0
        }
        
        # 闸门原因统计 - 用于诊断信号被阻止的原因
        self.gate_reason_stats = {
            'warmup_guard': 0,
            'lag_exceeded': 0,
            'consistency_low': 0,
            'divergence_blocked': 0,
            'scenario_blocked': 0,
            'spread_too_high': 0,
            'missing_msgs_rate': 0,
            'resync_cooldown': 0,
            'reconnect_cooldown': 0,
            'component_warmup': 0,
            'weak_signal_throttle': 0,
            'low_consistency': 0,
            'reverse_cooldown': 0,
            'insufficient_hold_time': 0
        }
        
        # 反向开仓防抖状态
        self._last_trade_time = {}  # symbol -> timestamp
        self._last_trade_direction = {}  # symbol -> direction
        self._consecutive_same_direction = {}  # symbol -> count
        
        # 从配置加载场景参数
        scenario_params = self.system_config.get('scenario_parameters', {})
        self._scenario_hold_times = {}
        self._scenario_consistency_thresholds = {}
        
        for scenario, params in scenario_params.items():
            self._scenario_hold_times[scenario] = params.get('min_hold_time_sec', 60)
            self._scenario_consistency_thresholds[scenario] = params.get('consistency_threshold', 0.6)
        
        # 初始化场景参数热更新支持
        self._scenario_params = scenario_params.copy()
        self._scenario_config_version = 'initial'
        
        # 新增：场景判定与统一取参
        self._current_scenario = 'A_H'
        self._gate_stats = {'weak': 0, 'low_consistency': 0, 'min_hold': 0}
        self.vol_tracker = None  # 波动率跟踪器
        
        # 初始化异步写入器（延迟初始化）
        self._async_writer = None
        self._output_dir = Path(os.getenv("V13_OUTPUT_DIR", "./runtime"))
    
    def update_scenario_params(self, scenario: str, params: Dict[str, Any]) -> bool:
        """
        热更新指定场景的参数
        
        Args:
            scenario: 场景名称 (A_H, A_L, Q_H, Q_L)
            params: 新参数
            
        Returns:
            bool: 是否成功
        """
        try:
            # 更新场景参数
            self._scenario_params[scenario] = params.copy()
            
            # 更新相关阈值
            if 'min_hold_time_sec' in params:
                self._scenario_hold_times[scenario] = params['min_hold_time_sec']
            if 'consistency_threshold' in params:
                self._scenario_consistency_thresholds[scenario] = params['consistency_threshold']
            
            # 更新配置版本
            self._scenario_config_version = f"{int(time.time())}"
            
            logger.info(f"Updated scenario params for {scenario}: {list(params.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update scenario params for {scenario}: {e}")
            return False
    
    def determine_scenario(self, is_active: bool, realized_vol: float, vol_split: float = 0.5):
        """
        场景判定：Active/Quiet × 高/低波动 → 2×2 场景标签
        
        Args:
            is_active: 是否活跃
            realized_vol: 实现波动率
            vol_split: 波动率分位点阈值
            
        Returns:
            str: 场景标签 (A_H, A_L, Q_H, Q_L)
        """
        if self.vol_tracker is None:
            # 初始化波动率跟踪器
            from collections import deque
            self.vol_tracker = deque(maxlen=100)
        
        self.vol_tracker.append(realized_vol)
        if len(self.vol_tracker) < 10:
            # 数据不足时使用默认场景
            return 'A_H' if is_active else 'Q_L'
        
        # 计算波动率分位点
        vol_percentile = sorted(self.vol_tracker)[int(len(self.vol_tracker) * vol_split)]
        hi = realized_vol >= vol_percentile
        
        return ('A_' if is_active else 'Q_') + ('H' if hi else 'L')
    
    def set_gates(self, weak_signal_threshold: float = 0.15, consistency_min: float = 0.5, min_hold_time_sec: int = 30):
        """
        设置闸门参数
        
        Args:
            weak_signal_threshold: 弱信号阈值
            consistency_min: 最小一致性
            min_hold_time_sec: 最小持仓时间
        """
        self._weak_signal_threshold = weak_signal_threshold
        self._consistency_min = consistency_min
        self._min_hold_time_sec = min_hold_time_sec
        logger.info(f"Set gates: weak={weak_signal_threshold}, consistency={consistency_min}, hold={min_hold_time_sec}")
    
    def set_targets(self, tp_bps: float = 25, sl_bps: float = 12):
        """
        设置止盈止损目标
        
        Args:
            tp_bps: 止盈基点
            sl_bps: 止损基点
        """
        self._tp_bps = tp_bps
        self._sl_bps = sl_bps
        logger.info(f"Set targets: TP={tp_bps}bps, SL={sl_bps}bps")
        
        # 可插拔Sink系统初始化
        base_dir = Path(os.getenv("V13_OUTPUT_DIR", "./runtime"))
        sink_type = os.getenv("V13_SINK", "jsonl").lower()
        if sink_type == "sqlite":
            self._sink = SqliteSink(base_dir)
        elif sink_type == "null":
            self._sink = NullSink()
        else:
            self._sink = JsonlSink(base_dir)

        # 优雅关闭（进程退出时 flush/close）
        atexit.register(lambda: getattr(self, "_sink", None) and self._sink.close())
    
    def _init_components(self):
        """初始化成熟组件 - 使用统一配置系统"""
        try:
            # 从统一配置获取参数
            fusion_config = self.system_config.get('fusion_metrics', {})
            divergence_config = self.system_config.get('divergence_detection', {})
            strategy_config = self.system_config.get('strategy', {})
            
            # 1. OFI计算器 - 使用优化后的参数
            ofi_config = OFIConfig(
                levels=5,
                z_window=150,  # 使用优化后的窗口大小
                ema_alpha=self.config.ema_alpha
            )
            self.ofi_calc = RealOFICalculator(self.symbol, ofi_config, self.config_loader)
            
            # 2. CVD计算器 - 使用优化后的参数
            cvd_config = CVDConfig(
                z_window=150,  # 使用优化后的窗口大小
                ema_alpha=self.config.ema_alpha,
                use_tick_rule=True,
                warmup_min=3  # 使用优化后的暖启动阈值
            )
            self.cvd_calc = RealCVDCalculator(self.symbol, cvd_config, self.config_loader)
            
            # 3. 融合指标 - 使用统一配置
            fusion_weights = fusion_config.get('weights', {})
            fusion_thresholds = fusion_config.get('thresholds', {})
            fusion_config_obj = OFICVDFusionConfig(
                w_ofi=fusion_weights.get('w_ofi', self.config.w_ofi),
                w_cvd=fusion_weights.get('w_cvd', self.config.w_cvd),
                fuse_buy=fusion_thresholds.get('fuse_buy', 1.5),
                fuse_strong_buy=fusion_thresholds.get('fuse_strong_buy', 2.5),
                fuse_sell=fusion_thresholds.get('fuse_sell', -1.5),
                fuse_strong_sell=fusion_thresholds.get('fuse_strong_sell', -2.5)
            )
            self.fusion = OFI_CVD_Fusion(fusion_config_obj, self.config_loader)
            
            # 4. 背离检测器 - 使用统一配置
            divergence_default = divergence_config.get('default', {})
            divergence_config_obj = DivergenceConfig(
                swing_L=divergence_default.get('swing_L', self.config.swing_L),
                z_hi=divergence_default.get('z_hi', self.config.z_hi),
                z_mid=divergence_default.get('z_mid', self.config.z_mid),
                min_separation=divergence_default.get('min_separation', 6),
                cooldown_secs=divergence_default.get('cooldown_secs', 1.0)
            )
            self.divergence = DivergenceDetector(divergence_config_obj, self.config_loader)
            
            # 5. 策略模式管理器 - 使用统一配置
            if not strategy_config:
                # 如果没有策略配置，使用默认配置
                strategy_config = {
                    'strategy': {
                        'mode': 'auto',
                        'hysteresis': {
                            'window_secs': 60,
                            'min_active_windows': 3,
                            'min_quiet_windows': 6
                        },
                        'triggers': {
                            'schedule': {
                                'enabled': True,
                                'timezone': 'Asia/Hong_Kong',
                                'calendar': 'CRYPTO',
                                'enabled_weekdays': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                                'holidays': [],
                                'active_windows': [],
                                'wrap_midnight': True
                            },
                            'market': {
                                'enabled': True,
                                'window_secs': 60,
                                'min_trades_per_min': 100.0,
                                'min_quote_updates_per_sec': 100,
                                'max_spread_bps': self.config.spread_bps_cap,
                                'min_volatility_bps': 0.02 * 10000,
                                'min_volume_usd': 1000000,
                                'use_median': True,
                                'winsorize_percentile': 95
                            }
                        }
                    }
                }
            self.strategy_manager = StrategyModeManager(strategy_config, self.config_loader)
            
            # 关键：把组件引用挂载到策略管理器，让2x2场景参数能够真正下发
            self.strategy_manager.core_algo = self
            self.strategy_manager.ofi_calc = self.ofi_calc
            self.strategy_manager.cvd_calc = self.cvd_calc
            self.strategy_manager.fusion_component = self.fusion
            
            self.logger.info(f"Core algorithm components initialized for {self.symbol} with unified config")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components for {self.symbol}: {e}")
            raise
        
    def update_ofi(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], 
                   event_time_ms: Optional[int] = None) -> Dict[str, Any]:
        """更新OFI计算器"""
        try:
            result = self.ofi_calc.update_with_snapshot(bids, asks, event_time_ms)
            self.stats['ofi_updates'] += 1
            
            # OFI质量诊断：检查数据质量指标
            meta = result.get('meta', {})
            bad_points = meta.get('bad_points', 0)
            std_zero = meta.get('std_zero', False)
            
            # 检查数据质量并记录到统计
            if bad_points > 0:
                self.gate_reason_stats['ofi_bad_data'] = self.gate_reason_stats.get('ofi_bad_data', 0) + 1
            
            if std_zero:
                self.gate_reason_stats['ofi_std_zero'] = self.gate_reason_stats.get('ofi_std_zero', 0) + 1
            
            # 记录跳价统计
            jump_stats = {
                'bid_jump_up': meta.get('bid_jump_up_cnt', 0),
                'bid_jump_down': meta.get('bid_jump_down_cnt', 0),
                'ask_jump_up': meta.get('ask_jump_up_cnt', 0),
                'ask_jump_down': meta.get('ask_jump_down_cnt', 0)
            }
            
            # 如果数据质量异常，标记为warmup
            if bad_points > 5 or std_zero:  # 阈值可调
                result['meta']['data_quality_issue'] = True
                result['z_ofi'] = None  # 强制返回None，触发warmup
            
            return result
        except Exception as e:
            self.logger.error(f"OFI calculation failed: {e}")
            return {"z_ofi": None, "ofi": 0.0, "meta": {"warmup": True}}
    
    def update_cvd(self, price: Optional[float] = None, qty: float = 0.0, 
                  is_buy: Optional[bool] = None, event_time_ms: Optional[int] = None) -> Dict[str, Any]:
        """更新CVD计算器"""
        try:
            result = self.cvd_calc.update_with_trade(
                price=price, qty=qty, is_buy=is_buy, event_time_ms=event_time_ms
            )
            self.stats['cvd_updates'] += 1
            return result
        except Exception as e:
            self.logger.error(f"CVD calculation failed: {e}")
            return {"z_cvd": None, "cvd": 0.0, "meta": {"warmup": True}}
    
    def update_fusion(self, z_ofi: float, z_cvd: float, ts: float, 
                     price: Optional[float] = None, lag_sec: float = 0.0) -> Dict[str, Any]:
        """更新融合指标"""
        try:
            # 传递当前regime给融合指标
            if hasattr(self, '_current_regime'):
                self.fusion._current_regime = self._current_regime
            result = self.fusion.update(z_ofi, z_cvd, ts, price, lag_sec)
            self.stats['fusion_updates'] += 1
            return result
        except Exception as e:
            self.logger.error(f"Fusion calculation failed: {e}")
            return {"fusion_score": 0.0, "signal": "neutral", "consistency": 0.0}
    
    def update_divergence(self, ts: float, price: float, z_ofi: float, z_cvd: float,
                         fusion_score: Optional[float] = None, consistency: Optional[float] = None,
                         warmup: bool = False, lag_sec: float = 0.0) -> Optional[Dict[str, Any]]:
        """更新背离检测器"""
        try:
            result = self.divergence.update(ts, price, z_ofi, z_cvd, fusion_score, 
                                           consistency, warmup, lag_sec)
            if result:
                self.stats['divergence_events'] += 1
            return result
        except Exception as e:
            self.logger.error(f"Divergence detection failed: {e}")
            return None
    
    def determine_regime(self, trade_rate: float, realized_vol: float, 
                        spread_bps: float = 5.0, volume_usd: float = 1000000.0) -> str:
        """确定市场状态 - 直接使用策略模式管理器"""
        try:
            # 创建市场活跃度数据
            activity = MarketActivity()
            activity.trades_per_min = trade_rate
            activity.quote_updates_per_sec = 100.0  # 默认值
            activity.spread_bps = spread_bps
            activity.volatility_bps = realized_vol * 10000  # 转换为bps
            activity.volume_usd = volume_usd
            
            # 使用策略模式管理器检测市场状态
            is_active = self.strategy_manager.check_market_active(activity)
            
            # 映射到我们的状态枚举
            if is_active:
                return "active"
            elif trade_rate > 10 and realized_vol > 0.005:
                return "normal"
            else:
                return "quiet"
                
        except Exception as e:
            self.logger.error(f"Strategy mode manager failed: {e}")
            # 回退到简化逻辑
            if trade_rate > 100 and realized_vol > 0.02:
                return "active"
            elif trade_rate > 10 and realized_vol > 0.005:
                return "normal"
            else:
                return "quiet"
    
    def check_guards(self, spread_bps: float, missing_msgs_rate: float, 
                    resync_detected: bool, reconnect_detected: bool) -> Tuple[bool, str]:
        """检查护栏条件"""
        current_time = datetime.now().timestamp()
        
        # 检查价差
        if spread_bps > self.config.spread_bps_cap:
            self.gate_reason_stats['spread_too_high'] += 1
            return True, f"spread_too_high_{spread_bps:.1f}bps"
        
        # 检查缺失消息率
        if missing_msgs_rate > self.config.missing_msgs_rate_cap:
            self.gate_reason_stats['missing_msgs_rate'] += 1
            return True, f"missing_msgs_rate_{missing_msgs_rate:.4f}"
        
        # 检查重同步冷却期
        if resync_detected:
            self.last_resync_time = current_time
            self.gate_reason_stats['resync_cooldown'] += 1
            return True, "resync_cooldown"
        
        if current_time - self.last_resync_time < self.config.resync_cooldown_sec:
            self.gate_reason_stats['resync_cooldown'] += 1
            return True, "resync_cooldown"
        
        # 检查重连冷却期
        if reconnect_detected:
            self.last_reconnect_time = current_time
            self.gate_reason_stats['reconnect_cooldown'] += 1
            return True, "reconnect_cooldown"
        
        if current_time - self.last_reconnect_time < self.config.reconnect_cooldown_sec:
            return True, "reconnect_cooldown"
        
        # 检查退出后冷却期
        if current_time - self.last_exit_time < self.config.cooldown_after_exit_sec:
            return True, "exit_cooldown"
        
        return False, ""
    
    def check_reverse_prevention(self, symbol: str, fusion_score: float, scenario: str = None) -> Tuple[bool, str]:
        """
        检查反向开仓防抖条件
        
        Args:
            symbol: 交易对符号
            fusion_score: 融合分数
            scenario: 场景标签 (A_H/A_L/Q_H/Q_L)
            
        Returns:
            (是否被阻止, 阻止原因)
        """
        current_time = time.time()
        direction = 1 if fusion_score > 0 else -1
        
        # 检查是否在最小持仓时间内
        if symbol in self._last_trade_time:
            time_since_last = current_time - self._last_trade_time[symbol]
            min_hold_time = self._scenario_hold_times.get(scenario, 60)  # 默认60秒
            
            if time_since_last < min_hold_time:
                self.gate_reason_stats['insufficient_hold_time'] += 1
                return True, f"insufficient_hold_time_{min_hold_time}s"
        
        # 检查连续同向信号确认
        if symbol in self._last_trade_direction:
            if self._last_trade_direction[symbol] == direction:
                # 同向信号，增加计数
                self._consecutive_same_direction[symbol] = self._consecutive_same_direction.get(symbol, 0) + 1
            else:
                # 反向信号，需要连续N=3个tick确认
                if self._consecutive_same_direction.get(symbol, 0) < 3:
                    self.gate_reason_stats['reverse_cooldown'] += 1
                    return True, "reverse_cooldown_insufficient_ticks"
                # 重置计数
                self._consecutive_same_direction[symbol] = 1
        
        # 更新状态
        self._last_trade_time[symbol] = current_time
        self._last_trade_direction[symbol] = direction
        
        return False, ""
    
    def get_gate_reason_stats(self) -> Dict[str, int]:
        """获取闸门原因统计"""
        return self.gate_reason_stats.copy()
    
    def _log_gate_stats_to_jsonl(self):
        """记录闸门统计到JSONL文件"""
        try:
            import json
            from datetime import datetime
            from pathlib import Path
            
            # 创建统计记录
            stats_record = {
                "timestamp": datetime.now().isoformat(),
                "type": "gate_stats",
                "total_signals": self._gate_stats_counter,
                "gate_reasons": self.gate_reason_stats.copy(),
                "current_regime": getattr(self, '_current_regime', 'unknown'),
                "guard_active": self.guard_active,
                "guard_reason": self.guard_reason
            }
            
            # 写入JSONL文件
            log_file = Path("artifacts/gate_stats.jsonl")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(stats_record, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"[WARN] 闸门统计记录失败: {e}")
    
    def check_gate_reason_thresholds(self) -> List[str]:
        """
        检查闸门原因阈值，返回建议调整的参数
        
        Returns:
            List[str]: 建议调整的参数列表
        """
        suggestions = []
        total_signals = self.stats['total_updates']
        
        if total_signals == 0:
            return suggestions
        
        # 检查各原因占比
        for reason, count in self.gate_reason_stats.items():
            percentage = (count / total_signals) * 100
            if percentage > 60:  # 超过60%的信号被同一原因阻止
                if reason == 'warmup_guard':
                    suggestions.append("降低min_warmup_samples参数")
                elif reason == 'lag_exceeded':
                    suggestions.append("增加max_lag参数")
                elif reason == 'consistency_low':
                    suggestions.append("降低min_consistency参数")
                elif reason == 'spread_too_high':
                    suggestions.append("增加spread_bps_cap参数")
                elif reason == 'component_warmup':
                    suggestions.append("检查OFI/CVD计算器warmup状态")
        
        return suggestions
    
    def process_signal(self, ts_ms: int, symbol: str, z_ofi: float, z_cvd: float,
                      price: float, trade_rate: float, realized_vol: float, 
                      spread_bps: float, missing_msgs_rate: float, 
                      resync_detected: bool = False, reconnect_detected: bool = False) -> SignalData:
        """处理信号 - 直接使用成熟组件"""
        
        self.stats['total_updates'] += 1
        ts = ts_ms / 1000.0  # 转换为秒
        
        # 1. 确定市场状态（只计算一次，避免重复计算）
        regime = self.determine_regime(trade_rate, realized_vol, spread_bps)
        self._current_regime = regime
        
        # 2. 使用融合指标计算融合分数（传递regime信息）
        fusion_result = self.update_fusion(z_ofi, z_cvd, ts, price)
        fusion_score = fusion_result.get('fusion_score', 0.0)
        fusion_signal = fusion_result.get('signal', 'neutral')
        consistency = fusion_result.get('consistency', 0.0)
        
        # 3. 使用背离检测器检测背离
        divergence_event = self.update_divergence(
            ts, price, z_ofi, z_cvd, fusion_score, consistency
        )
        div_type = divergence_event.get('type') if divergence_event else None
        
        # 4. 检查护栏
        guard_active, guard_reason = self.check_guards(
            spread_bps, missing_msgs_rate, resync_detected, reconnect_detected
        )
        
        # 4.1. 检查反向开仓防抖（需要场景信息）
        scenario = getattr(self, '_current_scenario', None)
        reverse_blocked, reverse_reason = self.check_reverse_prevention(symbol, fusion_score, scenario)
        if reverse_blocked:
            guard_active = True
            guard_reason = reverse_reason
        
        # 4.5. Warmup护栏检查
        warmup_active = False
        if z_ofi is None or z_cvd is None:
            warmup_active = True
            guard_reason = "warmup_guard"
            guard_active = True
            self.gate_reason_stats['warmup_guard'] += 1
        
        # 检查OFI/CVD的warmup状态
        ofi_warmup = getattr(self.ofi_calc, '_is_warmup', False)
        cvd_warmup = getattr(self.cvd_calc, '_is_warmup', False)
        if ofi_warmup or cvd_warmup:
            warmup_active = True
            guard_reason = "component_warmup"
            guard_active = True
            self.gate_reason_stats['component_warmup'] += 1
        
        # 5. 确认信号 - 质量优化版（分场景自适应阈值）
        # 从配置获取分场景阈值
        current_regime = getattr(self, '_current_regime', 'normal')
        
        # 获取融合阈值配置
        fusion_config = getattr(self.config_loader, 'fusion_metrics', {})
        thresholds = fusion_config.get('thresholds', {})
        regime_thresholds = thresholds.get('regime_thresholds', {})
        
        # 获取当前regime的阈值，如果没有则使用基础阈值
        regime_config = regime_thresholds.get(current_regime, {})
        if regime_config:
            # 使用分场景阈值
            fuse_buy = regime_config.get('fuse_buy', thresholds.get('fuse_buy', 1.2))
            fuse_sell = regime_config.get('fuse_sell', thresholds.get('fuse_sell', -1.2))
        else:
            # 使用基础阈值
            fuse_buy = thresholds.get('fuse_buy', 1.2)
            fuse_sell = thresholds.get('fuse_sell', -1.2)
        
        # 根据信号方向选择阈值
        if fusion_score > 0:
            threshold = fuse_buy
        else:
            threshold = abs(fuse_sell)
        
        # 背离信号加成：门槛再降20%（与文档对齐）
        if hasattr(self, '_last_divergence_type') and self._last_divergence_type in ("bull_div", "bear_div", "hidden_bull", "hidden_bear"):
            threshold *= 0.8  # 背离加成：门槛再降20%
        
        # 场景判定与统一取参
        is_active = getattr(self, '_is_active', True)  # 从策略管理器获取
        realized_vol = getattr(self, '_realized_vol', 0.01)  # 从市场数据获取
        self._current_scenario = self.determine_scenario(is_active, realized_vol)
        
        # 关键：让融合器读取到当前场景，让分场景一致性阈值真正生效
        if hasattr(self.fusion, '_current_regime'):
            self.fusion._current_regime = self._current_scenario
        
        # 从当前场景获取参数
        scenario_params = self._scenario_params.get(self._current_scenario, {})
        weak_signal_threshold = getattr(self, '_weak_signal_threshold', 0.15)
        consistency_threshold = getattr(self, '_consistency_min', scenario_params.get('consistency_min', 0.5))
        
        # 检查弱信号节流
        if abs(fusion_score) < weak_signal_threshold:
            # 弱信号直接丢弃
            self._gate_stats['weak'] += 1
            self.gate_reason_stats['weak_signal_throttle'] = self.gate_reason_stats.get('weak_signal_throttle', 0) + 1
            guard_active = True
            guard_reason = "weak_signal_throttle"
        elif consistency < consistency_threshold:
            # 一致性不足
            self._gate_stats['low_consistency'] += 1
            self.gate_reason_stats['low_consistency'] = self.gate_reason_stats.get('low_consistency', 0) + 1
            guard_active = True
            guard_reason = "low_consistency"
        
        confirm = (abs(fusion_score) >= threshold and
                  not guard_active and not warmup_active)
        
        # 更新护栏状态
        self.guard_active = guard_active
        self.guard_reason = guard_reason
        
        # 记录闸门原因统计到JSONL（每100次记录一次）
        if hasattr(self, '_gate_stats_counter'):
            self._gate_stats_counter += 1
        else:
            self._gate_stats_counter = 1
            
        if self._gate_stats_counter % 100 == 0:
            self._log_gate_stats_to_jsonl()
        
        # 更新统计
        if confirm:
            self.stats['valid_signals'] += 1
        if guard_active:
            self.stats['suppressed_signals'] += 1
        
        # 创建信号数据
        signal_data = SignalData(
            ts_ms=ts_ms,
            symbol=symbol,
            score=fusion_score,
            z_ofi=z_ofi,
            z_cvd=z_cvd,
            regime=regime,
            div_type=div_type,
            confirm=confirm,
            gating=guard_active
        )
        
        return signal_data
    
    def get_component_stats(self) -> Dict[str, Any]:
        """获取各组件统计信息"""
        return {
            'core_stats': self.stats,
            'ofi_stats': self.ofi_calc.get_state() if hasattr(self, 'ofi_calc') else {},
            'cvd_stats': self.cvd_calc.get_state() if hasattr(self, 'cvd_calc') else {},
            'fusion_stats': self.fusion.get_stats() if hasattr(self, 'fusion') else {},
            'divergence_stats': self.divergence.get_stats() if hasattr(self, 'divergence') else {},
            'strategy_manager_stats': self.strategy_manager.get_mode_stats() if hasattr(self, 'strategy_manager') else {}
        }
    
    def reset_components(self):
        """重置所有组件"""
        if hasattr(self, 'ofi_calc'):
            self.ofi_calc.reset()
        if hasattr(self, 'cvd_calc'):
            self.cvd_calc.reset()
        if hasattr(self, 'fusion'):
            self.fusion.reset()
        if hasattr(self, 'divergence'):
            self.divergence.reset()
        if hasattr(self, 'strategy_manager'):
            # 策略模式管理器没有reset方法，但可以重新初始化
            pass
        
        # 重置统计
        self.stats = {
            'total_updates': 0,
            'valid_signals': 0,
            'suppressed_signals': 0,
            'regime_changes': 0,
            'ofi_updates': 0,
            'cvd_updates': 0,
            'fusion_updates': 0,
            'divergence_events': 0
        }
    
    def log_signal(self, signal_data: SignalData, output_dir: str = None):
        """记录信号日志（健壮 I/O 版）"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(signal_data.ts_ms / 1000).isoformat(),
            "ts_ms": signal_data.ts_ms,
            "symbol": signal_data.symbol,
            "score": signal_data.score,
            "z_ofi": signal_data.z_ofi,
            "z_cvd": signal_data.z_cvd,
            "regime": signal_data.regime,
            "div_type": signal_data.div_type,
            "confirm": signal_data.confirm,
            "gating": signal_data.gating,
            "guard_reason": self.guard_reason if getattr(self, "guard_active", False) else None,
        }

        # 允许函数参数临时覆盖输出目录
        base_dir = Path(output_dir) if output_dir else Path(os.getenv("V13_OUTPUT_DIR", "./runtime"))
        rel_path = Path("signal") / signal_data.symbol / f"signals_{datetime.now().strftime('%Y%m%d')}.jsonl"

        try:
            # 优先使用实例级 writer；若被覆盖目录则临时写入
            writer = self._safe_writer if base_dir == self._safe_writer.base_dir else SafeJsonlWriter(base_dir)
            writer.write_line(rel_path, log_entry)
            self.logger.info(
                f"Signal logged: {signal_data.symbol} score={signal_data.score:.3f} "
                f"regime={signal_data.regime} confirm={signal_data.confirm} gating={signal_data.gating}"
            )
        except Exception as e:
            # 不让交易/信号路径被 I/O 拖垮
            self.logger.error(f"[FILE-IO] failed to write {rel_path} under {base_dir}: {e}; entry={log_entry}")

def main():
    """测试主函数 - 直接调用成熟组件"""
    print("=== 核心算法v1 - 直接调用成熟组件测试 ===")
    
    # 初始化核心算法
    symbol = "BTCUSDT"
    config = SignalConfig()
    algo = CoreAlgorithm(symbol, config)
    
    # 模拟数据
    ts_ms = int(datetime.now().timestamp() * 1000)
    z_ofi = 1.5
    z_cvd = 1.2
    price = 50000.0
    trade_rate = 50.0
    realized_vol = 0.01
    spread_bps = 5.0
    missing_msgs_rate = 0.0001
    
    print(f"\n--- 测试OFI计算器 ---")
    # 模拟订单簿数据
    bids = [(50000.0, 10.0), (49999.0, 8.0), (49998.0, 6.0), (49997.0, 4.0), (49996.0, 2.0)]
    asks = [(50001.0, 12.0), (50002.0, 9.0), (50003.0, 7.0), (50004.0, 5.0), (50005.0, 3.0)]
    ofi_result = algo.update_ofi(bids, asks, ts_ms)
    print(f"OFI结果: {ofi_result}")
    
    print(f"\n--- 测试CVD计算器 ---")
    cvd_result = algo.update_cvd(price=price, qty=1.0, is_buy=True, event_time_ms=ts_ms)
    print(f"CVD结果: {cvd_result}")
    
    print(f"\n--- 测试融合指标 ---")
    fusion_result = algo.update_fusion(z_ofi, z_cvd, ts_ms/1000.0, price)
    print(f"融合结果: {fusion_result}")
    
    print(f"\n--- 测试背离检测 ---")
    divergence_result = algo.update_divergence(ts_ms/1000.0, price, z_ofi, z_cvd)
    print(f"背离检测结果: {divergence_result}")
    
    print(f"\n--- 测试完整信号处理 ---")
    # 处理信号
    signal_data = algo.process_signal(
        ts_ms, symbol, z_ofi, z_cvd, price,
        trade_rate, realized_vol, spread_bps, missing_msgs_rate
    )
    
    print(f"信号数据: {signal_data}")
    print(f"护栏状态: {algo.guard_active}, 原因: {algo.guard_reason}")
    
    print(f"\n--- 组件统计信息 ---")
    stats = algo.get_component_stats()
    for component, stat in stats.items():
        print(f"{component}: {stat}")
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    main()
