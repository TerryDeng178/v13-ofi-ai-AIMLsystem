#!/usr/bin/env python3
"""
Task_1.2.13 修正评估 - 使用正确的WINSOR_LIMIT=8.0
按照任务卡要求进行60分钟最终验证，输出全量/Active/Quiet三套指标与Wilson CI
"""

import asyncio
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import deque
import logging

# 确保项目根目录在sys.path中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.run_realtime_cvd import ws_consume, parse_aggtrade_message, MonitoringMetrics
from src.real_cvd_calculator import RealCVDCalculator

# Fix Pack v2: 环境变量解析函数
def env_as_float(name, default):
    try: 
        return float(os.getenv(name, str(default)))
    except: 
        return default

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class CorrectedEvaluation:
    def __init__(self):
        self.data = deque()
        self.start_time = None
        self.end_time = None
        # Fix Pack v2: 维护最近60s的到达时间戳，实时计算滑窗速率
        self.ts_window = deque(maxlen=100000)
        self.WINDOW = 60.0
        # Fix Pack v2: aggTrade去重
        self.seen_ids = deque(maxlen=8192)
        
    def update_rate(self, ts_now):
        """Fix Pack v2: 60s滑窗速率计算"""
        self.ts_window.append(ts_now)
        # 剔除窗口外
        while self.ts_window and (ts_now - self.ts_window[0]) > self.WINDOW:
            self.ts_window.popleft()
        return len(self.ts_window) / self.WINDOW
        
    async def run_corrected_evaluation(self, symbol: str, duration: int = 3600):
        """运行修正的60分钟评估"""
        log.info(f"开始修正评估 - 使用WINSOR_LIMIT=8.0")
        log.info(f"交易对: {symbol}, 持续时间: {duration}秒")
        
        # Fix Pack v2: 强制Rank1参数生效 & 打印配置指纹
        EXPECTED = {
            "MAD_MULTIPLIER": 1.47,
            "SCALE_FAST_WEIGHT": 0.35,
            "Z_HI": 3.00,
            "Z_MID": 2.00,
        }
        # 环境变量优先，便于命令行/CI注入
        mad = env_as_float("MAD_MULTIPLIER", EXPECTED["MAD_MULTIPLIER"])
        fast = env_as_float("SCALE_FAST_WEIGHT", EXPECTED["SCALE_FAST_WEIGHT"])
        z_hi = env_as_float("Z_HI", EXPECTED["Z_HI"])
        z_mid = env_as_float("Z_MID", EXPECTED["Z_MID"])
        
        cfg_fingerprint = f"MAD={mad:.3f}|FAST={fast:.3f}|ZHI={z_hi:.2f}|ZMID={z_mid:.2f}"
        log.info(f"[CONFIG] {cfg_fingerprint}")
        
        # 硬性校验：若未按期望生效，直接失败（避免"默认参数滑落"）
        assert abs(mad-EXPECTED["MAD_MULTIPLIER"])<1e-6 and abs(fast-EXPECTED["SCALE_FAST_WEIGHT"])<1e-6, \
            f"Config not applied! got {cfg_fingerprint}"
        
        # 使用修正的参数配置
        config = {
            'CVD_Z_MODE': 'delta',
            'HALF_LIFE_TRADES': 100,  # 保持当前最优
            'WINSOR_LIMIT': 8.0,      # 修正：使用任务卡要求的8.0
            'STALE_THRESHOLD_MS': 5000,
            'FREEZE_MIN': 80,
            'SCALE_MODE': 'hybrid',
            'EWMA_FAST_HL': 80,
            'SCALE_FAST_WEIGHT': fast,  # Fix Pack v2: 使用强制参数
            'SCALE_SLOW_WEIGHT': 1.0 - fast,
            'MAD_WINDOW_TRADES': 300,
            'MAD_SCALE_FACTOR': 1.4826,
            'MAD_MULTIPLIER': mad,       # Fix Pack v2: 使用强制参数
            'SOFT_FREEZE_V2': 0,        # 先测试基础版本
        }
        
        # 设置环境变量
        for key, value in config.items():
            os.environ[key] = str(value)
        
        # Fix Pack v2: 初始化CVD计算器，传递强制参数
        cvd_calculator = RealCVDCalculator(
            symbol=symbol,
            mad_multiplier=mad,
            scale_fast_weight=fast,
            z_hi=z_hi,
            z_mid=z_mid
        )
        
        # WebSocket配置
        url = f"wss://fstream.binancefuture.com/stream?streams={symbol.lower()}@aggTrade"
        q = asyncio.Queue(maxsize=50000)
        stop_evt = asyncio.Event()
        metrics = MonitoringMetrics()
        
        # Fix Pack v2: 数据收集 - 去重/方向自检/硬停窗口
        dir_mismatch = 0              # 方向自检计数
        t0 = time.time()
        EVAL_SECS = duration  # Fix Pack v2: 硬停窗口
        
        async def collect_data():
            while (time.time() - t0) <= EVAL_SECS and not stop_evt.is_set():
                try:
                    ts_recv, raw_msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    parsed = parse_aggtrade_message(raw_msg)
                    if parsed:
                        price, qty, is_buy, event_ms, agg_trade_id = parsed

                        # Fix Pack v2: aggTrade去重 - 重连/抖动时避免重复样本放大尾部
                        if agg_trade_id in self.seen_ids:
                            continue
                        self.seen_ids.append(agg_trade_id)

                        # 方向一致性自检（m=True => 卖出主动方）
                        try:
                            raw_m = raw_msg.get('data', {}).get('m', None)
                            if isinstance(raw_m, bool):
                                if (not raw_m) != is_buy:
                                    nonlocal dir_mismatch
                                    dir_mismatch += 1
                        except Exception:
                            pass
                        
                        # 计算CVD Z-score
                        cvd_result = cvd_calculator.update_with_trade(
                            price=price, qty=qty, is_buy=is_buy, event_time_ms=event_ms
                        )
                        
                        # Fix Pack v2: 真启用z_raw（未截断） - 禁止回退
                        try:
                            # 尝试获取真实的未截断Z值
                            snap = cvd_calculator.get_last_zscores()
                            z_cvd = snap["z_cvd"]
                            z_raw = snap["z_raw"]
                            is_warmup = snap["is_warmup"]
                            is_flat = snap["is_flat"]
                        except AttributeError:
                            # 如果计算器没有get_last_zscores方法，回退到私有方法但记录警告
                            z_cvd, is_warmup, is_flat = cvd_calculator._z_last_excl()
                            z_raw = None
                            log.warning("计算器未实现get_last_zscores，z_raw将回退到None")
                        
                        if z_cvd is not None and not is_warmup:
                            # Fix Pack v2: 60s滑窗速率计算
                            recv_rate = self.update_rate(ts_recv)
                            
                            self.data.append({
                                'timestamp': ts_recv,
                                'event_time_ms': event_ms,
                                'price': price,
                                'qty': qty,
                                'is_buy': is_buy,
                                'z_cvd': z_cvd,
                                'z_raw': z_raw,  # Fix Pack v2: 真启用z_raw
                                'recv_rate': recv_rate,
                                'latency_ms': (ts_recv * 1000 - event_ms) if event_ms else 0.0,
                                'is_warmup': is_warmup,
                                'is_flat': is_flat
                            })
                            
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    log.error(f"数据处理错误: {e}")
        
        # 启动WebSocket消费
        consumer_task = asyncio.create_task(
            ws_consume(url, q, stop_evt, metrics, 60, 30, 20, 10)
        )
        collector_task = asyncio.create_task(collect_data())
        
        self.start_time = time.time()
        log.info(f"开始数据收集...")
        
        try:
            # Fix Pack v2: 硬停控制 - 严格运行 duration 秒
            await asyncio.wait_for(stop_evt.wait(), timeout=duration)
        except asyncio.TimeoutError:
            log.info(f"评估时间到达 ({duration}秒)")
            stop_evt.set()
        finally:
            await asyncio.sleep(1)
            consumer_task.cancel()
            collector_task.cancel()
            await asyncio.gather(consumer_task, collector_task, return_exceptions=True)
        
        self.end_time = time.time()
        log.info(f"数据收集完成，共收集 {len(self.data)} 条记录")
        
        # Fix Pack v2: 诊断断言
        if dir_mismatch > 0:
            log.error(f"方向自检不一致计数: {dir_mismatch}")
            # 按严格口径，此处可以直接失败
            raise RuntimeError(f"Direction mismatch={dir_mismatch}")
        
        # Fix Pack v2: 样本量检查（调整阈值适应20分钟）
        min_samples = 1000 if duration <= 1200 else 3000  # 20分钟≥1000，60分钟≥3000
        if len(self.data) < min_samples:
            log.error(f"样本量不足: {len(self.data)} < {min_samples}")
            raise RuntimeError(f"Too few samples in {duration}s: {len(self.data)}")
        
        return self.analyze_results()
    
    def analyze_results(self):
        """分析结果，输出全量/Active/Quiet三套指标与Wilson CI"""
        if not self.data:
            return None
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame(list(self.data))
        df['z_cvd'] = pd.to_numeric(df['z_cvd'], errors='coerce')
        df['z_raw'] = pd.to_numeric(df['z_raw'], errors='coerce')
        df['recv_rate'] = pd.to_numeric(df['recv_rate'], errors='coerce')
        
        # Fix Pack v2: z_raw真启用 - 禁止回退
        if df['z_raw'].isna().all():
            raise RuntimeError("z_raw not captured! Please check calculator instrumentation.")
        
        # Fix Pack v2: 过滤有效数据 + warmup剔除
        valid_data = df[(~df['is_warmup']) & df['z_cvd'].notna() & df['z_raw'].notna()].copy()
        
        if len(valid_data) == 0:
            log.error("没有有效的Z-score数据")
            return None
        
        # Fix Pack v2: Regime切箱修复（互斥 & 不漏）
        THRESH = float(os.getenv("ACTIVE_TPS", "1.0"))
        active_data = valid_data[valid_data['recv_rate'] >= THRESH].copy()
        quiet_data = valid_data[valid_data['recv_rate'] < THRESH].copy()
        
        # Fix Pack v2: 互斥性/覆盖率自检
        idx_a = set(active_data.index.tolist())
        idx_q = set(quiet_data.index.tolist())
        assert len(idx_a & idx_q) == 0, "Regime split is not mutually exclusive!"
        assert len(idx_a | idx_q) == len(valid_data), "Regime split lost rows!"
        
        log.info(f"[REGIME] THRESH={THRESH:.2f} | active={len(active_data)} quiet={len(quiet_data)} total={len(valid_data)}")
        
        # Fix Pack v2: 延迟画像
        lat = pd.to_numeric(valid_data['latency_ms'], errors='coerce').dropna()
        if len(lat) > 0:
            lat_p50, lat_p90, lat_p99 = np.percentile(lat, [50, 90, 99]).tolist()
            log.info(f"[LAT] p50={lat_p50:.1f}ms p90={lat_p90:.1f}ms p99={lat_p99:.1f}ms (n={len(lat)})")
        else:
            lat_p50, lat_p90, lat_p99 = np.nan, np.nan, np.nan
        
        results = {}
        
        # 全量分析
        results['overall'] = self._calculate_metrics(valid_data, 'z_cvd', 'z_raw')
        results['overall'].update(self._latency_percentiles(valid_data))
        
        # Fix Pack v2: Active Regime分析（使用修复后的切箱）
        if len(active_data) > 0:
            results['active'] = self._calculate_metrics(active_data, 'z_cvd', 'z_raw')
            results['active'].update(self._latency_percentiles(active_data))
        else:
            results['active'] = None
        
        # Fix Pack v2: Quiet Regime分析（使用修复后的切箱）
        if len(quiet_data) > 0:
            results['quiet'] = self._calculate_metrics(quiet_data, 'z_cvd', 'z_raw')
            results['quiet'].update(self._latency_percentiles(quiet_data))
        else:
            results['quiet'] = None
        
        # 软冻结相关指标
        results['soft_freeze'] = self._calculate_soft_freeze_metrics(valid_data)
        
        return results
    
    def _calculate_metrics(self, data, z_col, z_raw_col):
        """计算指标和Wilson CI"""
        z_values = data[z_col].values
        z_raw_values = data[z_raw_col].values
        
        # 基础统计
        n = len(z_values)
        p_z_gt_2 = np.mean(np.abs(z_values) > 2)
        p_z_gt_3 = np.mean(np.abs(z_values) > 3)
        median_z = np.median(z_values)
        p95_z = np.percentile(z_values, 95)
        
        # 原始Z-score统计（未截断）
        p_z_raw_gt_2 = np.mean(np.abs(z_raw_values) > 2)
        p_z_raw_gt_3 = np.mean(np.abs(z_raw_values) > 3)
        median_z_raw = np.median(z_raw_values)
        p95_z_raw = np.percentile(z_raw_values, 95)
        
        # Fix Pack v2: Wilson置信区间 - 避免scipy依赖
        def wilson_ci(p, n, confidence=0.95):
            if n == 0:
                return 0, 0
            z = 1.96  # 95% Z分位，避免scipy依赖
            p_hat = p
            n_hat = n
            ci_lower = (p_hat + z**2/(2*n_hat) - z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n_hat))/n_hat)) / (1 + z**2/n_hat)
            ci_upper = (p_hat + z**2/(2*n_hat) + z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n_hat))/n_hat)) / (1 + z**2/n_hat)
            return ci_lower, ci_upper
        
        p_z_gt_2_ci = wilson_ci(p_z_gt_2, n)
        p_z_gt_3_ci = wilson_ci(p_z_gt_3, n)
        
        return {
            'n': n,
            'p_z_gt_2': p_z_gt_2,
            'p_z_gt_2_ci': p_z_gt_2_ci,
            'p_z_gt_3': p_z_gt_3,
            'p_z_gt_3_ci': p_z_gt_3_ci,
            'median_z': median_z,
            'p95_z': p95_z,
            'p_z_raw_gt_2': p_z_raw_gt_2,
            'p_z_raw_gt_3': p_z_raw_gt_3,
            'median_z_raw': median_z_raw,
            'p95_z_raw': p95_z_raw,
        }
    
    def _calculate_soft_freeze_metrics(self, data):
        """计算软冻结相关指标"""
        # 这里需要根据实际需求实现z_after_silence_p3等指标
        # 暂时返回基础指标
        return {
            'z_after_silence_p3': 0.0,  # 待实现
            'silence_periods': 0,        # 待实现
        }

    def _latency_percentiles(self, data):
        import numpy as np
        lat = data['latency_ms'].dropna().astype(float).values
        if lat.size == 0:
            return {'latency_p50': None, 'latency_p90': None, 'latency_p99': None}
        p50, p90, p99 = np.percentile(lat, [50, 90, 99]).tolist()
        return {'latency_p50': p50, 'latency_p90': p90, 'latency_p99': p99}

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default=os.getenv("SYMBOL", "BTCUSDT"))
    parser.add_argument('--duration', type=int, default=int(os.getenv("DURATION", "3600")))
    parser.add_argument('--mad', type=float, default=None)
    parser.add_argument('--fast', type=float, default=None)
    parser.add_argument('--hl', type=int, default=None)
    parser.add_argument('--winsor', type=float, default=None)
    parser.add_argument('--label', default=None)
    args = parser.parse_args()

    symbol = args.symbol
    duration = args.duration
    
    evaluator = CorrectedEvaluation()
    results = await evaluator.run_corrected_evaluation(symbol, duration)
    
    if results:
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/cvd_corrected_evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        label_suffix = f"_{args.label}" if args.label else ""
        results_file = output_dir / f"corrected_evaluation_{symbol.lower()}_{timestamp}{label_suffix}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        log.info(f"修正评估结果已保存到: {results_file}")
        
        # 打印摘要
        log.info("\n" + "="*60)
        log.info("修正评估结果摘要")
        log.info("="*60)
        
        for regime, metrics in results.items():
            if metrics and 'n' in metrics:
                log.info(f"\n{regime.upper()} Regime:")
                log.info(f"  样本量: {metrics['n']}")
                log.info(f"  P(|Z|>2): {metrics['p_z_gt_2']:.4f} (CI: {metrics['p_z_gt_2_ci']})")
                log.info(f"  P(|Z|>3): {metrics['p_z_gt_3']:.4f} (CI: {metrics['p_z_gt_3_ci']})")
                log.info(f"  Median(Z): {metrics['median_z']:.4f}")
                log.info(f"  P95(Z): {metrics['p95_z']:.4f}")
                log.info(f"  P(|Z_raw|>3): {metrics['p_z_raw_gt_3']:.4f} (未截断)")
                log.info(f"  Latency p50/p90/p99(ms): {metrics.get('latency_p50')}, {metrics.get('latency_p90')}, {metrics.get('latency_p99')}")

if __name__ == "__main__":
    asyncio.run(main())
