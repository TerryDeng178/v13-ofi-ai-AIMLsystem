#!/usr/bin/env python3
"""
改进版CVD评测脚本 - 实现动态阈值切箱和尺度诊断
基于用户建议的三个关键改进：
1. 口径对齐（delta+hybrid+MAD地板）
2. 动态阈值切箱（保证每箱n≥1000）
3. 尺度诊断+地板命中率
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import deque
import argparse
import numpy as np
import pandas as pd

# 确保项目根目录在sys.path中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.run_realtime_cvd import ws_consume, parse_aggtrade_message, MonitoringMetrics
from src.real_cvd_calculator import RealCVDCalculator, CVDConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class ImprovedCVDCollector:
    def __init__(self, symbol: str, duration: int, mad_multiplier: float, 
                 scale_fast_weight: float, half_life_trades: int, winsor_limit: float):
        self.symbol = symbol
        self.duration = duration
        self.mad_multiplier = mad_multiplier
        self.scale_fast_weight = scale_fast_weight
        self.half_life_trades = half_life_trades
        self.winsor_limit = winsor_limit
        
        # 数据收集
        self.data = []
        self.seen_ids = deque(maxlen=4096)  # 去重
        self.start_time = None
        
        # 配置指纹校验
        self.cfg_fingerprint = {
            'z_mode': 'delta',
            'scale_mode': 'hybrid', 
            'mad_multiplier': mad_multiplier,
            'scale_fast_weight': scale_fast_weight,
            'half_life_trades': half_life_trades,
            'winsor_limit': winsor_limit
        }
        
        # 尺度诊断统计
        self.floor_hits = 0
        self.total_samples = 0
        
    def create_calculator(self):
        """创建CVD计算器，强制使用指定参数"""
        config = CVDConfig(
            z_mode="delta",
            scale_mode="hybrid",
            winsor_limit=self.winsor_limit,
            freeze_min=50,
            mad_multiplier=self.mad_multiplier,
            scale_fast_weight=self.scale_fast_weight,
            half_life_trades=self.half_life_trades,
            ewma_fast_hl=80,
            mad_window_trades=300,
            mad_scale_factor=1.4826,
            scale_slow_weight=0.7
        )
        
        calc = RealCVDCalculator(
            self.symbol, 
            cfg=config,
            mad_multiplier=self.mad_multiplier,
            scale_fast_weight=self.scale_fast_weight
        )
        
        # 配置指纹校验
        actual_cfg = {
            'z_mode': calc.cfg.z_mode,
            'scale_mode': calc.cfg.scale_mode,
            'mad_multiplier': calc.cfg.mad_multiplier,
            'scale_fast_weight': calc.cfg.scale_fast_weight,
            'half_life_trades': calc.cfg.half_life_trades,
            'winsor_limit': calc.cfg.winsor_limit
        }
        
        if actual_cfg != self.cfg_fingerprint:
            raise RuntimeError(f"配置指纹不匹配! 期望: {self.cfg_fingerprint}, 实际: {actual_cfg}")
        
        log.info(f"配置指纹校验通过: {actual_cfg}")
        return calc
    
    async def collect_data(self, symbol: str, duration: int):
        """收集数据"""
        url = f"wss://fstream.binancefuture.com/stream?streams={symbol.lower()}@aggTrade"
        q = asyncio.Queue(maxsize=50000)
        stop_evt = asyncio.Event()
        metrics = MonitoringMetrics()
        
        cvd_calculator = self.create_calculator()
        self.start_time = time.time()
        
        # 60秒滑窗用于recv_rate计算
        self.rate_window = deque(maxlen=60)
        
        async def consume_and_process():
            while not stop_evt.is_set():
                try:
                    ts_recv, raw_msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    parsed = parse_aggtrade_message(raw_msg)
                    if parsed:
                        price, qty, is_buy, event_ms, agg_trade_id = parsed
                        
                        # 去重检查
                        if agg_trade_id in self.seen_ids:
                            continue
                        self.seen_ids.append(agg_trade_id)
                        
                        # 更新CVD计算器
                        result = cvd_calculator.update_with_trade(
                            price=price,
                            qty=qty,
                            is_buy=is_buy,
                            event_time_ms=event_ms
                        )
                        
                        # 获取Z分数和尺度诊断
                        z_info = cvd_calculator.get_last_zscores()
                        z_stats = cvd_calculator.get_z_stats()
                        
                        # 计算recv_rate (60秒滑窗)
                        self.rate_window.append(ts_recv)
                        if len(self.rate_window) >= 2:
                            recv_rate = len(self.rate_window) / (self.rate_window[-1] - self.rate_window[0])
                        else:
                            recv_rate = 0.0
                        
                        # 计算延迟
                        latency_ms = (ts_recv * 1000 - event_ms) if event_ms else 0.0
                        
                        # 尺度诊断统计
                        if 'scale' in z_stats and 'sigma_floor' in z_stats:
                            if z_stats['scale'] == z_stats['sigma_floor']:
                                self.floor_hits += 1
                            self.total_samples += 1
                        
                        # 存储数据
                        self.data.append({
                            'timestamp': ts_recv,
                            'event_time_ms': event_ms,
                            'price': price,
                            'qty': qty,
                            'is_buy': is_buy,
                            'agg_trade_id': agg_trade_id,
                            'cvd': result['cvd'],
                            'z_cvd': z_info['z_cvd'],
                            'z_raw': z_info['z_raw'],
                            'is_warmup': z_info['is_warmup'],
                            'is_flat': z_info['is_flat'],
                            'recv_rate': recv_rate,
                            'latency_ms': latency_ms,
                            'z_stats': z_stats
                        })
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    log.error(f"处理消息错误: {e}")
        
        # 启动任务
        consumer_task = asyncio.create_task(
            ws_consume(url, q, stop_evt, metrics, 60, 30, 20, 10)
        )
        processor_task = asyncio.create_task(consume_and_process())
        
        # 等待指定时间
        try:
            await asyncio.wait_for(stop_evt.wait(), timeout=duration)
        except asyncio.TimeoutError:
            log.info(f"测试时间到达 ({duration}秒). 停止...")
            stop_evt.set()
        finally:
            await asyncio.sleep(1)
            consumer_task.cancel()
            processor_task.cancel()
            await asyncio.gather(consumer_task, processor_task, return_exceptions=True)
        
        return metrics
    
    def analyze_data(self):
        """分析数据，实现动态阈值切箱"""
        if not self.data:
            raise RuntimeError("没有收集到数据")
        
        df = pd.DataFrame(self.data)
        
        # 过滤warmup和flat数据
        valid_data = df[(~df['is_warmup']) & (~df['is_flat'])].copy()
        
        if len(valid_data) < 1000:
            raise RuntimeError(f"有效样本量不足: {len(valid_data)} < 1000")
        
        # 动态阈值切箱 (使用60分位数)
        thr = valid_data['recv_rate'].quantile(0.60)
        active_data = valid_data[valid_data['recv_rate'] >= thr].copy()
        quiet_data = valid_data[valid_data['recv_rate'] < thr].copy()
        
        # 互斥和覆盖断言
        assert len(set(active_data.index) & set(quiet_data.index)) == 0, "Active和Quiet有重叠"
        assert len(active_data) + len(quiet_data) == len(valid_data), "样本总数不匹配"
        
        log.info(f"动态阈值: {thr:.4f} tps")
        log.info(f"Active: {len(active_data)} 样本 ({len(active_data)/len(valid_data)*100:.1f}%)")
        log.info(f"Quiet: {len(quiet_data)} 样本 ({len(quiet_data)/len(valid_data)*100:.1f}%)")
        
        # 计算各档统计
        def calc_regime_stats(data, name):
            if len(data) == 0:
                return {
                    'name': name,
                    'n': 0,
                    'p_z_gt_2': 0.0,
                    'p_z_gt_3': 0.0,
                    'median_z': 0.0,
                    'p95_z': 0.0,
                    'floor_hit_rate': 0.0,
                    'scale_median': 0.0,
                    'sigma_floor_median': 0.0
                }
            
            z_abs = np.abs(data['z_cvd'])
            p_z_gt_2 = np.mean(z_abs > 2.0)
            p_z_gt_3 = np.mean(z_abs > 3.0)
            median_z = np.median(z_abs)
            p95_z = np.percentile(z_abs, 95)
            
            # 尺度诊断
            scales = [stats.get('scale', 0) for stats in data['z_stats'] if isinstance(stats, dict)]
            sigma_floors = [stats.get('sigma_floor', 0) for stats in data['z_stats'] if isinstance(stats, dict)]
            
            floor_hit_rate = 0.0
            scale_median = 0.0
            sigma_floor_median = 0.0
            
            if scales and sigma_floors:
                floor_hits = sum(1 for s, f in zip(scales, sigma_floors) if s == f)
                floor_hit_rate = floor_hits / len(scales) if scales else 0.0
                scale_median = np.median(scales)
                sigma_floor_median = np.median(sigma_floors)
            
            return {
                'name': name,
                'n': len(data),
                'p_z_gt_2': p_z_gt_2,
                'p_z_gt_3': p_z_gt_3,
                'median_z': median_z,
                'p95_z': p95_z,
                'floor_hit_rate': floor_hit_rate,
                'scale_median': scale_median,
                'sigma_floor_median': sigma_floor_median
            }
        
        # 计算各档统计
        overall_stats = calc_regime_stats(valid_data, 'OVERALL')
        active_stats = calc_regime_stats(active_data, 'ACTIVE')
        quiet_stats = calc_regime_stats(quiet_data, 'QUIET')
        
        # 延迟统计
        latency_p50 = np.percentile(valid_data['latency_ms'], 50)
        latency_p90 = np.percentile(valid_data['latency_ms'], 90)
        latency_p99 = np.percentile(valid_data['latency_ms'], 99)
        
        return {
            'overall': overall_stats,
            'active': active_stats,
            'quiet': quiet_stats,
            'threshold_recv_rate': thr,
            'latency_p50': latency_p50,
            'latency_p90': latency_p90,
            'latency_p99': latency_p99,
            'total_samples': len(valid_data),
            'floor_hit_rate_global': self.floor_hits / self.total_samples if self.total_samples > 0 else 0.0
        }
    
    def print_summary(self, results):
        """打印结果摘要"""
        log.info("\n" + "="*60)
        log.info("改进版CVD评测结果摘要")
        log.info("="*60)
        
        for regime in ['overall', 'active', 'quiet']:
            stats = results[regime]
            log.info(f"\n{stats['name']} Regime:")
            log.info(f"  样本量: {stats['n']}")
            log.info(f"  P(|Z|>2): {stats['p_z_gt_2']:.4f}")
            log.info(f"  P(|Z|>3): {stats['p_z_gt_3']:.4f}")
            log.info(f"  Median(|Z|): {stats['median_z']:.4f}")
            log.info(f"  P95(|Z|): {stats['p95_z']:.4f}")
            log.info(f"  Floor命中率: {stats['floor_hit_rate']:.4f}")
            log.info(f"  Scale中位数: {stats['scale_median']:.6f}")
            log.info(f"  Sigma地板中位数: {stats['sigma_floor_median']:.6f}")
        
        log.info(f"\n延迟统计:")
        log.info(f"  P50/P90/P99(ms): {results['latency_p50']:.1f}, {results['latency_p90']:.1f}, {results['latency_p99']:.1f}")
        log.info(f"动态阈值: {results['threshold_recv_rate']:.4f} tps")
        log.info(f"全局Floor命中率: {results['floor_hit_rate_global']:.4f}")

async def run_improved_evaluation(symbol: str, duration: int, mad_multiplier: float,
                                scale_fast_weight: float, half_life_trades: int, 
                                winsor_limit: float, label: str):
    """运行改进版评测"""
    log.info(f"开始改进版CVD评测 - 使用WINSOR_LIMIT={winsor_limit}")
    log.info(f"测试对: {symbol}, 测试时长: {duration}秒")
    log.info(f"参数: MAD={mad_multiplier}, FAST={scale_fast_weight}, HL={half_life_trades}")
    log.info("开始数据收集...")
    
    collector = ImprovedCVDCollector(
        symbol=symbol,
        duration=duration,
        mad_multiplier=mad_multiplier,
        scale_fast_weight=scale_fast_weight,
        half_life_trades=half_life_trades,
        winsor_limit=winsor_limit
    )
    
    # 收集数据
    metrics = await collector.collect_data(symbol, duration)
    
    log.info(f"数据收集完成，收集 {len(collector.data)} 条记录")
    
    # 分析数据
    results = collector.analyze_data()
    
    # 打印摘要
    collector.print_summary(results)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/improved_cvd_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"improved_evaluation_{symbol.lower()}_{timestamp}_{label}.json"
    
    # 准备保存数据
    save_data = {
        'test_info': {
            'symbol': symbol,
            'duration': duration,
            'timestamp': timestamp,
            'label': label,
            'parameters': {
                'mad_multiplier': mad_multiplier,
                'scale_fast_weight': scale_fast_weight,
                'half_life_trades': half_life_trades,
                'winsor_limit': winsor_limit
            }
        },
        'results': results,
        'metrics': metrics.to_dict()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    log.info(f"详细结果已保存到: {output_file}")
    
    return results

async def main():
    parser = argparse.ArgumentParser(description='改进版CVD评测')
    parser.add_argument('--symbol', default='BTCUSDT', help='交易对')
    parser.add_argument('--duration', type=int, default=1200, help='测试时长(秒)')
    parser.add_argument('--mad', type=float, default=1.8, help='MAD乘数')
    parser.add_argument('--fast', type=float, default=0.20, help='快速权重')
    parser.add_argument('--hl', type=int, default=600, help='半衰期')
    parser.add_argument('--winsor', type=float, default=8.0, help='Winsor限制')
    parser.add_argument('--label', default='improved_v1', help='测试标签')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['Z_MODE'] = 'delta'
    os.environ['SCALE_MODE'] = 'hybrid'
    os.environ['MAD_MULTIPLIER'] = str(args.mad)
    os.environ['SCALE_FAST_WEIGHT'] = str(args.fast)
    os.environ['HALF_LIFE_TRADES'] = str(args.hl)
    os.environ['WINSOR_LIMIT'] = str(args.winsor)
    
    results = await run_improved_evaluation(
        symbol=args.symbol,
        duration=args.duration,
        mad_multiplier=args.mad,
        scale_fast_weight=args.fast,
        half_life_trades=args.hl,
        winsor_limit=args.winsor,
        label=args.label
    )
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
