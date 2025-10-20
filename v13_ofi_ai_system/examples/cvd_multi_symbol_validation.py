#!/usr/bin/env python3
"""
CVD多交易对验证脚本 - 基于Phase A1优化参数
验证A1参数在不同交易对和时段的稳定性
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
from concurrent.futures import ThreadPoolExecutor

# 确保项目根目录在sys.path中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.run_realtime_cvd import ws_consume, parse_aggtrade_message, MonitoringMetrics
from src.real_cvd_calculator import RealCVDCalculator, CVDConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class MultiSymbolValidator:
    def __init__(self, symbols: list, duration: int, mad_multiplier: float, 
                 scale_fast_weight: float, half_life_trades: int, winsor_limit: float):
        self.symbols = symbols
        self.duration = duration
        self.mad_multiplier = mad_multiplier
        self.scale_fast_weight = scale_fast_weight
        self.half_life_trades = half_life_trades
        self.winsor_limit = winsor_limit
        
        # 结果存储
        self.results = {}
        
    def create_calculator(self, symbol: str):
        """创建CVD计算器，使用A1优化参数"""
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
            symbol, 
            cfg=config,
            mad_multiplier=self.mad_multiplier,
            scale_fast_weight=self.scale_fast_weight
        )
        
        return calc
    
    async def validate_symbol(self, symbol: str):
        """验证单个交易对"""
        log.info(f"开始验证 {symbol} - 使用A1参数")
        
        url = f"wss://fstream.binancefuture.com/stream?streams={symbol.lower()}@aggTrade"
        q = asyncio.Queue(maxsize=50000)
        stop_evt = asyncio.Event()
        metrics = MonitoringMetrics()
        
        cvd_calculator = self.create_calculator(symbol)
        data = []
        seen_ids = deque(maxlen=4096)
        start_time = time.time()
        
        # 60秒滑窗用于recv_rate计算
        rate_window = deque(maxlen=60)
        
        async def consume_and_process():
            while not stop_evt.is_set():
                try:
                    ts_recv, raw_msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    parsed = parse_aggtrade_message(raw_msg)
                    if parsed:
                        price, qty, is_buy, event_ms, agg_trade_id = parsed
                        
                        # 去重检查
                        if agg_trade_id in seen_ids:
                            continue
                        seen_ids.append(agg_trade_id)
                        
                        # 更新CVD计算器
                        result = cvd_calculator.update_with_trade(
                            price=price,
                            qty=qty,
                            is_buy=is_buy,
                            event_time_ms=event_ms
                        )
                        
                        # 获取Z分数
                        z_info = cvd_calculator.get_last_zscores()
                        
                        # 计算recv_rate
                        rate_window.append(ts_recv)
                        if len(rate_window) >= 2:
                            recv_rate = len(rate_window) / (rate_window[-1] - rate_window[0])
                        else:
                            recv_rate = 0.0
                        
                        # 计算延迟
                        latency_ms = (ts_recv * 1000 - event_ms) if event_ms else 0.0
                        
                        # 存储数据
                        data.append({
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
                            'latency_ms': latency_ms
                        })
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    log.error(f"处理消息错误 ({symbol}): {e}")
        
        # 启动任务
        consumer_task = asyncio.create_task(
            ws_consume(url, q, stop_evt, metrics, 60, 30, 20, 10)
        )
        processor_task = asyncio.create_task(consume_and_process())
        
        # 等待指定时间
        try:
            await asyncio.wait_for(stop_evt.wait(), timeout=self.duration)
        except asyncio.TimeoutError:
            log.info(f"验证时间到达 ({self.duration}秒) - {symbol}")
            stop_evt.set()
        finally:
            await asyncio.sleep(1)
            consumer_task.cancel()
            processor_task.cancel()
            await asyncio.gather(consumer_task, processor_task, return_exceptions=True)
        
        # 分析数据
        if not data:
            log.warning(f"{symbol}: 没有收集到数据")
            return None
        
        df = pd.DataFrame(data)
        valid_data = df[(~df['is_warmup']) & (~df['is_flat'])].copy()
        
        if len(valid_data) < 100:
            log.warning(f"{symbol}: 有效样本量不足 ({len(valid_data)})")
            return None
        
        # 动态阈值切箱
        thr = valid_data['recv_rate'].quantile(0.60)
        active_data = valid_data[valid_data['recv_rate'] >= thr].copy()
        quiet_data = valid_data[valid_data['recv_rate'] < thr].copy()
        
        # 计算统计
        def calc_stats(data, name):
            if len(data) == 0:
                return {'name': name, 'n': 0, 'p_z_gt_2': 0.0, 'p_z_gt_3': 0.0}
            
            z_abs = np.abs(data['z_cvd'])
            p_z_gt_2 = np.mean(z_abs > 2.0)
            p_z_gt_3 = np.mean(z_abs > 3.0)
            median_z = np.median(z_abs)
            p95_z = np.percentile(z_abs, 95)
            
            return {
                'name': name,
                'n': len(data),
                'p_z_gt_2': p_z_gt_2,
                'p_z_gt_3': p_z_gt_3,
                'median_z': median_z,
                'p95_z': p95_z
            }
        
        overall_stats = calc_stats(valid_data, 'OVERALL')
        active_stats = calc_stats(active_data, 'ACTIVE')
        quiet_stats = calc_stats(quiet_data, 'QUIET')
        
        # 延迟统计
        latency_p50 = np.percentile(valid_data['latency_ms'], 50)
        latency_p90 = np.percentile(valid_data['latency_ms'], 90)
        latency_p99 = np.percentile(valid_data['latency_ms'], 99)
        
        result = {
            'symbol': symbol,
            'duration': self.duration,
            'total_samples': len(valid_data),
            'threshold_recv_rate': thr,
            'overall': overall_stats,
            'active': active_stats,
            'quiet': quiet_stats,
            'latency_p50': latency_p50,
            'latency_p90': latency_p90,
            'latency_p99': latency_p99,
            'metrics': metrics.to_dict()
        }
        
        log.info(f"{symbol} 验证完成: P(|Z|>2)={overall_stats['p_z_gt_2']:.4f}, P(|Z|>3)={overall_stats['p_z_gt_3']:.4f}")
        
        return result
    
    async def run_validation(self):
        """运行多交易对验证"""
        log.info(f"开始多交易对验证 - 使用A1参数")
        log.info(f"交易对: {self.symbols}")
        log.info(f"验证时长: {self.duration}秒")
        log.info(f"参数: MAD={self.mad_multiplier}, FAST={self.scale_fast_weight}, HL={self.half_life_trades}")
        
        # 并发验证所有交易对
        tasks = [self.validate_symbol(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.error(f"验证 {self.symbols[i]} 失败: {result}")
                self.results[self.symbols[i]] = None
            else:
                self.results[self.symbols[i]] = result
        
        return self.results
    
    def generate_report(self):
        """生成验证报告"""
        log.info("\n" + "="*80)
        log.info("多交易对验证报告 - Phase A1参数")
        log.info("="*80)
        
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        if not valid_results:
            log.error("没有成功的验证结果")
            return
        
        # 汇总统计
        p_z_gt_2_values = [r['overall']['p_z_gt_2'] for r in valid_results.values()]
        p_z_gt_3_values = [r['overall']['p_z_gt_3'] for r in valid_results.values()]
        
        log.info(f"\n验证结果汇总:")
        log.info(f"  成功验证: {len(valid_results)}/{len(self.symbols)} 个交易对")
        log.info(f"  P(|Z|>2) 平均: {np.mean(p_z_gt_2_values):.4f}")
        log.info(f"  P(|Z|>2) 范围: {np.min(p_z_gt_2_values):.4f} - {np.max(p_z_gt_2_values):.4f}")
        log.info(f"  P(|Z|>3) 平均: {np.mean(p_z_gt_3_values):.4f}")
        log.info(f"  P(|Z|>3) 范围: {np.min(p_z_gt_3_values):.4f} - {np.max(p_z_gt_3_values):.4f}")
        
        # 详细结果
        log.info(f"\n详细结果:")
        for symbol, result in valid_results.items():
            log.info(f"\n{symbol}:")
            log.info(f"  样本量: {result['total_samples']}")
            log.info(f"  P(|Z|>2): {result['overall']['p_z_gt_2']:.4f}")
            log.info(f"  P(|Z|>3): {result['overall']['p_z_gt_3']:.4f}")
            log.info(f"  Active: {result['active']['n']} 样本, P(|Z|>2)={result['active']['p_z_gt_2']:.4f}")
            log.info(f"  Quiet: {result['quiet']['n']} 样本, P(|Z|>2)={result['quiet']['p_z_gt_2']:.4f}")
            log.info(f"  延迟P99: {result['latency_p99']:.1f}ms")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/multi_symbol_validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"multi_symbol_validation_{timestamp}.json"
        
        save_data = {
            'test_info': {
                'timestamp': timestamp,
                'symbols': self.symbols,
                'duration': self.duration,
                'parameters': {
                    'mad_multiplier': self.mad_multiplier,
                    'scale_fast_weight': self.scale_fast_weight,
                    'half_life_trades': self.half_life_trades,
                    'winsor_limit': self.winsor_limit
                }
            },
            'results': valid_results,
            'summary': {
                'successful_symbols': len(valid_results),
                'total_symbols': len(self.symbols),
                'p_z_gt_2_mean': float(np.mean(p_z_gt_2_values)),
                'p_z_gt_2_std': float(np.std(p_z_gt_2_values)),
                'p_z_gt_3_mean': float(np.mean(p_z_gt_3_values)),
                'p_z_gt_3_std': float(np.std(p_z_gt_3_values))
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        log.info(f"\n详细结果已保存到: {output_file}")

async def main():
    parser = argparse.ArgumentParser(description='CVD多交易对验证')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'ADAUSDT'], 
                       help='要验证的交易对列表')
    parser.add_argument('--duration', type=int, default=600, help='每个交易对验证时长(秒)')
    parser.add_argument('--mad', type=float, default=1.8, help='MAD乘数')
    parser.add_argument('--fast', type=float, default=0.20, help='快速权重')
    parser.add_argument('--hl', type=int, default=600, help='半衰期')
    parser.add_argument('--winsor', type=float, default=8.0, help='Winsor限制')
    
    args = parser.parse_args()
    
    validator = MultiSymbolValidator(
        symbols=args.symbols,
        duration=args.duration,
        mad_multiplier=args.mad,
        scale_fast_weight=args.fast,
        half_life_trades=args.hl,
        winsor_limit=args.winsor
    )
    
    await validator.run_validation()
    validator.generate_report()

if __name__ == "__main__":
    asyncio.run(main())
