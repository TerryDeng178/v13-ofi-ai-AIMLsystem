#!/usr/bin/env python3
"""
CVD时间衰减EWMA A/B测试脚本
测试P0修复：时间衰减EWMA + 活动度自适应 vs 原始按笔更新

A测试：高活跃时段（22:00-22:20）
B测试：低活跃时段（03:31-03:51）
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

class TimeDecayABTestCollector:
    def __init__(self, symbol: str, duration: int, test_type: str):
        self.symbol = symbol
        self.duration = duration
        self.test_type = test_type  # "A" (高活跃) 或 "B" (低活跃)
        
        # 数据收集
        self.data = []
        self.seen_ids = deque(maxlen=4096)  # 去重
        self.start_time = None
        
        # 配置参数（A1最优参数）
        self.mad_multiplier = 1.8
        self.scale_fast_weight = 0.20
        self.half_life_trades = 600
        self.winsor_limit = 8.0
        
        # 测试配置
        self.config_old = self._create_config(use_time_decay=False)
        self.config_new = self._create_config(use_time_decay=True)
        
    def _create_config(self, use_time_decay: bool) -> CVDConfig:
        """创建测试配置"""
        config = CVDConfig(
            z_mode="delta",
            scale_mode="hybrid",
            mad_multiplier=self.mad_multiplier,
            scale_fast_weight=self.scale_fast_weight,
            half_life_trades=self.half_life_trades,
            winsor_limit=self.winsor_limit,
            freeze_min=50,
            stale_threshold_ms=5000,
            soft_freeze_ms=4000,
            hard_freeze_ms=5000,
            ewma_fast_hl=80,
            mad_window_trades=300,
            mad_scale_factor=1.4826,
            scale_slow_weight=0.7
        )
        
        # 设置时间衰减标志（通过环境变量）
        if use_time_decay:
            os.environ['CVD_USE_TIME_DECAY'] = '1'
        else:
            os.environ['CVD_USE_TIME_DECAY'] = '0'
            
        return config
    
    async def run_test(self, config: CVDConfig, label: str) -> dict:
        """运行单个测试"""
        log.info(f"开始{label}测试...")
        
        # 创建计算器
        calc = RealCVDCalculator(self.symbol, config)
        
        # WebSocket连接
        url = f"wss://fstream.binancefuture.com/stream?streams={self.symbol.lower()}@aggTrade"
        q: asyncio.Queue = asyncio.Queue(maxsize=50000)
        stop_evt = asyncio.Event()
        metrics = MonitoringMetrics()
        
        # 数据收集
        data = []
        seen_ids = deque(maxlen=4096)
        start_time = time.time()
        
        async def collect_data():
            nonlocal data, start_time
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
                        result = calc.update_with_trade(
                            price=price,
                            qty=qty,
                            is_buy=is_buy,
                            event_time_ms=event_ms
                        )
                        
                        if result['z_cvd'] is not None:
                            # 计算recv_rate（60秒滑动窗口）
                            recv_rate = metrics.total_messages / (ts_recv - start_time) if ts_recv > start_time else 0.0
                            
                            # 获取尺度诊断
                            z_stats = calc.get_z_stats()
                            
                            data.append({
                                'timestamp': ts_recv,
                                'event_time_ms': event_ms,
                                'price': price,
                                'qty': qty,
                                'is_buy': is_buy,
                                'z_cvd': result['z_cvd'],
                                'z_raw': result.get('z_raw'),
                                'cvd': result['cvd'],
                                'recv_rate': recv_rate,
                                'floor_used': z_stats.get('floor_used', False),
                                'scale': z_stats.get('scale', 0.0),
                                'sigma_floor': z_stats.get('sigma_floor', 0.0),
                                'current_tps': z_stats.get('current_tps', 0.0),
                                'boost': z_stats.get('boost', 1.0),
                                'w_fast_eff': z_stats.get('w_fast_eff', 0.0),
                                'w_slow_eff': z_stats.get('w_slow_eff', 0.0)
                            })
                            
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    log.error(f"Error processing message: {e}")
        
        # 启动任务
        consumer_task = asyncio.create_task(
            ws_consume(url, q, stop_evt, metrics, 60, 30, 20, 10)
        )
        collector_task = asyncio.create_task(collect_data())
        
        # 等待测试完成
        try:
            await asyncio.wait_for(stop_evt.wait(), timeout=self.duration)
        except asyncio.TimeoutError:
            log.info(f"测试时长达到 ({self.duration}s). 停止...")
            stop_evt.set()
        finally:
            await asyncio.sleep(1)
            consumer_task.cancel()
            collector_task.cancel()
            await asyncio.gather(consumer_task, collector_task, return_exceptions=True)
        
        # 分析结果
        if not data:
            return {"error": "No data collected"}
        
        df = pd.DataFrame(data)
        
        # 计算指标
        z_abs = df['z_cvd'].abs()
        p_z_gt_2 = (z_abs > 2).mean()
        p_z_gt_3 = (z_abs > 3).mean()
        median_z = z_abs.median()
        p95_z = z_abs.quantile(0.95)
        
        # 尺度指标
        scale_median = df['scale'].median()
        sigma_floor_median = df['sigma_floor'].median()
        floor_hit_rate = df['floor_used'].mean()
        
        # 活动度指标
        recv_rate_median = df['recv_rate'].median()
        current_tps_median = df['current_tps'].median()
        boost_median = df['boost'].median()
        
        # 权重指标
        w_fast_eff_median = df['w_fast_eff'].median()
        w_slow_eff_median = df['w_slow_eff'].median()
        
        return {
            'label': label,
            'n': len(df),
            'p_z_gt_2': p_z_gt_2,
            'p_z_gt_3': p_z_gt_3,
            'median_z': median_z,
            'p95_z': p95_z,
            'scale_median': scale_median,
            'sigma_floor_median': sigma_floor_median,
            'floor_hit_rate': floor_hit_rate,
            'recv_rate_median': recv_rate_median,
            'current_tps_median': current_tps_median,
            'boost_median': boost_median,
            'w_fast_eff_median': w_fast_eff_median,
            'w_slow_eff_median': w_slow_eff_median,
            'raw_data': df.to_dict('records')  # 保存原始数据用于进一步分析
        }
    
    async def run_ab_test(self) -> dict:
        """运行A/B测试"""
        log.info(f"开始{self.test_type}测试 - {self.symbol} ({self.duration}秒)")
        
        # 测试A：原始按笔更新
        log.info("运行测试A：原始按笔更新")
        result_a = await self.run_test(self.config_old, "原始按笔更新")
        
        # 等待5秒
        await asyncio.sleep(5)
        
        # 测试B：时间衰减EWMA
        log.info("运行测试B：时间衰减EWMA")
        result_b = await self.run_test(self.config_new, "时间衰减EWMA")
        
        # 计算改善指标
        if 'error' not in result_a and 'error' not in result_b:
            p2_improvement = (result_a['p_z_gt_2'] - result_b['p_z_gt_2']) / result_a['p_z_gt_2'] * 100
            p3_improvement = (result_a['p_z_gt_3'] - result_b['p_z_gt_3']) / result_a['p_z_gt_3'] * 100
            scale_improvement = (result_b['scale_median'] - result_a['scale_median']) / result_a['scale_median'] * 100
            
            result_a['p2_improvement'] = p2_improvement
            result_a['p3_improvement'] = p3_improvement
            result_a['scale_improvement'] = scale_improvement
        
        return {
            'test_type': self.test_type,
            'symbol': self.symbol,
            'duration': self.duration,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'result_a': result_a,
            'result_b': result_b
        }

async def main():
    parser = argparse.ArgumentParser(description='CVD时间衰减EWMA A/B测试')
    parser.add_argument('--symbol', default='BTCUSDT', help='交易对')
    parser.add_argument('--duration', type=int, default=1200, help='测试时长（秒）')
    parser.add_argument('--test-type', choices=['A', 'B'], default='A', help='测试类型：A=高活跃，B=低活跃')
    parser.add_argument('--output-dir', default='./data/time_decay_ab_test', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行测试
    collector = TimeDecayABTestCollector(
        symbol=args.symbol,
        duration=args.duration,
        test_type=args.test_type
    )
    
    results = await collector.run_ab_test()
    
    # 保存结果
    timestamp = results['timestamp']
    output_file = output_dir / f"time_decay_ab_test_{args.test_type}_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    log.info(f"测试结果已保存到: {output_file}")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print(f"时间衰减EWMA A/B测试结果 - {args.test_type}测试")
    print(f"{'='*60}")
    
    if 'error' not in results['result_a'] and 'error' not in results['result_b']:
        print(f"\n原始按笔更新:")
        print(f"  P(|Z|>2): {results['result_a']['p_z_gt_2']:.4f}")
        print(f"  P(|Z|>3): {results['result_a']['p_z_gt_3']:.4f}")
        print(f"  Scale中位数: {results['result_a']['scale_median']:.2f}")
        print(f"  Floor命中率: {results['result_a']['floor_hit_rate']:.4f}")
        
        print(f"\n时间衰减EWMA:")
        print(f"  P(|Z|>2): {results['result_b']['p_z_gt_2']:.4f}")
        print(f"  P(|Z|>3): {results['result_b']['p_z_gt_3']:.4f}")
        print(f"  Scale中位数: {results['result_b']['scale_median']:.2f}")
        print(f"  Floor命中率: {results['result_b']['floor_hit_rate']:.4f}")
        
        print(f"\n改善指标:")
        print(f"  P(|Z|>2)改善: {results['result_a']['p2_improvement']:.1f}%")
        print(f"  P(|Z|>3)改善: {results['result_a']['p3_improvement']:.1f}%")
        print(f"  Scale改善: {results['result_a']['scale_improvement']:.1f}%")
        
        # 判断是否通过验收标准
        if args.test_type == 'A':  # 高活跃时段
            passed = (results['result_b']['p_z_gt_2'] <= 0.05 and 
                     results['result_b']['p_z_gt_3'] <= 0.01)
        else:  # 低活跃时段
            p2_improvement = results['result_a']['p2_improvement']
            scale_median = results['result_b']['scale_median']
            passed = (p2_improvement >= 60 and scale_median >= 20)
        
        print(f"\n验收结果: {'✅ 通过' if passed else '❌ 未通过'}")
    else:
        print("测试失败，请检查错误信息")

if __name__ == "__main__":
    asyncio.run(main())
