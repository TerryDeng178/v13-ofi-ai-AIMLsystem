#!/usr/bin/env python3
"""
Task_1.2.13 软冻结A/B测试
对比SOFT_FREEZE_V2=0/1的效果，重点关注z_after_silence_p3指标
"""

import asyncio
import json
import os
import sys
import time
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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class SoftFreezeABTest:
    def __init__(self):
        self.test_a_data = deque()  # SOFT_FREEZE_V2=0
        self.test_b_data = deque()  # SOFT_FREEZE_V2=1
        self.silence_periods = []
        
    async def run_ab_test(self, symbol: str, duration_per_test: int = 1200):  # 20分钟每个测试
        """运行软冻结A/B测试"""
        log.info(f"开始软冻结A/B测试")
        log.info(f"交易对: {symbol}, 每个测试持续时间: {duration_per_test}秒")
        
        # 基础配置
        base_config = {
            'CVD_Z_MODE': 'delta',
            'HALF_LIFE_TRADES': 100,
            'WINSOR_LIMIT': 8.0,  # 使用正确的评估值
            'STALE_THRESHOLD_MS': 5000,
            'FREEZE_MIN': 80,
            'SCALE_MODE': 'hybrid',
            'EWMA_FAST_HL': 80,
            'SCALE_FAST_WEIGHT': 0.15,
            'SCALE_SLOW_WEIGHT': 0.85,
            'MAD_WINDOW_TRADES': 300,
            'MAD_SCALE_FACTOR': 1.4826,
            'MAD_MULTIPLIER': 1.2,
        }
        
        # 测试A: SOFT_FREEZE_V2=0
        log.info("\n" + "="*50)
        log.info("测试A: SOFT_FREEZE_V2=0 (基础版本)")
        log.info("="*50)
        
        config_a = base_config.copy()
        config_a['SOFT_FREEZE_V2'] = 0
        
        result_a = await self._run_single_test(
            symbol, duration_per_test, config_a, "A", self.test_a_data
        )
        
        # 等待5秒
        await asyncio.sleep(5)
        
        # 测试B: SOFT_FREEZE_V2=1
        log.info("\n" + "="*50)
        log.info("测试B: SOFT_FREEZE_V2=1 (软冻结版本)")
        log.info("="*50)
        
        config_b = base_config.copy()
        config_b['SOFT_FREEZE_V2'] = 1
        
        result_b = await self._run_single_test(
            symbol, duration_per_test, config_b, "B", self.test_b_data
        )
        
        # 分析对比结果
        return self._analyze_ab_results(result_a, result_b)
    
    async def _run_single_test(self, symbol: str, duration: int, config: dict, 
                             test_name: str, data_container: deque):
        """运行单个测试"""
        # 设置环境变量
        for key, value in config.items():
            os.environ[key] = str(value)
        
        # 初始化CVD计算器
        cvd_calculator = RealCVDCalculator(symbol)
        
        # WebSocket配置
        url = f"wss://fstream.binancefuture.com/stream?streams={symbol.lower()}@aggTrade"
        q = asyncio.Queue(maxsize=50000)
        stop_evt = asyncio.Event()
        metrics = MonitoringMetrics()
        
        # 数据收集
        start_time = time.time()
        async def collect_data():
            last_event_time = None
            silence_start = None
            
            while not stop_evt.is_set():
                try:
                    ts_recv, raw_msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    parsed = parse_aggtrade_message(raw_msg)
                    if parsed:
                        price, qty, is_buy, event_ms, agg_trade_id = parsed
                        
                        # 检测静默期
                        if last_event_time is not None:
                            silence_duration = (ts_recv * 1000) - last_event_time
                            if silence_duration > 1000:  # 1秒无数据视为静默
                                if silence_start is None:
                                    silence_start = last_event_time
                            else:
                                if silence_start is not None:
                                    # 静默期结束
                                    self.silence_periods.append({
                                        'start': silence_start,
                                        'end': last_event_time,
                                        'duration': last_event_time - silence_start
                                    })
                                    silence_start = None
                        
                        last_event_time = ts_recv * 1000
                        
                        # 计算CVD Z-score
                        cvd_result = cvd_calculator.update_with_trade(
                            price=price, qty=qty, is_buy=is_buy, event_time_ms=event_ms
                        )
                        
                        # 获取Z-score
                        z_cvd, is_warmup, is_flat = cvd_calculator._z_last_excl()
                        
                        if z_cvd is not None and not is_warmup:
                            # 计算接收率（基于总消息数）
                            recv_rate = metrics.total_messages / (ts_recv - start_time) if (ts_recv - start_time) > 0 else 0
                            
                            data_container.append({
                                'timestamp': ts_recv,
                                'event_time_ms': event_ms,
                                'price': price,
                                'qty': qty,
                                'is_buy': is_buy,
                                'z_cvd': z_cvd,
                                'z_raw': z_cvd,  # 暂时使用相同值
                                'recv_rate': recv_rate,
                                'latency_ms': (ts_recv * 1000 - event_ms) if event_ms else 0.0,
                                'is_after_silence': silence_start is not None
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
        
        start_time = time.time()
        log.info(f"开始测试{test_name}数据收集...")
        
        try:
            await asyncio.wait_for(stop_evt.wait(), timeout=duration)
        except asyncio.TimeoutError:
            log.info(f"测试{test_name}时间到达 ({duration}秒)")
            stop_evt.set()
        finally:
            await asyncio.sleep(1)
            consumer_task.cancel()
            collector_task.cancel()
            await asyncio.gather(consumer_task, collector_task, return_exceptions=True)
        
        end_time = time.time()
        log.info(f"测试{test_name}完成，共收集 {len(data_container)} 条记录")
        
        return {
            'test_name': test_name,
            'config': config,
            'duration': end_time - start_time,
            'data_count': len(data_container),
            'metrics': metrics.to_dict()
        }
    
    def _analyze_ab_results(self, result_a, result_b):
        """分析A/B测试结果"""
        import pandas as pd
        import numpy as np
        
        # 转换为DataFrame
        df_a = pd.DataFrame(list(self.test_a_data))
        df_b = pd.DataFrame(list(self.test_b_data))
        
        # 过滤有效数据
        df_a = df_a[df_a['z_cvd'].notna()]
        df_b = df_b[df_b['z_cvd'].notna()]
        
        if len(df_a) == 0 or len(df_b) == 0:
            log.error("A/B测试数据不足")
            return None
        
        # 计算基础指标
        metrics_a = self._calculate_basic_metrics(df_a, "A")
        metrics_b = self._calculate_basic_metrics(df_b, "B")
        
        # 计算静默后指标
        silence_metrics_a = self._calculate_silence_metrics(df_a, "A")
        silence_metrics_b = self._calculate_silence_metrics(df_b, "B")
        
        # 计算改善程度
        improvements = self._calculate_improvements(metrics_a, metrics_b, silence_metrics_a, silence_metrics_b)
        
        return {
            'test_a': {
                'basic': metrics_a,
                'silence': silence_metrics_a,
                'result': result_a
            },
            'test_b': {
                'basic': metrics_b,
                'silence': silence_metrics_b,
                'result': result_b
            },
            'improvements': improvements,
            'silence_periods': self.silence_periods
        }
    
    def _calculate_basic_metrics(self, df, test_name):
        """计算基础指标"""
        z_values = df['z_cvd'].values
        
        return {
            'n': len(z_values),
            'p_z_gt_2': np.mean(np.abs(z_values) > 2),
            'p_z_gt_3': np.mean(np.abs(z_values) > 3),
            'median_z': np.median(z_values),
            'p95_z': np.percentile(z_values, 95),
            'mean_z': np.mean(z_values),
            'std_z': np.std(z_values)
        }
    
    def _calculate_silence_metrics(self, df, test_name):
        """计算静默后指标"""
        if 'is_after_silence' not in df.columns:
            return {
                'z_after_silence_p3': 0.0,
                'silence_count': 0,
                'avg_z_after_silence': 0.0
            }
        
        silence_data = df[df['is_after_silence'] == True]
        
        if len(silence_data) == 0:
            return {
                'z_after_silence_p3': 0.0,
                'silence_count': 0,
                'avg_z_after_silence': 0.0
            }
        
        z_silence = silence_data['z_cvd'].values
        
        return {
            'z_after_silence_p3': np.mean(np.abs(z_silence) > 3),
            'silence_count': len(silence_data),
            'avg_z_after_silence': np.mean(z_silence),
            'median_z_after_silence': np.median(z_silence)
        }
    
    def _calculate_improvements(self, metrics_a, metrics_b, silence_a, silence_b):
        """计算改善程度"""
        improvements = {}
        
        # 基础指标改善
        for key in ['p_z_gt_2', 'p_z_gt_3', 'median_z', 'p95_z']:
            if key in metrics_a and key in metrics_b:
                val_a = metrics_a[key]
                val_b = metrics_b[key]
                if val_a > 0:
                    improvement = (val_a - val_b) / val_a * 100
                    improvements[key] = {
                        'absolute': val_a - val_b,
                        'relative': improvement
                    }
        
        # 静默后指标改善
        if 'z_after_silence_p3' in silence_a and 'z_after_silence_p3' in silence_b:
            val_a = silence_a['z_after_silence_p3']
            val_b = silence_b['z_after_silence_p3']
            if val_a > 0:
                improvement = (val_a - val_b) / val_a * 100
                improvements['z_after_silence_p3'] = {
                    'absolute': val_a - val_b,
                    'relative': improvement
                }
        
        return improvements

async def main():
    symbol = os.getenv("SYMBOL", "BTCUSDT")
    duration = int(os.getenv("DURATION", "1200"))  # 20分钟每个测试
    
    ab_test = SoftFreezeABTest()
    results = await ab_test.run_ab_test(symbol, duration)
    
    if results:
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/cvd_soft_freeze_ab")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"soft_freeze_ab_{symbol.lower()}_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        log.info(f"软冻结A/B测试结果已保存到: {results_file}")
        
        # 打印摘要
        log.info("\n" + "="*60)
        log.info("软冻结A/B测试结果摘要")
        log.info("="*60)
        
        log.info(f"\n测试A (SOFT_FREEZE_V2=0):")
        log.info(f"  样本量: {results['test_a']['basic']['n']}")
        log.info(f"  P(|Z|>3): {results['test_a']['basic']['p_z_gt_3']:.4f}")
        log.info(f"  z_after_silence_p3: {results['test_a']['silence']['z_after_silence_p3']:.4f}")
        
        log.info(f"\n测试B (SOFT_FREEZE_V2=1):")
        log.info(f"  样本量: {results['test_b']['basic']['n']}")
        log.info(f"  P(|Z|>3): {results['test_b']['basic']['p_z_gt_3']:.4f}")
        log.info(f"  z_after_silence_p3: {results['test_b']['silence']['z_after_silence_p3']:.4f}")
        
        log.info(f"\n改善程度:")
        for metric, improvement in results['improvements'].items():
            log.info(f"  {metric}: {improvement['relative']:.1f}% 相对改善")

if __name__ == "__main__":
    asyncio.run(main())
