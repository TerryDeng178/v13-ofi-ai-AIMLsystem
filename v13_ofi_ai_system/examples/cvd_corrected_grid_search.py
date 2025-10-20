#!/usr/bin/env python3
"""
Task_1.2.13 修正网格搜索
使用正确的WINSOR_LIMIT=8.0进行3×3×3网格搜索
按照任务卡要求的窄域小步策略
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

class CorrectedGridSearch:
    def __init__(self):
        self.results = []
        
    async def run_corrected_grid_search(self, symbol: str, test_duration: int = 900):  # 15分钟每个测试
        """运行修正的网格搜索"""
        log.info(f"开始修正网格搜索 - 使用WINSOR_LIMIT=8.0")
        log.info(f"交易对: {symbol}, 每个测试持续时间: {test_duration}秒")
        
        # 按照任务卡要求的窄域小步参数范围
        param_ranges = {
            'MAD_MULTIPLIER': [1.46, 1.47, 1.48],
            'SCALE_FAST_WEIGHT': [0.30, 0.32, 0.35],
            'HALF_LIFE_TRADES': [280, 300, 320]
        }
        
        total_tests = len(param_ranges['MAD_MULTIPLIER']) * len(param_ranges['SCALE_FAST_WEIGHT']) * len(param_ranges['HALF_LIFE_TRADES'])
        log.info(f"总共需要测试 {total_tests} 个参数组合")
        
        test_count = 0
        
        for mad_mult in param_ranges['MAD_MULTIPLIER']:
            for scale_fast in param_ranges['SCALE_FAST_WEIGHT']:
                for half_life in param_ranges['HALF_LIFE_TRADES']:
                    test_count += 1
                    
                    log.info(f"\n{'='*60}")
                    log.info(f"测试 {test_count}/{total_tests}")
                    log.info(f"参数: MAD_MULTIPLIER={mad_mult}, SCALE_FAST_WEIGHT={scale_fast}, HALF_LIFE_TRADES={half_life}")
                    log.info(f"{'='*60}")
                    
                    # 运行单个测试
                    result = await self._run_single_test(
                        symbol, test_duration, mad_mult, scale_fast, half_life
                    )
                    
                    if result:
                        self.results.append(result)
                        log.info(
                            f"测试完成: P(|Z|>2)={result['overall']['p_z_gt_2']:.4f}, "
                            f"P(|Z|>3)={result['overall']['p_z_gt_3']:.4f}"
                        )
                    
                    # 测试间隔
                    if test_count < total_tests:
                        log.info("等待5秒后开始下一个测试...")
                        await asyncio.sleep(5)
        
        # 分析结果
        return self._analyze_results()
    
    async def _run_single_test(self, symbol: str, duration: int, mad_mult: float, 
                             scale_fast: float, half_life: int):
        """运行单个参数组合测试"""
        # 设置参数配置
        config = {
            'CVD_Z_MODE': 'delta',
            'HALF_LIFE_TRADES': half_life,
            'WINSOR_LIMIT': 8.0,  # 修正：使用任务卡要求的8.0
            'STALE_THRESHOLD_MS': 5000,
            'FREEZE_MIN': 80,
            'SCALE_MODE': 'hybrid',
            'EWMA_FAST_HL': 80,
            'SCALE_FAST_WEIGHT': scale_fast,
            'SCALE_SLOW_WEIGHT': 1.0 - scale_fast,
            'MAD_WINDOW_TRADES': 300,
            'MAD_SCALE_FACTOR': 1.4826,
            'MAD_MULTIPLIER': mad_mult,
            'SOFT_FREEZE_V2': 0,  # 基础版本
        }
        
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
        
        data = deque()
        # 统一起点时间，避免后续被覆盖
        t0 = time.time()
        async def collect_data():
            while not stop_evt.is_set():
                try:
                    ts_recv, raw_msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    parsed = parse_aggtrade_message(raw_msg)
                    if parsed:
                        price, qty, is_buy, event_ms, agg_trade_id = parsed
                        
                        # 自检：校验 is_buy 语义与原始 m 是否一致（如果拿得到）
                        try:
                            raw_m = raw_msg.get('data', {}).get('m', None)
                            if isinstance(raw_m, bool):
                                inferred_is_buy = (not raw_m)  # m=True => 卖出主动方
                                if inferred_is_buy != is_buy:
                                    log.warning("方向语义不一致：parse_aggtrade_message(is_buy) 与原始 m 推断不一致")
                        except Exception:
                            pass
                        
                        # 计算CVD Z-score
                        cvd_result = cvd_calculator.update_with_trade(
                            price=price, qty=qty, is_buy=is_buy, event_time_ms=event_ms
                        )
                        
                        # 获取Z-score
                        # TODO: 优先调用公开API；临时兼容私有函数
                        z_cvd, is_warmup, is_flat = cvd_calculator._z_last_excl()
                        
                        if z_cvd is not None and not is_warmup:
                            # 计算接收率（基于总消息数）
                            recv_rate = metrics.total_messages / (ts_recv - t0) if (ts_recv - t0) > 0 else 0
                            
                            data.append({
                                'timestamp': ts_recv,
                                'event_time_ms': event_ms,
                                'price': price,
                                'qty': qty,
                                'is_buy': is_buy,
                                'z_cvd': z_cvd,
                                # 如果无法拿到未截断 z_raw，这里置为 None，分析阶段做回退策略
                                'z_raw': None,
                                'recv_rate': recv_rate,
                                'latency_ms': (ts_recv * 1000 - event_ms) if event_ms else 0.0
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
        
        start_time = t0  # 与上面的 t0 对齐
        
        try:
            await asyncio.wait_for(stop_evt.wait(), timeout=duration)
        except asyncio.TimeoutError:
            log.info(f"测试时间到达 ({duration}秒)")
            stop_evt.set()
        finally:
            await asyncio.sleep(1)
            consumer_task.cancel()
            collector_task.cancel()
            await asyncio.gather(consumer_task, collector_task, return_exceptions=True)
        
        end_time = time.time()
        
        # 分析数据
        if len(data) == 0:
            log.error("没有收集到有效数据")
            return None
        
        return self._analyze_single_test(data, config, end_time - start_time)
    
    def _analyze_single_test(self, data, config, duration):
        """分析单个测试结果"""
        import pandas as pd
        import numpy as np
        # 避免额外依赖：不用 scipy，直接用 1.96 作为 95% 分位
        
        df = pd.DataFrame(list(data))
        df['z_cvd'] = pd.to_numeric(df['z_cvd'], errors='coerce')
        df['z_raw'] = pd.to_numeric(df['z_raw'], errors='coerce')
        df['recv_rate'] = pd.to_numeric(df['recv_rate'], errors='coerce')
        
        # 若 z_raw 缺失，则用 z_cvd 回退，同时标记
        if df['z_raw'].isna().all():
            df['z_raw'] = df['z_cvd']
            log.warning("未获取原始未截断Z（z_raw），本次分析以 z_cvd 回退代用")
        valid_data = df[df['z_cvd'].notna() & df['z_raw'].notna()]
        
        if len(valid_data) == 0:
            return None
        
        # 定义Regime
        valid_data['regime'] = valid_data['recv_rate'].apply(
            lambda x: 'Active' if x >= 1.0 else 'Quiet'
        )
        
        # 计算指标
        z_values = valid_data['z_cvd'].values
        z_raw_values = valid_data['z_raw'].values
        
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
        
        # Wilson置信区间
        def wilson_ci(p, n, confidence=0.95):
            if n == 0:
                return 0, 0
            z = 1.96  # 95% Z
            p_hat = p
            n_hat = n
            ci_lower = (p_hat + z**2/(2*n_hat) - z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n_hat))/n_hat)) / (1 + z**2/n_hat)
            ci_upper = (p_hat + z**2/(2*n_hat) + z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n_hat))/n_hat)) / (1 + z**2/n_hat)
            return ci_lower, ci_upper
        
        p_z_gt_2_ci = wilson_ci(p_z_gt_2, n)
        p_z_gt_3_ci = wilson_ci(p_z_gt_3, n)
        
        # Regime分析
        active_data = valid_data[valid_data['regime'] == 'Active']
        quiet_data = valid_data[valid_data['regime'] == 'Quiet']
        # 样本数兜底，避免小样本导致抖动
        MIN_N = 100
        if len(active_data) < MIN_N: active_data = active_data.iloc[0:0]
        if len(quiet_data)  < MIN_N: quiet_data  = quiet_data.iloc[0:0]
        
        active_metrics = None
        quiet_metrics = None
        
        if len(active_data) > 0:
            active_z = active_data['z_cvd'].values
            active_metrics = {
                'n': len(active_z),
                'p_z_gt_2': np.mean(np.abs(active_z) > 2),
                'p_z_gt_3': np.mean(np.abs(active_z) > 3),
                'median_z': np.median(active_z),
                'p95_z': np.percentile(active_z, 95)
            }
        
        if len(quiet_data) > 0:
            quiet_z = quiet_data['z_cvd'].values
            quiet_metrics = {
                'n': len(quiet_z),
                'p_z_gt_2': np.mean(np.abs(quiet_z) > 2),
                'p_z_gt_3': np.mean(np.abs(quiet_z) > 3),
                'median_z': np.median(quiet_z),
                'p95_z': np.percentile(quiet_z, 95)
            }
        
        return {
            'config': config,
            'duration': duration,
            'overall': {
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
            },
            'active': active_metrics,
            'quiet': quiet_metrics
        }
    
    def _analyze_results(self):
        """分析所有测试结果"""
        if not self.results:
            return None
        
        # 按P(|Z|>2)排序
        sorted_results = sorted(self.results, key=lambda x: x['overall']['p_z_gt_2'])
        
        # 生成排名表
        rank_table = []
        for i, result in enumerate(sorted_results):
            config = result['config']
            overall = result['overall']
            
            rank_table.append({
                'rank': i + 1,
                'mad_multiplier': config['MAD_MULTIPLIER'],
                'scale_fast_weight': config['SCALE_FAST_WEIGHT'],
                'half_life_trades': config['HALF_LIFE_TRADES'],
                'p_z_gt_2': overall['p_z_gt_2'],
                'p_z_gt_2_ci_upper': overall['p_z_gt_2_ci'][1],
                'p_z_gt_3': overall['p_z_gt_3'],
                'p_z_gt_3_ci_upper': overall['p_z_gt_3_ci'][1],
                'median_z': overall['median_z'],
                'p95_z': overall['p95_z'],
                'n': overall['n']
            })
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/cvd_corrected_grid_search")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存排名表
        rank_file = output_dir / f"grid_rank_table_{timestamp}.csv"
        import pandas as pd
        rank_df = pd.DataFrame(rank_table)
        rank_df.to_csv(rank_file, index=False, encoding='utf-8')
        
        # 保存详细结果
        results_file = output_dir / f"corrected_grid_search_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'rank_table': rank_table,
                'detailed_results': self.results
            }, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存样本数据（便于复盘）
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        for i, result in enumerate(self.results):
            config = result['config']
            config_str = f"mad{config['MAD_MULTIPLIER']}_scale{config['SCALE_FAST_WEIGHT']}_hl{config['HALF_LIFE_TRADES']}"
            sample_file = samples_dir / f"samples_{config_str}_{timestamp}.parquet"
            # 这里需要从原始数据重建DataFrame，暂时跳过
            log.info(f"样本数据保存路径: {sample_file}")
        
        log.info(f"修正网格搜索结果已保存到: {output_dir}")
        log.info(f"排名表: {rank_file}")
        log.info(f"详细结果: {results_file}")
        
        return {
            'rank_table': rank_table,
            'best_config': sorted_results[0] if sorted_results else None,
            'output_dir': output_dir
        }

async def main():
    symbol = os.getenv("SYMBOL", "BTCUSDT")
    duration = int(os.getenv("DURATION", "900"))  # 15分钟每个测试
    
    grid_search = CorrectedGridSearch()
    results = await grid_search.run_corrected_grid_search(symbol, duration)
    
    if results:
        log.info("\n" + "="*60)
        log.info("修正网格搜索结果摘要")
        log.info("="*60)
        
        best = results['best_config']
        if best:
            log.info(f"\n最佳配置:")
            log.info(f"  MAD_MULTIPLIER: {best['config']['MAD_MULTIPLIER']}")
            log.info(f"  SCALE_FAST_WEIGHT: {best['config']['SCALE_FAST_WEIGHT']}")
            log.info(f"  HALF_LIFE_TRADES: {best['config']['HALF_LIFE_TRADES']}")
            log.info(f"\n最佳指标:")
            log.info(f"  P(|Z|>2): {best['overall']['p_z_gt_2']:.4f} (CI: {best['overall']['p_z_gt_2_ci']})")
            log.info(f"  P(|Z|>3): {best['overall']['p_z_gt_3']:.4f} (CI: {best['overall']['p_z_gt_3_ci']})")
            log.info(f"  Median(|Z|): {best['overall']['median_z']:.4f}")
            log.info(f"  P95(|Z|): {best['overall']['p95_z']:.4f}")
            log.info(f"  P(|Z_raw|>3): {best['overall']['p_z_raw_gt_3']:.4f} (未截断)")

if __name__ == "__main__":
    asyncio.run(main())
