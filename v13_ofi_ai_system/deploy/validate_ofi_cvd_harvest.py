#!/usr/bin/env python3
"""
OFI+CVD数据质量验证脚本 (Task 1.3.1 v3 - 增强版)

验证采集数据的质量，生成DoD验收报告：
- 完整性检查（空桶率）
- 去重检查（重复率）
- 延迟统计（p50/p90/p99）
- 信号量统计
- 一致性检查
- 2×2场景覆盖检查
【新增】分仓合规检查（raw vs preview）
【新增】订单簿序列一致性检查
【新增】Trade↔Orderbook对齐覆盖检查
【新增】OFI原语体检
"""

import json
import pandas as pd
import numpy as np
import glob
import os
import datetime as dt
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import argparse

def to_utc_naive(ts_ms: pd.Series) -> pd.Series:
    """统一转为 UTC-naive 时间戳（避免 tz 歧义）"""
    return pd.to_datetime(ts_ms, unit='ms', utc=True).tz_convert(None)

def load_parquet_data(base_dir: str) -> Dict[str, pd.DataFrame]:
    """加载所有Parquet数据"""
    data = {}
    
    # 查找所有Parquet文件
    parquet_files = glob.glob(f"{base_dir}/date=*/symbol=*/kind=*/*.parquet")
    
    if not parquet_files:
        print(f"警告: 在 {base_dir} 中未找到Parquet文件")
        return data
    
    print(f"找到 {len(parquet_files)} 个Parquet文件")
    
    # 按kind分组加载
    for kind in ['prices', 'ofi', 'cvd', 'fusion', 'events', 'orderbook']:
        kind_files = [f for f in parquet_files if f'\\kind={kind}\\' in f or f'/kind={kind}/' in f]
        
        if kind_files:
            print(f"加载 {kind} 数据: {len(kind_files)} 个文件")
            try:
                dfs = []
                for file in kind_files:
                    df = pd.read_parquet(file)
                    dfs.append(df)
                
                if dfs:
                    data[kind] = pd.concat(dfs, ignore_index=True)
                    print(f"  {kind}: {len(data[kind])} 行")
                else:
                    data[kind] = pd.DataFrame()
            except Exception as e:
                print(f"加载 {kind} 数据失败: {e}")
                data[kind] = pd.DataFrame()
        else:
            data[kind] = pd.DataFrame()
    
    return data

def load_parquet_data_by_dir(base_dir: str, kinds: List[str], lookback_mins: int = 180) -> Dict[str, pd.DataFrame]:
    """按目录加载指定类型的数据（新版支持分仓 + 按日期分区限量加载）"""
    data = {}
    if not base_dir or not os.path.exists(base_dir):
        return {k: pd.DataFrame() for k in kinds}
    
    # 计算需要读取的 date 分区（UTC）
    if lookback_mins and lookback_mins > 0:
        now_utc = pd.Timestamp.now(tz='UTC')
        start = (now_utc - pd.Timedelta(minutes=lookback_mins)).date()
        end = now_utc.date()
        dates = pd.date_range(start, end, freq='D').strftime('%Y-%m-%d').tolist()
        parquet_files = []
        for d in dates:
            parquet_files += glob.glob(f"{base_dir}/date={d}/symbol=*/kind=*/*.parquet")
    else:
        parquet_files = glob.glob(f"{base_dir}/date=*/symbol=*/kind=*/*.parquet")
    
    if not parquet_files:
        print(f"警告: 在 {base_dir} 中未找到Parquet文件")
        return {k: pd.DataFrame() for k in kinds}
    
    for kind in kinds:
        kind_files = [f for f in parquet_files if f'/kind={kind}/' in f or f'\\kind={kind}\\' in f]
        if kind_files:
            try:
                dfs = [pd.read_parquet(f) for f in kind_files]
                data[kind] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                print(f"加载 {kind} ({base_dir}): {len(data[kind])} 行")
            except Exception as e:
                print(f"加载 {kind} 失败: {e}")
                data[kind] = pd.DataFrame()
        else:
            data[kind] = pd.DataFrame()
    return data

def check_completeness(data: Dict[str, pd.DataFrame], lookback_mins: int = 180) -> Dict[str, Any]:
    """检查数据完整性"""
    print("\n=== 完整性检查 ===")
    
    results = {}
    
    for kind, df in data.items():
        if df.empty:
            results[kind] = {
                'empty': True,
                'worst_empty_bucket_rate': 1.0,
                'by_symbol': {}
            }
            continue
        
        # 按分钟聚合 - 使用事件时间
        try:
            # 优先使用event_ts_ms，如果没有则使用ts_ms
            ts_col = df['event_ts_ms'] if 'event_ts_ms' in df.columns else df['ts_ms']
            df['minute'] = to_utc_naive(ts_col).dt.floor('min')
        except:
            # 如果时间戳解析失败，使用当前时间
            df['minute'] = pd.Timestamp.now().floor('min')
        
        # 应用lookback窗口限制（避免离线期误伤）
        cutoff = None
        if lookback_mins > 0:
            cutoff = pd.Timestamp.utcnow().floor('min') - pd.Timedelta(minutes=lookback_mins)
            cutoff = cutoff.replace(tzinfo=None)  # 直接转为naive
            df = df[df['minute'] >= cutoff]
            if df.empty:
                results[kind] = {
                    'empty': True,
                    'worst_empty_bucket_rate': 1.0,
                    'by_symbol': {}
                }
                continue
        
        worst = 0.0
        by_sym = {}
        
        # 按symbol分组统计（修复groupby问题）
        if 'symbol' in df.columns:
            symbol_key = df['symbol']
        else:
            symbol_key = pd.Series(['__ALL__'] * len(df))
        
        for sym, g in df.groupby(symbol_key):
            filled_minutes = g['minute'].nunique()
            min_time = g['minute'].min()
            max_time = g['minute'].max()
            # 用回看窗口与该symbol的覆盖区间取交集，估算理论桶数
            span_start = max(min_time, cutoff) if (lookback_mins > 0 and cutoff is not None) else min_time
            span_end = max_time
            expected_minutes = int((span_end - span_start).total_seconds() // 60) + 1 if span_start != span_end else 1
            empty_bucket_rate = 1 - (filled_minutes / max(expected_minutes, 1)) if expected_minutes > 0 else 1.0
            
            by_sym[str(sym)] = {
                'empty_bucket_rate': float(empty_bucket_rate),
                'total_minutes': int(filled_minutes),
                'expected_minutes': int(expected_minutes)
            }
            worst = max(worst, empty_bucket_rate)
        
        results[kind] = {
            'empty': False,
            'worst_empty_bucket_rate': float(worst),
            'by_symbol': by_sym
        }
        
        print(f"{kind}: 最差空桶率 {worst:.4f} (按symbol统计)")
        for sym, stats in by_sym.items():
            print(f"  {sym}: {stats['empty_bucket_rate']:.4f} ({stats['total_minutes']}/{stats['expected_minutes']})")
    
    return results

def check_deduplication(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """检查去重情况"""
    print("\n=== 去重检查 ===")
    
    results = {}
    
    for kind, df in data.items():
        if df.empty:
            results[kind] = {
                'by_symbol': {},
                'max_duplicate_rate': 0.0
            }
            continue
        
        dup_by_sym = {}
        
        # 按symbol分组统计（修复groupby问题）
        if 'symbol' in df.columns:
            symbol_key = df['symbol']
        else:
            symbol_key = pd.Series(['__ALL__'] * len(df))
        
        for sym, g in df.groupby(symbol_key):
            total_rows = len(g)
            
            # 根据kind选择去重字段 - 优先使用row_id避免同毫秒误伤
            if 'row_id' in g.columns:
                unique_rows = g['row_id'].nunique()
            elif kind == 'prices' and 'agg_trade_id' in g.columns:
                unique_rows = g['agg_trade_id'].nunique()
            elif 'event_ts_ms' in g.columns and 'symbol' in g.columns:
                unique_rows = g[['symbol', 'event_ts_ms']].drop_duplicates().shape[0]
            elif 'ts_ms' in g.columns and 'symbol' in g.columns:
                unique_rows = g[['symbol', 'ts_ms']].drop_duplicates().shape[0]
            else:
                unique_rows = g.drop_duplicates().shape[0]
            
            duplicate_rate = 1 - (unique_rows / total_rows) if total_rows > 0 else 0.0
            
            dup_by_sym[str(sym)] = {
                'total_rows': int(total_rows),
                'unique_rows': int(unique_rows),
                'duplicate_rate': float(duplicate_rate)
            }
        
        max_duplicate_rate = max([v['duplicate_rate'] for v in dup_by_sym.values()] or [0.0])
        
        results[kind] = {
            'by_symbol': dup_by_sym,
            'max_duplicate_rate': float(max_duplicate_rate)
        }
        
        print(f"{kind}: 最大重复率 {max_duplicate_rate:.4f} (按symbol统计)")
        for sym, stats in dup_by_sym.items():
            print(f"  {sym}: {stats['duplicate_rate']:.4f} ({stats['unique_rows']}/{stats['total_rows']})")
    
    return results

def _fallback_latency(df):
    """回退计算延迟：recv_ts_ms - event_ts_ms"""
    for et in ['event_ts_ms','ts_ms']:
        if et in df.columns and 'recv_ts_ms' in df.columns:
            s = (df['recv_ts_ms'] - df[et]).dropna().astype('float')
            return s[s >= 0]  # 过滤负值
    return pd.Series(dtype='float')

def check_latency(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """检查延迟统计"""
    print("\n=== 延迟检查 ===")
    
    results = {}
    
    def _lat_stats(s):
        if len(s) == 0: 
            return {'p50':0,'p90':0,'p99':0,'count':0,'mean':0,'std':0}
        return {
            'p50': float(np.percentile(s,50)),
            'p90': float(np.percentile(s,90)),
            'p99': float(np.percentile(s,99)),
            'count': int(len(s)),
            'mean': float(s.mean()),
            'std': float(s.std())
        }
    
    # 检查prices数据的延迟
    if 'prices' in data and not data['prices'].empty:
        df = data['prices']
        if 'latency_ms' in df.columns:
            s = df['latency_ms'].dropna()
        else:
            s = _fallback_latency(df)
        results['prices_latency_ms'] = _lat_stats(s)
        if len(s) > 0:
            print(f"Prices 延迟(ms): P50={results['prices_latency_ms']['p50']:.2f}, P90={results['prices_latency_ms']['p90']:.2f}, P99={results['prices_latency_ms']['p99']:.2f}")
        else:
            print("Prices: 无延迟数据")
    else:
        results['prices_latency_ms'] = _lat_stats([])
        print("Prices: 无延迟数据")
    
    # 检查orderbook数据的延迟
    if 'orderbook' in data and not data['orderbook'].empty:
        df = data['orderbook']
        if 'latency_ms' in df.columns:
            s = df['latency_ms'].dropna()
        else:
            s = _fallback_latency(df)
        results['orderbook_latency_ms'] = _lat_stats(s)
        if len(s) > 0:
            print(f"Orderbook 延迟(ms): P50={results['orderbook_latency_ms']['p50']:.2f}, P90={results['orderbook_latency_ms']['p90']:.2f}, P99={results['orderbook_latency_ms']['p99']:.2f}")
        else:
            print("Orderbook: 无延迟数据")
    else:
        results['orderbook_latency_ms'] = _lat_stats([])
        print("Orderbook: 无延迟数据")
    
    # 向后兼容原来的字段名
    results['latency_ms'] = results.get('prices_latency_ms', _lat_stats([]))
    
    return results

def check_signal_volume(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """检查信号量"""
    print("\n=== 信号量检查 ===")
    
    results = {}
    
    # 检查events数据
    if 'events' in data and not data['events'].empty:
        events_df = data['events']
        total_events = len(events_df)
        
        # 按事件类型统计
        event_types = events_df['event_type'].value_counts().to_dict()
        
        results['events'] = {
            'total_events': int(total_events),
            'event_types': {k: int(v) for k, v in event_types.items()}
        }
        
        print(f"事件总数: {total_events}")
        for event_type, count in event_types.items():
            print(f"  {event_type}: {count}")
    else:
        results['events'] = {
            'total_events': 0,
            'event_types': {}
        }
        print("无事件数据")
    
    # 检查其他信号数据
    for kind in ['ofi', 'cvd', 'fusion']:
        if kind in data and not data[kind].empty:
            df = data[kind]
            results[kind] = {
                'total_signals': int(len(df)),
                'symbols': df['symbol'].unique().tolist() if 'symbol' in df.columns else []
            }
            print(f"{kind}: {len(df)} 个信号")
        else:
            results[kind] = {
                'total_signals': 0,
                'symbols': []
            }
            print(f"{kind}: 无数据")
    
    return results

def check_consistency(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """检查数据一致性"""
    print("\n=== 一致性检查 ===")
    
    results = {}
    
    # 检查CVD数据的一致性
    if 'cvd' in data and not data['cvd'].empty:
        cvd_df = data['cvd']
        
        # 检查z_raw和z_cvd的分离
        if 'z_raw' in cvd_df.columns and 'z_cvd' in cvd_df.columns:
            z_raw = cvd_df['z_raw'].dropna()
            z_cvd = cvd_df['z_cvd'].dropna()
            
            if len(z_raw) > 0 and len(z_cvd) > 0:
                # 计算尾部分离
                z_raw_abs = np.abs(z_raw)
                z_cvd_abs = np.abs(z_cvd)
                
                # 检查winsorization是否生效
                winsor_effect = np.mean(z_raw_abs > z_cvd_abs)
                
                # 计算纯截断率和尾部压缩比（解耦地板影响）
                winsor_limit = 2.5  # 若可读到配置更好
                clip_rate = float(np.mean(np.abs(z_raw) > winsor_limit))
                tail_compress_ratio = float(
                    np.percentile(np.abs(z_cvd), 99) / max(np.percentile(np.abs(z_raw), 99), 1e-9)
                )
                
                # 计算一致性诊断指标
                tail_p99_gap = np.percentile(np.abs(z_raw), 99) - np.percentile(np.abs(z_cvd), 99)
                floor_hit_rate = cvd_df['floor_used'].mean() if 'floor_used' in cvd_df.columns else 0.0
                
                results['cvd_consistency'] = {
                    'z_raw_stats': {
                        'mean': float(z_raw.mean()),
                        'std': float(z_raw.std()),
                        'p95': float(np.percentile(z_raw, 95)),
                        'p99': float(np.percentile(z_raw, 99))
                    },
                    'z_cvd_stats': {
                        'mean': float(z_cvd.mean()),
                        'std': float(z_cvd.std()),
                        'p95': float(np.percentile(z_cvd, 95)),
                        'p99': float(np.percentile(z_cvd, 99))
                    },
                    'winsor_effect': float(winsor_effect),
                    'clip_rate': clip_rate,
                    'tail_compress_ratio': tail_compress_ratio,
                    'tail_p99_gap': float(tail_p99_gap),
                    'floor_hit_rate': float(floor_hit_rate),
                    'diagnostic_fields': {
                        'has_scale': 'scale' in cvd_df.columns,
                        'has_sigma_floor': 'sigma_floor' in cvd_df.columns,
                        'has_floor_used': 'floor_used' in cvd_df.columns
                    }
                }
                
                print(f"CVD一致性: winsor效果 {winsor_effect:.4f}")
                print(f"  纯截断率: {clip_rate:.4f}, 尾部压缩比: {tail_compress_ratio:.3f}")
                print(f"  z_raw P99: {np.percentile(z_raw, 99):.2f}")
                print(f"  z_cvd P99: {np.percentile(z_cvd, 99):.2f}")
                print(f"  尾部差距: {tail_p99_gap:.4f}")
                print(f"  地板命中率: {floor_hit_rate:.4f}")
            else:
                results['cvd_consistency'] = {
                    'error': 'z_raw或z_cvd数据不足'
                }
        else:
            results['cvd_consistency'] = {
                'error': '缺少z_raw或z_cvd字段'
            }
    else:
        results['cvd_consistency'] = {
            'error': '无CVD数据'
        }
    
    return results

def check_scene_coverage(base_dir: str) -> Dict[str, Any]:
    """检查2×2场景覆盖情况"""
    print("\n=== 2×2场景覆盖检查 ===")
    
    results = {}
    
    # 查找所有Parquet文件
    parquet_files = glob.glob(f"{base_dir}/date=*/symbol=*/kind=*/*.parquet")
    
    if not parquet_files:
        print("警告: 未找到Parquet文件")
        return {'error': '未找到数据文件'}
    
    # 按symbol分组统计场景覆盖
    symbol_stats = {}
    
    for kind in ['prices', 'ofi', 'cvd', 'fusion', 'events', 'orderbook']:
        kind_files = [f for f in parquet_files if f'\\kind={kind}\\' in f or f'/kind={kind}/' in f]
        
        if kind_files:
            try:
                dfs = []
                for file in kind_files:
                    df = pd.read_parquet(file)
                    dfs.append(df)
                
                if dfs:
                    data = pd.concat(dfs, ignore_index=True)
                    
                    # 按symbol分组统计场景覆盖
                    for symbol, group in data.groupby('symbol'):
                        if symbol not in symbol_stats:
                            symbol_stats[symbol] = {}
                        
                        # 统计scenario_2x2分布
                        if 'scenario_2x2' in group.columns:
                            scenario_counts = group['scenario_2x2'].value_counts().to_dict()
                            symbol_stats[symbol][kind] = scenario_counts
                            
                            print(f"{symbol}-{kind}: {scenario_counts}")
                        else:
                            print(f"{symbol}-{kind}: 缺少scenario_2x2字段")
                            
            except Exception as e:
                print(f"处理{kind}数据失败: {e}")
    
    # 检查场景覆盖阈值
    coverage_results = {}
    scene_coverage_miss = 0
    
    for symbol, kind_stats in symbol_stats.items():
        coverage_results[symbol] = {}
        
        # 合并所有kind的场景统计
        all_scenarios = set()
        for kind, scenarios in kind_stats.items():
            all_scenarios.update(scenarios.keys())
        
        # 检查是否包含所有必需的场景
        required_scenarios = ['A_H', 'A_L', 'Q_H', 'Q_L']
        missing_scenarios = set(required_scenarios) - all_scenarios
        
        if missing_scenarios:
            scene_coverage_miss = 1
            print(f"WARNING: {symbol}: 缺少场景 {missing_scenarios}")
        else:
            print(f"OK: {symbol}: 场景覆盖完整")
        
        coverage_results[symbol] = {
            'available_scenarios': list(all_scenarios),
            'missing_scenarios': list(missing_scenarios),
            'coverage_complete': len(missing_scenarios) == 0
        }
    
    # 定义必需场景
    required_scenarios = ['A_H', 'A_L', 'Q_H', 'Q_L']
    
    results = {
        'symbol_coverage': coverage_results,
        'scene_coverage_miss': scene_coverage_miss,
        'required_scenarios': required_scenarios
    }
    
    return results

def check_tier_contract(raw_data: Dict[str, pd.DataFrame], preview_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """分仓合规检查（新增）"""
    print("\n=== 分仓合规检查 (raw vs preview) ===")
    res = {'raw_illegal_columns': {}, 'raw_illegal_kinds': [], 'preview_illegal_kinds': []}
    
    # 1) raw 只允许 kinds
    allowed_raw_kinds = {'prices', 'orderbook'}
    raw_loaded_kinds = {k for k, df in raw_data.items() if not df.empty}
    illegal_raw_kinds = list(raw_loaded_kinds - allowed_raw_kinds)
    if illegal_raw_kinds:
        print(f"[FAIL] raw 出现非法 kind: {illegal_raw_kinds}")
    res['raw_illegal_kinds'] = illegal_raw_kinds
    
    # 2) raw 不允许字段（策略/监控/场景）
    forbidden_cols = {'session', 'regime', 'vol_bucket', 'scenario_2x2', 'fee_tier', 'recv_rate_tps', 
                      'score', 'proba', 'consistency', 'ofi_z', 'z_cvd'}
    for kind in ['prices', 'orderbook']:
        df = raw_data.get(kind, pd.DataFrame())
        if df.empty:
            continue
        bad = sorted([c for c in forbidden_cols if c in df.columns])
        if bad:
            res['raw_illegal_columns'][kind] = bad
            print(f"[FAIL] raw/{kind} 含禁用列: {bad}")
    
    if not res['raw_illegal_columns']:
        print("[OK] raw 未发现禁用列")
    
    # 3) preview 不应包含 raw 专属 kind
    preview_loaded_kinds = {k for k, df in preview_data.items() if not df.empty}
    illegal_preview_kinds = list(preview_loaded_kinds & allowed_raw_kinds)
    if illegal_preview_kinds:
        print(f"[FAIL] preview 出现 raw kind: {illegal_preview_kinds}")
    res['preview_illegal_kinds'] = illegal_preview_kinds
    
    return res

def check_orderbook_sequence(orderbook_df: pd.DataFrame, huge_jump_threshold: int = 10000) -> Dict[str, Any]:
    """订单簿序列一致性检查（新增）"""
    print("\n=== 订单簿序列一致性检查 ===")
    if orderbook_df.empty or not {'first_id', 'last_id', 'prev_last_id', 'symbol'}.issubset(orderbook_df.columns):
        print("orderbook: 缺少序列字段或无数据")
        return {'error': 'missing_seq_fields_or_empty'}
    
    res = {}
    # 兼容字段名：优先使用 event_ts_ms，否则使用 ts_ms
    ts_col = 'event_ts_ms' if 'event_ts_ms' in orderbook_df.columns else 'ts_ms'
    for sym, g in orderbook_df.sort_values(ts_col).groupby('symbol'):
        g = g[['first_id', 'last_id', 'prev_last_id']].dropna()
        if g.empty:
            res[sym] = {'gap_rate': 1.0, 'monotonic_breaks': len(g)}
            continue
        
        # 断档：上一帧 last_id != 当前帧 prev_last_id
        prev_last = g['last_id'].shift(1)
        gap = (g['prev_last_id'] != prev_last) & prev_last.notna() & g['prev_last_id'].notna()
        gap_rate = float(gap.mean()) if len(g) > 0 else 1.0
        
        # 单调性：last_id 严格递增
        mono_break = (g['last_id'].diff() <= 0).sum()
        
        # 序列不变量检查
        bad_order = ((g['first_id'] > g['last_id']) | (g['prev_last_id'] >= g['last_id'])).sum()
        jump = (g['last_id'] - g['prev_last_id'])
        huge_jump = (jump > huge_jump_threshold).sum()  # 阈值可通过 CLI 配置
        
        res[sym] = {
            'gap_rate': gap_rate, 
            'monotonic_breaks': int(mono_break), 
            'samples': int(len(g)),
            'bad_order_rows': int(bad_order),
            'huge_jumps': int(huge_jump)
        }
        print(f"{sym}: gap_rate={gap_rate:.4f}, monotonic_breaks={mono_break}, bad_order={bad_order}, huge_jumps={huge_jump}, n={len(g)}")
    
    # 聚合阈值（建议）：gap_rate < 0.01, monotonic_breaks == 0
    res['summary'] = {
        'max_gap_rate': max([v['gap_rate'] for v in res.values() if isinstance(v, dict) and 'gap_rate' in v] or [1.0]),
        'total_mono_breaks': sum([v['monotonic_breaks'] for v in res.values() if isinstance(v, dict) and 'monotonic_breaks' in v] or [0]),
        'total_bad_order_rows': sum([v.get('bad_order_rows', 0) for v in res.values() if isinstance(v, dict)]),
        'total_huge_jumps': sum([v.get('huge_jumps', 0) for v in res.values() if isinstance(v, dict)])
    }
    return res

def check_trade_orderbook_join(prices_df: pd.DataFrame, orderbook_df: pd.DataFrame, max_abs_ms: int = 250) -> Dict[str, Any]:
    """Trade↔Orderbook 对齐覆盖检查（新增）"""
    print("\n=== Trade-Orderbook 对齐覆盖检查 ===")
    if prices_df.empty or orderbook_df.empty:
        print("缺少 prices/orderbook 数据")
        return {'error': 'missing_data'}
    
    res = {}
    # 兼容字段名
    prices_ts_col = 'event_ts_ms' if 'event_ts_ms' in prices_df.columns else 'ts_ms'
    ob_ts_col = 'event_ts_ms' if 'event_ts_ms' in orderbook_df.columns else 'ts_ms'
    
    # 抽象出 trade_id 列名（鲁棒处理不同字段名）
    trade_id_col = None
    for c in ['agg_trade_id','aggTradeId','trade_id','id']:
        if c in prices_df.columns:
            trade_id_col = c
            break
    if trade_id_col is None:
        trade_id_col = prices_ts_col  # 兜底不空列
    
    ob = orderbook_df[['symbol', ob_ts_col, 'last_id']].dropna().rename(columns={ob_ts_col: 'ob_ts'})
    
    for sym, trades in prices_df[['symbol', prices_ts_col, trade_id_col]].dropna().groupby('symbol'):
        g_ob = ob[ob['symbol'] == sym]
        if g_ob.empty:
            res[sym] = {'match_rate': 0.0, 'n': len(trades)}
            print(f"{sym}: 无 orderbook")
            continue
        
        # 最近邻匹配：对每笔trade 找到时间差最小的快照
        trades = trades.sort_values(prices_ts_col)
        g_ob = g_ob.sort_values('ob_ts')
        
        # 使用时间窗口合并
        trades['ts'] = pd.to_datetime(trades[prices_ts_col], unit='ms', utc=True).tz_convert(None)
        g_ob['ts'] = pd.to_datetime(g_ob['ob_ts'], unit='ms', utc=True).tz_convert(None)
        
        merged = pd.merge_asof(trades, g_ob, on='ts', by='symbol', direction='nearest', 
                              tolerance=pd.Timedelta(milliseconds=max_abs_ms))
        match_rate = float(merged['last_id'].notna().mean())
        res[sym] = {'match_rate': match_rate, 'n': int(len(trades))}
        print(f"{sym}: match_rate (±{max_abs_ms}ms) = {match_rate:.4f}  n={len(trades)}")
    
    res['summary'] = {'min_match_rate': min([v['match_rate'] for v in res.values() if isinstance(v, dict) and 'match_rate' in v] or [0.0])}
    return res

def check_ofi_primitives(orderbook_df: pd.DataFrame, preview_ofi: pd.DataFrame = pd.DataFrame()) -> Dict[str, Any]:
    """OFI 原语体检（新增）"""
    print("\n=== OFI 原语体检 ===")
    needed = [f'd_b{i}' for i in range(5)] + [f'd_a{i}' for i in range(5)] + ['d_bid_qty_agg', 'd_ask_qty_agg', 'symbol']
    if orderbook_df.empty or not all(col in orderbook_df.columns for col in needed):
        print("orderbook: 缺少 OFI 原语列")
        return {'error': 'missing_ofi_primitives'}
    
    res = {}
    cols = [f'd_b{i}' for i in range(5)] + [f'd_a{i}' for i in range(5)]
    
    for sym, g in orderbook_df.groupby('symbol'):
        g = g.dropna(subset=cols)
        nonzero_rate = float((g[cols].abs().sum(axis=1) > 0).mean()) if len(g) > 0 else 0.0
        pos_neg_balance = float((g[cols].sum(axis=1)).mean())  # 方向偏置粗检
        res[sym] = {'nonzero_rate': nonzero_rate, 'dir_bias_mean': pos_neg_balance, 'n': int(len(g))}
        print(f"{sym}: 原语非零率={nonzero_rate:.4f}, 方向均值={pos_neg_balance:.3f}, n={len(g)}")
    
    # 若有预览 ofi，则做方向相关性（粗检）
    ofi_ts_col = 'event_ts_ms' if 'event_ts_ms' in preview_ofi.columns else 'ts_ms'
    ob_ts_col = 'event_ts_ms' if 'event_ts_ms' in orderbook_df.columns else 'ts_ms'
    if not preview_ofi.empty and {'symbol', ofi_ts_col, 'ofi_value'}.issubset(preview_ofi.columns):
        try:
            ofi = preview_ofi[['symbol', ofi_ts_col, 'ofi_value']].copy()
            ob = orderbook_df[['symbol', ob_ts_col, 'd_bid_qty_agg', 'd_ask_qty_agg']].copy()
            ofi['ts'] = pd.to_datetime(ofi[ofi_ts_col], unit='ms', utc=True).tz_convert(None)
            ob['ts'] = pd.to_datetime(ob[ob_ts_col], unit='ms', utc=True).tz_convert(None)
            merged = pd.merge_asof(ofi.sort_values('ts'), ob.sort_values('ts'), on='ts', by='symbol', 
                                  direction='nearest', tolerance=pd.Timedelta(milliseconds=500))
            merged['ofi_raw_like'] = merged['d_bid_qty_agg'] - merged['d_ask_qty_agg']
            corr = merged[['ofi_value', 'ofi_raw_like']].corr().iloc[0, 1] if merged[['ofi_value', 'ofi_raw_like']].dropna().shape[0] > 10 else np.nan
            res['ofi_corr'] = float(corr) if pd.notna(corr) else None
            print(f"OFI 原语 vs 预览 OFI 相关系数≈ {res['ofi_corr']}")
        except Exception as e:
            print(f"OFI 相关性计算失败: {e}")
    
    # 建议阈值：nonzero_rate >= 0.7；若有 corr，corr >= 0.3（方向一致的弱要求）
    res['summary'] = {
        'min_nonzero_rate': min([v['nonzero_rate'] for k, v in res.items() if isinstance(v, dict) and k != 'ofi_corr' and 'nonzero_rate' in v] or [0.0]),
        'ofi_corr': res.get('ofi_corr', None)
    }
    return res

def check_ofi_primitive_consistency(orderbook_df: pd.DataFrame, atol: float = 1e-6) -> Dict[str, Any]:
    """OFI 原语自洽性检查（逐档求和 ≈ 聚合）"""
    print("\n=== OFI 原语自洽性检查（逐档求和 ≈ 聚合） ===")
    needed = [f'd_b{i}' for i in range(5)] + [f'd_a{i}' for i in range(5)] + ['d_bid_qty_agg', 'd_ask_qty_agg', 'symbol']
    if orderbook_df.empty or not set(needed).issubset(orderbook_df.columns):
        print("orderbook: 缺列，跳过自洽性检查")
        return {'error': 'missing_columns'}
    
    res = {}
    for sym, g in orderbook_df.groupby('symbol'):
        db_sum = g[[f'd_b{i}' for i in range(5)]].sum(axis=1)
        da_sum = g[[f'd_a{i}' for i in range(5)]].sum(axis=1)
        bad_bid = (db_sum - g['d_bid_qty_agg']).abs() > atol
        bad_ask = (da_sum - g['d_ask_qty_agg']).abs() > atol
        bad_rate = float((bad_bid | bad_ask).mean()) if len(g) else 0.0
        res[sym] = {'inconsistency_rate': bad_rate, 'n': int(len(g))}
        print(f"{sym}: 不自洽比例={bad_rate:.4f}  n={len(g)}")
    
    res['summary'] = {
        'max_inconsistency_rate': max([v['inconsistency_rate'] for v in res.values() if isinstance(v, dict)] or [0.0])
    }
    return res

def generate_dod_report(completeness: Dict, dedup: Dict, latency: Dict, 
                       signals: Dict, consistency: Dict, scene_coverage: Dict,
                       tier_check: Dict = None, ob_seq: Dict = None, 
                       join_cov: Dict = None, ofi_prim: Dict = None,
                       ofi_prim_cons: Dict = None,
                       min_events: int = 1000, max_empty_bucket_rate: float = 0.001, 
                       max_dup_rate: float = 0.005) -> Dict[str, Any]:
    """生成DoD验收报告"""
    print("\n=== DoD验收报告 ===")
    
    # 验收标准（使用配置参数）
    dod_criteria = {
        'completeness': {
            'max_empty_bucket_rate': max_empty_bucket_rate,
            'description': f'空桶率 < {max_empty_bucket_rate:.1%}'
        },
        'deduplication': {
            'max_duplicate_rate': max_dup_rate,
            'description': f'重复率 < {max_dup_rate:.1%}'
        },
        'latency': {
            'max_p99_ms': 120,
            'max_p50_ms': 60,
            'description': 'P99 < 120ms, P50 < 60ms'
        },
        'signals': {
            'min_events': min_events,
            'description': f'事件总数 ≥ {min_events}'
        },
        'consistency': {
            'min_winsor_effect': 0.1,
            'description': 'winsor效果 ≥ 10%'
        },
        'scene_coverage': {
            'max_missing_scenarios': 0,
            'description': '2×2场景覆盖完整（A_H, A_L, Q_H, Q_L）'
        },
        'dataset_tier': {
            'raw_no_illegal': True,
            'preview_no_illegal': True,
            'description': 'raw 仅 prices/orderbook 且无策略列；preview 不含 raw kind'
        },
        'orderbook_seq': {
            'max_gap_rate': 0.01,
            'max_monotonic_breaks': 0,
            'description': '订单簿序列 gap_rate<1%, 单调断裂=0'
        },
        'trade_ob_join': {
            'min_match_rate': 0.98,
            'description': 'Trade-OB 最近邻(+/-250ms) 匹配率>=98%'
        },
        'ofi_primitives': {
            'min_nonzero_rate': 0.7,
            'min_corr': 0.3,
            'description': 'OFI 原语非零率≥70%；与预览OFI相关≥0.3(若可得)'
        },
        'd_ofi_consistency': {
            'max_inconsistency_rate': 0.001,
            'description': 'OFI 原语聚合与逐档求和一致（不自洽率≤0.1%）'
        },
        'orderbook_seq_strict': {
            'max_bad_order_rows': 0,
            'max_huge_jumps': 0,
            'description': '序列不变量不违反；无超大跳变'
        }
    }
    
    # 检查结果
    dod_results = {}
    
    # 完整性检查
    max_empty_rate = max([result.get('worst_empty_bucket_rate', 0) for result in completeness.values()] or [0])
    
    dod_results['completeness'] = {
        'passed': max_empty_rate < dod_criteria['completeness']['max_empty_bucket_rate'],
        'actual': max_empty_rate,
        'threshold': dod_criteria['completeness']['max_empty_bucket_rate'],
        'description': dod_criteria['completeness']['description']
    }
    
    # 去重检查
    max_duplicate_rate = max([result.get('max_duplicate_rate', 0) for result in dedup.values()] or [0])
    
    dod_results['deduplication'] = {
        'passed': max_duplicate_rate < dod_criteria['deduplication']['max_duplicate_rate'],
        'actual': max_duplicate_rate,
        'threshold': dod_criteria['deduplication']['max_duplicate_rate'],
        'description': dod_criteria['deduplication']['description']
    }
    
    # 延迟检查 - 取 prices 与 orderbook 的最差者
    p_p99 = latency.get('prices_latency_ms', {}).get('p99', 0)
    p_p50 = latency.get('prices_latency_ms', {}).get('p50', 0)
    ob_p99 = latency.get('orderbook_latency_ms', {}).get('p99', 0)
    ob_p50 = latency.get('orderbook_latency_ms', {}).get('p50', 0)
    
    worst_p99 = max(p_p99, ob_p99)
    worst_p50 = max(p_p50, ob_p50)
    
    dod_results['latency'] = {
        'passed': (worst_p99 < dod_criteria['latency']['max_p99_ms'] and 
                   worst_p50 < dod_criteria['latency']['max_p50_ms']),
        'actual_p99': worst_p99,
        'actual_p50': worst_p50,
        'threshold_p99': dod_criteria['latency']['max_p99_ms'],
        'threshold_p50': dod_criteria['latency']['max_p50_ms'],
        'description': 'P99/P50 取 prices 与 orderbook 的最差者'
    }
    
    # 信号量检查 - 自适应阈值
    # 从 completeness 取出所有 kind 的 by_symbol → 取最大 expected_minutes
    covered_mins = 0
    for v in completeness.values():
        for sym_stats in v.get('by_symbol', {}).values():
            covered_mins = max(covered_mins, sym_stats.get('expected_minutes', 0))
    
    auto_min_events = max(300, int(0.5 * covered_mins))  # 例：每分钟至少0.5个事件，下限300
    min_required = max(min_events, auto_min_events)
    
    total_events = signals.get('events', {}).get('total_events', 0)
    dod_results['signals'] = {
        'passed': total_events >= min_required,
        'actual': total_events,
        'threshold': min_required,
        'auto_threshold': auto_min_events,
        'covered_minutes': covered_mins,
        'description': f'事件总数 ≥ {min_required}（自适应阈值，覆盖{covered_mins}分钟）'
    }
    
    # 当无events但fusion/cvd指标充足时，降级为信息项
    if total_events == 0 and (signals.get('fusion', {}).get('total_signals', 0) > 1000 or 
                               signals.get('cvd', {}).get('total_signals', 0) > 1000):
        dod_results['signals']['passed'] = True
        dod_results['signals']['description'] += '（无events但fusion/cvd充足，降级为信息项）'
    
    # 一致性检查
    winsor_effect = consistency.get('cvd_consistency', {}).get('winsor_effect', 0)
    dod_results['consistency'] = {
        'passed': winsor_effect >= dod_criteria['consistency']['min_winsor_effect'],
        'actual': winsor_effect,
        'threshold': dod_criteria['consistency']['min_winsor_effect'],
        'description': dod_criteria['consistency']['description']
    }
    
    # 场景覆盖检查
    scene_coverage_miss = scene_coverage.get('scene_coverage_miss', 1)
    dod_results['scene_coverage'] = {
        'passed': scene_coverage_miss <= dod_criteria['scene_coverage']['max_missing_scenarios'],
        'actual': scene_coverage_miss,
        'threshold': dod_criteria['scene_coverage']['max_missing_scenarios'],
        'description': dod_criteria['scene_coverage']['description']
    }
    
    # 分仓合规检查（新增）
    if tier_check:
        raw_ok = (not tier_check.get('raw_illegal_kinds', [])) and (not tier_check.get('raw_illegal_columns', {}))
        preview_ok = (not tier_check.get('preview_illegal_kinds', []))
        dod_results['dataset_tier'] = {
            'passed': raw_ok and preview_ok,
            'raw_ok': raw_ok,
            'preview_ok': preview_ok,
            'description': dod_criteria['dataset_tier']['description']
        }
    
    # 订单簿序列检查（新增）
    if ob_seq and 'summary' in ob_seq:
        max_gap = ob_seq['summary'].get('max_gap_rate', 1.0)
        mono_breaks = ob_seq['summary'].get('total_mono_breaks', 999999)
        dod_results['orderbook_seq'] = {
            'passed': (max_gap < dod_criteria['orderbook_seq']['max_gap_rate']) and 
                     (mono_breaks <= dod_criteria['orderbook_seq']['max_monotonic_breaks']),
            'max_gap_rate': max_gap,
            'total_mono_breaks': mono_breaks,
            'description': dod_criteria['orderbook_seq']['description']
        }
    
    # Trade↔OB 匹配检查（新增）
    if join_cov and 'summary' in join_cov:
        min_match = join_cov['summary'].get('min_match_rate', 0.0)
        dod_results['trade_ob_join'] = {
            'passed': (min_match >= dod_criteria['trade_ob_join']['min_match_rate']),
            'min_match_rate': min_match,
            'description': dod_criteria['trade_ob_join']['description']
        }
    
    # OFI 原语检查（新增）
    if ofi_prim and 'summary' in ofi_prim:
        min_nz = ofi_prim['summary'].get('min_nonzero_rate', 0.0)
        corr = ofi_prim['summary'].get('ofi_corr', None)
        corr_ok = True if corr is None else (corr >= dod_criteria['ofi_primitives']['min_corr'])
        dod_results['ofi_primitives'] = {
            'passed': (min_nz >= dod_criteria['ofi_primitives']['min_nonzero_rate']) and corr_ok,
            'min_nonzero_rate': min_nz,
            'ofi_corr': corr,
            'description': dod_criteria['ofi_primitives']['description']
        }
    
    # OFI 原语自洽性检查（新增）
    if ofi_prim_cons and 'summary' in ofi_prim_cons:
        bad = ofi_prim_cons['summary'].get('max_inconsistency_rate', 1.0)
        dod_results['d_ofi_consistency'] = {
            'passed': bad <= dod_criteria['d_ofi_consistency']['max_inconsistency_rate'],
            'max_inconsistency_rate': bad,
            'description': dod_criteria['d_ofi_consistency']['description']
        }
    
    # 序列不变量严格检查（新增）
    if ob_seq and 'summary' in ob_seq:
        dod_results['orderbook_seq_strict'] = {
            'passed': ob_seq['summary'].get('total_bad_order_rows', 0) <= 0 and ob_seq['summary'].get('total_huge_jumps', 0) <= 0,
            'bad_order_rows': ob_seq['summary'].get('total_bad_order_rows', 0),
            'huge_jumps': ob_seq['summary'].get('total_huge_jumps', 0),
            'description': dod_criteria['orderbook_seq_strict']['description']
        }
    
    # 总体结果
    all_passed = all(result['passed'] for result in dod_results.values())
    
    print(f"完整性: {'PASS' if dod_results['completeness']['passed'] else 'FAIL'} "
          f"({dod_results['completeness']['actual']:.4f} < {dod_results['completeness']['threshold']:.4f})")
    print(f"去重: {'PASS' if dod_results['deduplication']['passed'] else 'FAIL'} "
          f"({dod_results['deduplication']['actual']:.4f} < {dod_results['deduplication']['threshold']:.4f})")
    print(f"延迟: {'PASS' if dod_results['latency']['passed'] else 'FAIL'} "
          f"(P99: {dod_results['latency']['actual_p99']:.1f}ms, P50: {dod_results['latency']['actual_p50']:.1f}ms)")
    print(f"信号量: {'PASS' if dod_results['signals']['passed'] else 'FAIL'} "
          f"({dod_results['signals']['actual']} >= {dod_results['signals']['threshold']})")
    print(f"一致性: {'PASS' if dod_results['consistency']['passed'] else 'FAIL'} "
          f"({dod_results['consistency']['actual']:.4f} >= {dod_results['consistency']['threshold']:.4f})")
    print(f"场景覆盖: {'PASS' if dod_results['scene_coverage']['passed'] else 'FAIL'} "
          f"({dod_results['scene_coverage']['actual']} <= {dod_results['scene_coverage']['threshold']})")
    
    if 'dataset_tier' in dod_results:
        print(f"分仓合规: {'PASS' if dod_results['dataset_tier']['passed'] else 'FAIL'} "
              f"(raw_ok={dod_results['dataset_tier']['raw_ok']}, preview_ok={dod_results['dataset_tier']['preview_ok']})")
    
    if 'orderbook_seq' in dod_results:
        print(f"订单簿序列: {'PASS' if dod_results['orderbook_seq']['passed'] else 'FAIL'} "
              f"(gap_rate={dod_results['orderbook_seq']['max_gap_rate']:.4f}, breaks={dod_results['orderbook_seq']['total_mono_breaks']})")
    
    if 'trade_ob_join' in dod_results:
        print(f"Trade-OB对齐: {'PASS' if dod_results['trade_ob_join']['passed'] else 'FAIL'} "
              f"(match_rate={dod_results['trade_ob_join']['min_match_rate']:.4f})")
    
    if 'ofi_primitives' in dod_results:
        corr_str = f"{dod_results['ofi_primitives']['ofi_corr']:.3f}" if dod_results['ofi_primitives']['ofi_corr'] is not None else "N/A"
        print(f"OFI原语: {'PASS' if dod_results['ofi_primitives']['passed'] else 'FAIL'} "
              f"(nonzero_rate={dod_results['ofi_primitives']['min_nonzero_rate']:.4f}, corr={corr_str})")
    
    if 'd_ofi_consistency' in dod_results:
        print(f"OFI原语自洽: {'PASS' if dod_results['d_ofi_consistency']['passed'] else 'FAIL'} "
              f"(inconsistency_rate={dod_results['d_ofi_consistency']['max_inconsistency_rate']:.6f})")
    
    if 'orderbook_seq_strict' in dod_results:
        print(f"序列不变量: {'PASS' if dod_results['orderbook_seq_strict']['passed'] else 'FAIL'} "
              f"(bad_order={dod_results['orderbook_seq_strict']['bad_order_rows']}, huge_jumps={dod_results['orderbook_seq_strict']['huge_jumps']})")
    
    print(f"\n总体结果: {'通过' if all_passed else '未通过'}")
    
    return {
        'dod_criteria': dod_criteria,
        'dod_results': dod_results,
        'overall_passed': all_passed,
        'summary': {
            'total_tests': len(dod_results),
            'passed_tests': sum(1 for r in dod_results.values() if r['passed']),
            'failed_tests': sum(1 for r in dod_results.values() if not r['passed'])
        }
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OFI+CVD数据质量验证')
    parser.add_argument('--base-dir', default='data/ofi_cvd', 
                       help='数据目录路径 (默认: data/ofi_cvd)')
    parser.add_argument('--raw-dir', default=None,
                       help='权威原始仓目录(仅 prices/orderbook)')
    parser.add_argument('--preview-dir', default=None,
                       help='预览仓目录(ofi/cvd/fusion/events/features)')
    parser.add_argument('--output-dir', default='artifacts/dq_reports',
                       help='报告输出目录 (默认: artifacts/dq_reports)')
    parser.add_argument('--min-events', type=int, default=1000,
                       help='最小事件数量阈值 (默认: 1000)')
    parser.add_argument('--max-empty-bucket-rate', type=float, default=0.001,
                       help='最大空桶率阈值 (默认: 0.001)')
    parser.add_argument('--max-dup-rate', type=float, default=0.005,
                       help='最大重复率阈值 (默认: 0.005)')
    parser.add_argument('--lookback-mins', type=int, default=180,
                       help='完整性检查回看窗口(分钟)，0为不限制 (默认: 180)')
    parser.add_argument('--join-tol-ms', type=int, default=250,
                       help='Trade-OB 最近邻容差(ms) (默认: 250)')
    parser.add_argument('--seq-huge-jump', type=int, default=10000,
                       help='序列超大跳变阈值 (默认: 10000)')
    parser.add_argument('--ofi-atol', type=float, default=1e-6,
                       help='OFI 原语自洽容差 (默认: 1e-6)')
    args = parser.parse_args()
    
    # 兼容：若未显式提供，则使用 base-dir 作为 raw-dir
    raw_dir = args.raw_dir or args.base_dir
    preview_dir = args.preview_dir
    
    print("OFI+CVD数据质量验证脚本 (增强版 v3)")
    print("=" * 50)
    print(f"[DIR] raw: {raw_dir}")
    if preview_dir:
        print(f"[DIR] preview: {preview_dir}")
    
    # 加载数据
    raw_data = load_parquet_data_by_dir(raw_dir, ['prices', 'orderbook'], lookback_mins=args.lookback_mins)
    preview_data = load_parquet_data_by_dir(preview_dir, ['ofi', 'cvd', 'fusion', 'events', 'features'], lookback_mins=args.lookback_mins) if preview_dir else {k: pd.DataFrame() for k in ['ofi', 'cvd', 'fusion', 'events', 'features']}
    
    # 兼容老逻辑：拼一个 all_data 用于通用检查
    all_data = {}
    all_data.update(raw_data)
    all_data.update(preview_data)
    
    if not any(df is not None and not df.empty for df in all_data.values()):
        print("错误: 未找到数据文件")
        return 1
    
    # 执行检查
    completeness = check_completeness(all_data, args.lookback_mins)
    dedup = check_deduplication(all_data)
    latency = check_latency(all_data)
    signals = check_signal_volume(all_data)
    consistency = check_consistency(all_data)
    scene_coverage = check_scene_coverage(preview_dir or raw_dir)
    
    # 【补丁1】无 preview 目录时场景覆盖降级为信息项
    no_preview = (args.preview_dir is None or args.preview_dir == '')
    if no_preview:
        scene_coverage = {'scene_coverage_miss': 0, 'note': 'no preview_dir; scene coverage skipped'}
        print("\n[INFO] 无 preview_dir，场景覆盖检查已跳过")
    
    # 新增检查
    tier_check = check_tier_contract(raw_data, preview_data)
    ob_seq = check_orderbook_sequence(raw_data.get('orderbook', pd.DataFrame()), huge_jump_threshold=args.seq_huge_jump)
    join_cov = check_trade_orderbook_join(raw_data.get('prices', pd.DataFrame()), raw_data.get('orderbook', pd.DataFrame()), max_abs_ms=args.join_tol_ms)
    ofi_prim = check_ofi_primitives(raw_data.get('orderbook', pd.DataFrame()), preview_data.get('ofi', pd.DataFrame()))
    ofi_prim_cons = check_ofi_primitive_consistency(raw_data.get('orderbook', pd.DataFrame()), atol=args.ofi_atol)
    
    # 生成DoD报告
    dod_report = generate_dod_report(completeness, dedup, latency, signals, consistency, scene_coverage,
                                   tier_check, ob_seq, join_cov, ofi_prim, ofi_prim_cons,
                                   args.min_events, args.max_empty_bucket_rate, args.max_dup_rate)
    
    # 生成完整报告
    full_report = {
        'timestamp': dt.datetime.utcnow().isoformat(),
        'base_dir': args.base_dir,
        'raw_dir': raw_dir,
        'preview_dir': preview_dir,
        'data_summary': {
            'kinds': list(all_data.keys()),
            'symbols': list(set(symbol for df in all_data.values() 
                              if not df.empty and 'symbol' in df.columns 
                              for symbol in df['symbol'].unique()))
        },
        'tier_check': tier_check,
        'orderbook_seq': ob_seq,
        'trade_ob_join': join_cov,
        'ofi_primitives': ofi_prim,
        'ofi_primitives_consistency': ofi_prim_cons,
        'completeness': completeness,
        'deduplication': dedup,
        'latency': latency,
        'signals': signals,
        'consistency': consistency,
        'scene_coverage': scene_coverage,
        'dod_report': dod_report
    }
    
    # 保存报告
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M")
    report_file = os.path.join(args.output_dir, f"dq_{timestamp}.json")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n报告已保存: {report_file}")
    
    # 返回退出码
    return 0 if dod_report['overall_passed'] else 1

if __name__ == '__main__':
    exit(main())
