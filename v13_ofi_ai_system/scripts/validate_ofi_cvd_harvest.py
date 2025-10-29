#!/usr/bin/env python3
"""
OFI+CVD数据质量验证脚本 (Task 1.3.1 v2)

验证采集数据的质量，生成DoD验收报告：
- 完整性检查（空桶率）
- 去重检查（重复率）
- 延迟统计（p50/p90/p99）
- 信号量统计
- 一致性检查
"""

import json
import pandas as pd
import numpy as np
import glob
import os
import datetime as dt
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse

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
            if 'event_ts_ms' in df.columns:
                df['minute'] = pd.to_datetime(df['event_ts_ms'], unit='ms').dt.floor('min')
            else:
                df['minute'] = pd.to_datetime(df['ts_ms'], unit='ms').dt.floor('min')
        except:
            # 如果时间戳解析失败，使用当前时间
            df['minute'] = pd.Timestamp.now().floor('min')
        
        # 应用lookback窗口限制（避免离线期误伤）
        if lookback_mins > 0:
            cutoff = pd.Timestamp.utcnow().floor('min') - pd.Timedelta(minutes=lookback_mins)
            # 确保时区一致性
            if df['minute'].dt.tz is None:
                cutoff = cutoff.tz_localize(None)
            elif cutoff.tz is None:
                cutoff = cutoff.tz_localize('UTC')
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
            total_minutes = g['minute'].nunique()
            min_time = g['minute'].min()
            max_time = g['minute'].max()
            expected_minutes = int((max_time - min_time).total_seconds() / 60) + 1 if min_time != max_time else 1
            empty_bucket_rate = 1 - (total_minutes / expected_minutes) if expected_minutes > 0 else 1.0
            
            by_sym[str(sym)] = {
                'empty_bucket_rate': float(empty_bucket_rate),
                'total_minutes': int(total_minutes),
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
        
        max_duplicate_rate = max((v['duplicate_rate'] for v in dup_by_sym.values()), default=0.0)
        
        results[kind] = {
            'by_symbol': dup_by_sym,
            'max_duplicate_rate': float(max_duplicate_rate)
        }
        
        print(f"{kind}: 最大重复率 {max_duplicate_rate:.4f} (按symbol统计)")
        for sym, stats in dup_by_sym.items():
            print(f"  {sym}: {stats['duplicate_rate']:.4f} ({stats['unique_rows']}/{stats['total_rows']})")
    
    return results

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
    if 'prices' in data and not data['prices'].empty and 'latency_ms' in data['prices'].columns:
        s = data['prices']['latency_ms'].dropna()
        results['prices_latency_ms'] = _lat_stats(s)
        print(f"Prices 延迟(ms): P50={results['prices_latency_ms']['p50']:.2f}, P90={results['prices_latency_ms']['p90']:.2f}, P99={results['prices_latency_ms']['p99']:.2f}")
    else:
        results['prices_latency_ms'] = _lat_stats([])
        print("Prices: 无延迟数据")
    
    # 检查orderbook数据的延迟
    if 'orderbook' in data and not data['orderbook'].empty and 'latency_ms' in data['orderbook'].columns:
        s = data['orderbook']['latency_ms'].dropna()
        results['orderbook_latency_ms'] = _lat_stats(s)
        print(f"Orderbook 延迟(ms): P50={results['orderbook_latency_ms']['p50']:.2f}, P90={results['orderbook_latency_ms']['p90']:.2f}, P99={results['orderbook_latency_ms']['p99']:.2f}")
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

def generate_dod_report(completeness: Dict, dedup: Dict, latency: Dict, 
                       signals: Dict, consistency: Dict, scene_coverage: Dict, 
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
        }
    }
    
    # 检查结果
    dod_results = {}
    
    # 完整性检查
    max_empty_rate = max((result.get('worst_empty_bucket_rate', 0) for result in completeness.values()), default=0)
    
    dod_results['completeness'] = {
        'passed': max_empty_rate < dod_criteria['completeness']['max_empty_bucket_rate'],
        'actual': max_empty_rate,
        'threshold': dod_criteria['completeness']['max_empty_bucket_rate'],
        'description': dod_criteria['completeness']['description']
    }
    
    # 去重检查
    max_duplicate_rate = max((result.get('max_duplicate_rate', 0) for result in dedup.values()), default=0)
    
    dod_results['deduplication'] = {
        'passed': max_duplicate_rate < dod_criteria['deduplication']['max_duplicate_rate'],
        'actual': max_duplicate_rate,
        'threshold': dod_criteria['deduplication']['max_duplicate_rate'],
        'description': dod_criteria['deduplication']['description']
    }
    
    # 延迟检查
    latency_p99 = latency.get('latency_ms', {}).get('p99', 0)
    latency_p50 = latency.get('latency_ms', {}).get('p50', 0)
    
    dod_results['latency'] = {
        'passed': (latency_p99 < dod_criteria['latency']['max_p99_ms'] and 
                  latency_p50 < dod_criteria['latency']['max_p50_ms']),
        'actual_p99': latency_p99,
        'actual_p50': latency_p50,
        'threshold_p99': dod_criteria['latency']['max_p99_ms'],
        'threshold_p50': dod_criteria['latency']['max_p50_ms'],
        'description': dod_criteria['latency']['description']
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
    args = parser.parse_args()
    
    print("OFI+CVD数据质量验证脚本")
    print("=" * 50)
    
    # 加载数据
    print(f"加载数据从: {args.base_dir}")
    data = load_parquet_data(args.base_dir)
    
    if not data:
        print("错误: 未找到数据文件")
        return 1
    
    # 执行检查
    completeness = check_completeness(data, args.lookback_mins)
    dedup = check_deduplication(data)
    latency = check_latency(data)
    signals = check_signal_volume(data)
    consistency = check_consistency(data)
    scene_coverage = check_scene_coverage(args.base_dir)
    
    # 生成DoD报告
    dod_report = generate_dod_report(completeness, dedup, latency, signals, consistency, scene_coverage,
                                   args.min_events, args.max_empty_bucket_rate, args.max_dup_rate)
    
    # 生成完整报告
    full_report = {
        'timestamp': dt.datetime.utcnow().isoformat(),
        'base_dir': args.base_dir,
        'data_summary': {
            'kinds': list(data.keys()),
            'total_files': sum(len(glob.glob(f"{args.base_dir}/date=*/symbol=*/kind={kind}/*.parquet")) 
                             for kind in data.keys()),
            'symbols': list(set(symbol for df in data.values() 
                              if not df.empty and 'symbol' in df.columns 
                              for symbol in df['symbol'].unique()))
        },
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
