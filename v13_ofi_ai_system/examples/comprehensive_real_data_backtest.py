"""
综合真实数据背离检测回测脚本

整合多个历史数据源进行背离检测回测：
- V13 CVD数据（BTC/ETH，多个时间点）
- V12 OFI数据
- 归档的历史测试数据

验证背离检测在真实市场数据上的有效性

Author: V13 OFI+CVD AI System
Created: 2025-01-20
"""

import sys
import os
import time
import json
import csv
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
import glob

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ofi_cvd_divergence import DivergenceConfig, DivergenceDetector, DivergenceType


class ComprehensiveRealDataBacktester:
    """综合真实数据回测器"""
    
    def __init__(self, config: DivergenceConfig):
        self.config = config
        self.detector = DivergenceDetector(config)
        self.events = []
        self.data_sources = []
        
    def discover_data_sources(self) -> List[Dict[str, Any]]:
        """发现所有可用的数据源"""
        sources = []
        
        # V13 当前数据
        current_data_dir = "data"
        if os.path.exists(current_data_dir):
            for root, dirs, files in os.walk(current_data_dir):
                for file in files:
                    if file.endswith('.parquet'):
                        full_path = os.path.join(root, file)
                        sources.append({
                            'path': full_path,
                            'type': 'v13_cvd',
                            'symbol': self._extract_symbol_from_path(file),
                            'size': os.path.getsize(full_path)
                        })
        
        # 归档的V13数据
        archive_v13_dir = "../archive/v13_test_data_history"
        if os.path.exists(archive_v13_dir):
            for root, dirs, files in os.walk(archive_v13_dir):
                for file in files:
                    if file.endswith('.parquet'):
                        full_path = os.path.join(root, file)
                        sources.append({
                            'path': full_path,
                            'type': 'v13_archive',
                            'symbol': self._extract_symbol_from_path(file),
                            'size': os.path.getsize(full_path)
                        })
        
        # V12 OFI数据
        v12_data_dir = "../archive/test_data"
        if os.path.exists(v12_data_dir):
            for file in os.listdir(v12_data_dir):
                if file.startswith('v12_real_ofi_data_ofi_'):
                    full_path = os.path.join(v12_data_dir, file)
                    sources.append({
                        'path': full_path,
                        'type': 'v12_ofi',
                        'symbol': 'OFI_DATA',
                        'size': os.path.getsize(full_path)
                    })
        
        # 按大小排序，优先使用较大的数据集
        sources.sort(key=lambda x: x['size'], reverse=True)
        return sources
    
    def _extract_symbol_from_path(self, filename: str) -> str:
        """从文件名提取交易对符号"""
        if 'btcusdt' in filename.lower():
            return 'BTCUSDT'
        elif 'ethusdt' in filename.lower():
            return 'ETHUSDT'
        else:
            return 'UNKNOWN'
    
    def load_data_source(self, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """加载单个数据源"""
        try:
            print(f"加载数据源: {source['path']}")
            
            if source['type'] in ['v13_cvd', 'v13_archive']:
                return self._load_cvd_data(source)
            elif source['type'] == 'v12_ofi':
                return self._load_ofi_data(source)
            else:
                print(f"未知数据类型: {source['type']}")
                return None
                
        except Exception as e:
            print(f"加载数据源失败 {source['path']}: {e}")
            return None
    
    def _load_cvd_data(self, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """加载CVD数据"""
        df = pd.read_parquet(source['path'])
        
        if len(df) < 50:  # 数据太少
            print(f"数据量太少: {len(df)} 条记录")
            return None
        
        # 检查必需的列
        required_cols = ['timestamp', 'price', 'z_cvd']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"缺少必需的列: {missing_cols}")
            return None
        
        # 处理NaN值 - 只保留有效数据
        valid_mask = df['z_cvd'].notna()
        if valid_mask.sum() < 50:  # 有效数据太少
            print(f"有效数据太少: {valid_mask.sum()} 条记录")
            return None
        
        # 过滤有效数据
        df_valid = df[valid_mask].copy()
        timestamps = df_valid['timestamp'].values
        prices = df_valid['price'].values
        z_cvd = df_valid['z_cvd'].values
        
        # CVD数据源：只使用真实CVD，不合成OFI
        z_ofi = None  # 标记为无OFI数据
        
        # 生成融合分数（仅当两个指标都存在时）
        if z_ofi is not None and z_cvd is not None:
            fusion_scores = (z_cvd + z_ofi) / 2
            fusion_scores = np.clip(fusion_scores, -5, 5)
        else:
            fusion_scores = None  # 无融合分数
        
        # 生成一致性分数（仅当两个指标都存在时）
        if z_ofi is not None and z_cvd is not None:
            consistency = np.abs(z_cvd * z_ofi) / (
                np.abs(z_cvd) + np.abs(z_ofi) + 1e-8
            )
        else:
            consistency = None  # 无一致性分数
        
        return {
            'symbol': source['symbol'],
            'type': source['type'],
            'timestamps': timestamps,
            'prices': prices,
            'z_ofi': z_ofi,
            'z_cvd': z_cvd,
            'fusion_scores': fusion_scores,
            'consistency': consistency,
            'original_data': df_valid
        }
    
    def _load_ofi_data(self, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """加载OFI数据"""
        df = pd.read_csv(source['path'])
        
        if len(df) < 50:  # 数据太少
            print(f"数据量太少: {len(df)} 条记录")
            return None
        
        # 检查必需的列
        required_cols = ['timestamp', 'ofi', 'mid_price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"缺少必需的列: {missing_cols}")
            return None
        
        # 计算OFI Z-score
        ofi_values = df['ofi'].values
        ofi_mean = np.mean(ofi_values)
        ofi_std = np.std(ofi_values)
        z_ofi = (ofi_values - ofi_mean) / (ofi_std + 1e-8)
        z_ofi = np.clip(z_ofi, -5, 5)
        
        # OFI数据源：只使用真实OFI，不合成CVD
        z_cvd = None  # 标记为无CVD数据
        
        # 生成融合分数（仅当两个指标都存在时）
        if z_ofi is not None and z_cvd is not None:
            fusion_scores = (z_ofi + z_cvd) / 2
            fusion_scores = np.clip(fusion_scores, -5, 5)
        else:
            fusion_scores = None  # 无融合分数
        
        # 生成一致性分数（仅当两个指标都存在时）
        if z_ofi is not None and z_cvd is not None:
            consistency = np.abs(z_ofi * z_cvd) / (np.abs(z_ofi) + np.abs(z_cvd) + 1e-8)
        else:
            consistency = None  # 无一致性分数
        
        return {
            'symbol': source['symbol'],
            'type': source['type'],
            'timestamps': df['timestamp'].values,
            'prices': df['mid_price'].values,
            'z_ofi': z_ofi,
            'z_cvd': z_cvd,
            'fusion_scores': fusion_scores,
            'consistency': consistency,
            'original_data': df
        }
    
    def run_comprehensive_backtest(self) -> Dict[str, Any]:
        """运行综合回测"""
        print("=== 综合真实数据背离检测回测 ===\n")
        
        # 发现数据源
        sources = self.discover_data_sources()
        print(f"发现 {len(sources)} 个数据源")
        
        if not sources:
            print("没有找到可用的数据源")
            return {}
        
        # 显示数据源信息
        print("\n数据源列表:")
        for i, source in enumerate(sources[:10]):  # 只显示前10个
            print(f"{i+1:2d}. {source['type']:12s} {source['symbol']:8s} {source['size']:8d} bytes - {source['path']}")
        
        if len(sources) > 10:
            print(f"... 还有 {len(sources) - 10} 个数据源")
        
        # 按数据源类型分桶
        buckets = {
            'cvd_only': [],      # 只有CVD的数据源
            'ofi_only': [],      # 只有OFI的数据源  
            'mixed': []          # 同时有OFI和CVD的数据源
        }
        
        for source in sources:
            if 'cvd' in source['type'].lower() and 'ofi' not in source['type'].lower():
                buckets['cvd_only'].append(source)
            elif 'ofi' in source['type'].lower() and 'cvd' not in source['type'].lower():
                buckets['ofi_only'].append(source)
            else:
                buckets['mixed'].append(source)
        
        print(f"\n数据源分桶结果:")
        print(f"  CVD专用: {len(buckets['cvd_only'])} 个")
        print(f"  OFI专用: {len(buckets['ofi_only'])} 个") 
        print(f"  混合数据: {len(buckets['mixed'])} 个")
        
        # 选择每个桶的代表性数据源
        selected_sources = []
        for bucket_name, bucket_sources in buckets.items():
            if bucket_sources:
                # 选择最大的数据源
                selected = max(bucket_sources, key=lambda x: x['size'])
                selected_sources.append(selected)
                print(f"  选择 {bucket_name}: {selected['symbol']} ({selected['size']} bytes)")
        
        all_results = {}
        bucket_results = {bucket: [] for bucket in buckets.keys()}
        total_events = 0
        total_samples = 0
        
        for i, source in enumerate(selected_sources):
            print(f"\n--- 处理数据源 {i+1}/{len(selected_sources)}: {source['symbol']} ---")
            
            # 加载数据
            data = self.load_data_source(source)
            if data is None:
                continue
            
            # 运行回测
            result = self._run_single_backtest(data, source)
            if result:
                all_results[source['symbol']] = result
                total_events += result['total_events']
                total_samples += result['total_samples']
                
                # 按桶分类结果
                if 'cvd' in source['type'].lower() and 'ofi' not in source['type'].lower():
                    bucket_results['cvd_only'].append(result)
                elif 'ofi' in source['type'].lower() and 'cvd' not in source['type'].lower():
                    bucket_results['ofi_only'].append(result)
                else:
                    bucket_results['mixed'].append(result)
        
        # 按桶计算统计结果
        bucket_stats = {}
        for bucket_name, results in bucket_results.items():
            if not results:
                bucket_stats[bucket_name] = {
                    'events': 0, 'samples': 0, 'accuracy_10': 0.0, 'accuracy_20': 0.0,
                    'avg_return_10': 0.0, 'avg_return_20': 0.0, 'dod_passed': False
                }
                continue
                
            bucket_events = sum(r.get('total_events', 0) for r in results)
            bucket_samples = sum(r.get('total_samples', 0) for r in results)
            bucket_correct_10 = sum(r.get('correct_predictions_10', 0) for r in results)
            bucket_correct_20 = sum(r.get('correct_predictions_20', 0) for r in results)
            
            bucket_returns_10 = []
            bucket_returns_20 = []
            for r in results:
                bucket_returns_10.extend(r.get('total_returns_10', []))
                bucket_returns_20.extend(r.get('total_returns_20', []))
            
            accuracy_10 = bucket_correct_10 / bucket_events if bucket_events > 0 else 0.0
            accuracy_20 = bucket_correct_20 / bucket_events if bucket_events > 0 else 0.0
            avg_return_10 = np.mean(bucket_returns_10) if bucket_returns_10 else 0.0
            avg_return_20 = np.mean(bucket_returns_20) if bucket_returns_20 else 0.0
            
            # DoD判定：accuracy@10 >= 55% 或 accuracy@20 >= 55%
            dod_passed = accuracy_10 >= 0.55 or accuracy_20 >= 0.55
            
            bucket_stats[bucket_name] = {
                'events': bucket_events,
                'samples': bucket_samples,
                'accuracy_10': accuracy_10,
                'accuracy_20': accuracy_20,
                'avg_return_10': avg_return_10,
                'avg_return_20': avg_return_20,
                'dod_passed': dod_passed
            }
        
        # 汇总结果
        summary = {
            'total_sources': len(selected_sources),
            'successful_sources': len(all_results),
            'total_events': total_events,
            'total_samples': total_samples,
            'results_by_symbol': all_results,
            'bucket_results': bucket_results,
            'bucket_stats': bucket_stats,
            'overall_accuracy_10': 0.0,
            'overall_accuracy_20': 0.0,
            'overall_avg_return_10': 0.0,
            'overall_avg_return_20': 0.0,
            'overall_dod_passed': False
        }
        
        # 计算总体准确率（仅方向性背离，排除ofi_cvd_conflict）
        if total_events > 0:
            total_correct_10 = sum(r.get('correct_predictions_10', 0) for r in all_results.values())
            total_correct_20 = sum(r.get('correct_predictions_20', 0) for r in all_results.values())
            total_returns_10 = []
            total_returns_20 = []
            for r in all_results.values():
                total_returns_10.extend(r.get('total_returns_10', []))
                total_returns_20.extend(r.get('total_returns_20', []))
            
            summary['overall_accuracy_10'] = total_correct_10 / total_events if total_events > 0 else 0.0
            summary['overall_accuracy_20'] = total_correct_20 / total_events if total_events > 0 else 0.0
            summary['overall_avg_return_10'] = np.mean(total_returns_10) if total_returns_10 else 0.0
            summary['overall_avg_return_20'] = np.mean(total_returns_20) if total_returns_20 else 0.0
            summary['overall_dod_passed'] = summary['overall_accuracy_10'] >= 0.55 or summary['overall_accuracy_20'] >= 0.55
        
        return summary
    
    def _run_single_backtest(self, data: Dict[str, Any], source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """运行单个数据源的回测"""
        timestamps = data['timestamps']
        prices = data['prices']
        z_ofi = data['z_ofi']
        z_cvd = data['z_cvd']
        fusion_scores = data['fusion_scores']
        consistency = data['consistency']
        
        print(f"数据点数量: {len(timestamps)}")
        print(f"价格范围: {prices.min():.2f} - {prices.max():.2f}")
        if z_ofi is not None:
            print(f"OFI Z范围: {z_ofi.min():.2f} - {z_ofi.max():.2f}")
        else:
            print("OFI Z范围: 无数据")
        if z_cvd is not None:
            print(f"CVD Z范围: {z_cvd.min():.2f} - {z_cvd.max():.2f}")
        else:
            print("CVD Z范围: 无数据")
        
        # 重置检测器
        self.detector.reset()
        
        events = []
        detection_times = []
        correct_predictions_10 = 0
        correct_predictions_20 = 0
        returns_10 = []
        returns_20 = []
        
        # 处理None值，创建默认数组
        z_ofi_array = z_ofi if z_ofi is not None else np.zeros(len(timestamps))
        z_cvd_array = z_cvd if z_cvd is not None else np.zeros(len(timestamps))
        fusion_array = fusion_scores if fusion_scores is not None else np.zeros(len(timestamps))
        consistency_array = consistency if consistency is not None else np.zeros(len(timestamps))
        
        for i, (ts, price, ofi, cvd, fusion, cons) in enumerate(zip(
            timestamps, prices, z_ofi_array, z_cvd_array, fusion_array, consistency_array
        )):
            # 处理None值
            ofi_val = ofi if ofi is not None else 0.0
            cvd_val = cvd if cvd is not None else 0.0
            fusion_val = fusion if fusion is not None else None
            cons_val = cons if cons is not None else None
            
            # 检测背离
            start_time = time.perf_counter()
            event = self.detector.update(
                ts=ts,
                price=price,
                z_ofi=ofi_val,
                z_cvd=cvd_val,
                fusion_score=fusion_val,
                consistency=cons_val,
                warmup=False,
                lag_sec=0.0
            )
            detection_time = time.perf_counter() - start_time
            detection_times.append(detection_time)
            
            if event and event.get('type'):
                events.append({
                    'index': i,
                    'timestamp': ts,
                    'price': price,
                    'event': event
                })
                
                # 只统计方向性背离的准确率，排除冲突事件
                if event['type'] not in ['ofi_cvd_conflict']:
                    # 计算未来收益
                    if i + 10 < len(prices):
                        future_price_10 = prices[i + 10]
                        return_10 = (future_price_10 - price) / price
                        returns_10.append(return_10)
                        
                        # 检查预测是否正确
                        if event['type'] in ['bull_div', 'hidden_bull'] and return_10 > 0:
                            correct_predictions_10 += 1
                        elif event['type'] in ['bear_div', 'hidden_bear'] and return_10 < 0:
                            correct_predictions_10 += 1
                    
                    if i + 20 < len(prices):
                        future_price_20 = prices[i + 20]
                        return_20 = (future_price_20 - price) / price
                        returns_20.append(return_20)
                        
                        # 检查预测是否正确
                        if event['type'] in ['bull_div', 'hidden_bull'] and return_20 > 0:
                            correct_predictions_20 += 1
                        elif event['type'] in ['bear_div', 'hidden_bear'] and return_20 < 0:
                            correct_predictions_20 += 1
        
        # 只统计方向性背离事件
        directional_events = [e for e in events if e['event']['type'] not in ['ofi_cvd_conflict']]
        
        # 计算结果
        accuracy_10 = correct_predictions_10 / len(directional_events) if directional_events else 0.0
        accuracy_20 = correct_predictions_20 / len(directional_events) if directional_events else 0.0
        avg_return_10 = np.mean(returns_10) if returns_10 else 0.0
        avg_return_20 = np.mean(returns_20) if returns_20 else 0.0
        avg_detection_time = np.mean(detection_times) if detection_times else 0.0
        p95_detection_time = np.percentile(detection_times, 95) if detection_times else 0.0
        
        # 按类型统计事件
        type_counts = {}
        for event_info in events:
            event_type = event_info['event']['type']
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        result = {
            'symbol': source['symbol'],
            'type': source['type'],
            'total_events': len(events),
            'directional_events': len(directional_events),
            'conflict_events': len(events) - len(directional_events),
            'total_samples': len(timestamps),
            'accuracy_10': accuracy_10,
            'accuracy_20': accuracy_20,
            'avg_return_10': avg_return_10,
            'avg_return_20': avg_return_20,
            'avg_detection_time': avg_detection_time,
            'p95_detection_time': p95_detection_time,
            'events_by_type': type_counts,
            'correct_predictions_10': correct_predictions_10,
            'correct_predictions_20': correct_predictions_20,
            'total_returns_10': returns_10,
            'total_returns_20': returns_20
        }
        
        print(f"检测到 {len(events)} 个背离事件")
        print(f"10期准确率: {accuracy_10:.2%}")
        print(f"20期准确率: {accuracy_20:.2%}")
        print(f"P95检测延迟: {p95_detection_time*1000:.3f}ms")
        
        return result
    
    def generate_comprehensive_report(self, results: Dict[str, Any], output_dir: str) -> str:
        """生成综合回测报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成CSV报告
        csv_file = os.path.join(output_dir, 'comprehensive_divergence_events.csv')
        self._export_comprehensive_events_to_csv(csv_file, results)
        
        # 生成JSON报告
        json_file = os.path.join(output_dir, 'comprehensive_backtest_results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成可视化
        plot_file = os.path.join(output_dir, 'comprehensive_divergence_analysis.png')
        self._create_comprehensive_visualization(plot_file, results)
        
        # 生成Markdown报告
        md_file = os.path.join(output_dir, 'comprehensive_backtest_report.md')
        self._generate_comprehensive_markdown_report(md_file, results)
        
        return md_file
    
    def _export_comprehensive_events_to_csv(self, csv_file: str, results: Dict[str, Any]):
        """导出综合事件到CSV"""
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'symbol', 'type', 'total_events', 'total_samples', 'accuracy_10', 'accuracy_20',
                'avg_return_10', 'avg_return_20', 'avg_detection_time', 'p95_detection_time'
            ])
            
            for symbol, result in results.get('results_by_symbol', {}).items():
                writer.writerow([
                    symbol,
                    result.get('type', ''),
                    result.get('total_events', 0),
                    result.get('total_samples', 0),
                    result.get('accuracy_10', 0.0),
                    result.get('accuracy_20', 0.0),
                    result.get('avg_return_10', 0.0),
                    result.get('avg_return_20', 0.0),
                    result.get('avg_detection_time', 0.0),
                    result.get('p95_detection_time', 0.0)
                ])
    
    def _create_comprehensive_visualization(self, plot_file: str, results: Dict[str, Any]):
        """创建综合可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 准确率对比
        ax1 = axes[0, 0]
        symbols = list(results.get('results_by_symbol', {}).keys())
        accuracies_10 = [results['results_by_symbol'][s].get('accuracy_10', 0) for s in symbols]
        accuracies_20 = [results['results_by_symbol'][s].get('accuracy_20', 0) for s in symbols]
        
        x = np.arange(len(symbols))
        width = 0.35
        ax1.bar(x - width/2, accuracies_10, width, label='10期准确率', alpha=0.8)
        ax1.bar(x + width/2, accuracies_20, width, label='20期准确率', alpha=0.8)
        ax1.set_xlabel('交易对')
        ax1.set_ylabel('准确率')
        ax1.set_title('各交易对背离检测准确率对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(symbols, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 事件数量分布
        ax2 = axes[0, 1]
        event_counts = [results['results_by_symbol'][s].get('total_events', 0) for s in symbols]
        ax2.bar(symbols, event_counts, color='skyblue', alpha=0.8)
        ax2.set_xlabel('交易对')
        ax2.set_ylabel('事件数量')
        ax2.set_title('各交易对背离事件数量')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 平均收益分布
        ax3 = axes[1, 0]
        avg_returns_10 = [results['results_by_symbol'][s].get('avg_return_10', 0) for s in symbols]
        avg_returns_20 = [results['results_by_symbol'][s].get('avg_return_20', 0) for s in symbols]
        
        x = np.arange(len(symbols))
        ax3.bar(x - width/2, avg_returns_10, width, label='10期平均收益', alpha=0.8)
        ax3.bar(x + width/2, avg_returns_20, width, label='20期平均收益', alpha=0.8)
        ax3.set_xlabel('交易对')
        ax3.set_ylabel('平均收益')
        ax3.set_title('各交易对平均收益对比')
        ax3.set_xticks(x)
        ax3.set_xticklabels(symbols, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 检测延迟分布
        ax4 = axes[1, 1]
        detection_times = [results['results_by_symbol'][s].get('p95_detection_time', 0) * 1000 for s in symbols]
        ax4.bar(symbols, detection_times, color='lightcoral', alpha=0.8)
        ax4.set_xlabel('交易对')
        ax4.set_ylabel('P95检测延迟 (ms)')
        ax4.set_title('各交易对检测延迟对比')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_comprehensive_markdown_report(self, md_file: str, results: Dict[str, Any]):
        """生成综合Markdown报告"""
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# 综合真实数据背离检测回测报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 回测配置\n\n")
            f.write(f"- 枢轴窗口长度: {self.config.swing_L}\n")
            f.write(f"- 高强度阈值: {self.config.z_hi}\n")
            f.write(f"- 中等强度阈值: {self.config.z_mid}\n")
            f.write(f"- 最小枢轴间距: {self.config.min_separation}\n")
            f.write(f"- 冷却时间: {self.config.cooldown_secs}秒\n")
            f.write(f"- 暖启动样本数: {self.config.warmup_min}\n")
            f.write(f"- 使用融合指标: {self.config.use_fusion}\n\n")
            
            f.write("## 数据源统计\n\n")
            f.write(f"- **总数据源数**: {results.get('total_sources', 0)}\n")
            f.write(f"- **成功处理数**: {results.get('successful_sources', 0)}\n")
            f.write(f"- **总样本数**: {results.get('total_samples', 0):,}\n")
            f.write(f"- **总事件数**: {results.get('total_events', 0)}\n\n")
            
            f.write("## 总体结果\n\n")
            f.write(f"- **10期准确率**: {results.get('overall_accuracy_10', 0):.2%}\n")
            f.write(f"- **20期准确率**: {results.get('overall_accuracy_20', 0):.2%}\n")
            f.write(f"- **10期平均收益**: {results.get('overall_avg_return_10', 0):.4f}\n")
            f.write(f"- **20期平均收益**: {results.get('overall_avg_return_20', 0):.4f}\n")
            f.write(f"- **DoD通过状态**: {'✅ 通过' if results.get('overall_dod_passed', False) else '❌ 未通过'}\n\n")
            
            f.write("## 分桶结果统计\n\n")
            bucket_stats = results.get('bucket_stats', {})
            for bucket_name, stats in bucket_stats.items():
                f.write(f"### {bucket_name.upper()} 数据源\n\n")
                f.write(f"- **事件数**: {stats.get('events', 0)}\n")
                f.write(f"- **样本数**: {stats.get('samples', 0):,}\n")
                f.write(f"- **10期准确率**: {stats.get('accuracy_10', 0):.2%}\n")
                f.write(f"- **20期准确率**: {stats.get('accuracy_20', 0):.2%}\n")
                f.write(f"- **10期平均收益**: {stats.get('avg_return_10', 0):.4f}\n")
                f.write(f"- **20期平均收益**: {stats.get('avg_return_20', 0):.4f}\n")
                f.write(f"- **DoD通过**: {'✅ 通过' if stats.get('dod_passed', False) else '❌ 未通过'}\n\n")
            
            f.write("## 各交易对详细结果\n\n")
            for symbol, result in results.get('results_by_symbol', {}).items():
                f.write(f"### {symbol}\n\n")
                f.write(f"- **数据源类型**: {result.get('type', 'Unknown')}\n")
                f.write(f"- **样本数**: {result.get('total_samples', 0):,}\n")
                f.write(f"- **事件数**: {result.get('total_events', 0)}\n")
                f.write(f"- **10期准确率**: {result.get('accuracy_10', 0):.2%}\n")
                f.write(f"- **20期准确率**: {result.get('accuracy_20', 0):.2%}\n")
                f.write(f"- **10期平均收益**: {result.get('avg_return_10', 0):.4f}\n")
                f.write(f"- **20期平均收益**: {result.get('avg_return_20', 0):.4f}\n")
                f.write(f"- **P95检测延迟**: {result.get('p95_detection_time', 0)*1000:.3f}ms\n")
                
                # 事件类型分布
                events_by_type = result.get('events_by_type', {})
                if events_by_type:
                    f.write(f"- **事件类型分布**:\n")
                    for event_type, count in events_by_type.items():
                        f.write(f"  - {event_type}: {count}\n")
                f.write("\n")
            
            f.write("## 结论\n\n")
            overall_acc_10 = results.get('overall_accuracy_10', 0)
            overall_acc_20 = results.get('overall_accuracy_20', 0)
            
            if overall_acc_10 >= 0.55:
                f.write("✅ **通过**: 10期准确率达到55%以上\n")
            else:
                f.write("❌ **未通过**: 10期准确率未达到55%\n")
            
            if overall_acc_20 >= 0.55:
                f.write("✅ **通过**: 20期准确率达到55%以上\n")
            else:
                f.write("❌ **未通过**: 20期准确率未达到55%\n")
            
            f.write(f"\n**总体评估**: 基于 {results.get('successful_sources', 0)} 个数据源，")
            f.write(f"共 {results.get('total_events', 0)} 个背离事件的回测结果，")
            f.write(f"背离检测系统在真实数据上的表现需要进一步优化。\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='综合真实数据背离检测回测脚本')
    parser.add_argument('--output', type=str, default='comprehensive_real_data_backtest', 
                       help='输出目录')
    parser.add_argument('--swing_L', type=int, default=20, 
                       help='枢轴窗口长度')
    parser.add_argument('--z_hi', type=float, default=2.0, 
                       help='高强度阈值')
    parser.add_argument('--z_mid', type=float, default=1.0, 
                       help='中等强度阈值')
    
    args = parser.parse_args()
    
    # 创建配置（使用新的回测友好参数）
    config = DivergenceConfig(
        swing_L=args.swing_L,
        z_hi=args.z_hi,
        z_mid=args.z_mid,
        min_separation=6,      # 降低最小间距
        cooldown_secs=1.0,     # 减少冷却时间
        warmup_min=100,        # 减少暖启动样本
        max_lag=0.300,
        use_fusion=True,
        cons_min=0.3,
        weak_threshold=35.0    # 降低弱阈值
    )
    
    # 创建回测器
    backtester = ComprehensiveRealDataBacktester(config)
    
    # 运行综合回测
    results = backtester.run_comprehensive_backtest()
    
    if not results:
        print("回测失败，没有可用的数据源")
        return 1
    
    # 生成报告
    report_file = backtester.generate_comprehensive_report(results, args.output)
    print(f"\n综合回测报告已生成: {report_file}")
    
    # 打印关键结果
    print("\n=== 综合回测结果摘要 ===")
    print(f"成功处理数据源: {results['successful_sources']}/{results['total_sources']}")
    print(f"总样本数: {results['total_samples']:,}")
    print(f"总事件数: {results['total_events']}")
    print(f"10期准确率: {results['overall_accuracy_10']:.2%}")
    print(f"20期准确率: {results['overall_accuracy_20']:.2%}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
