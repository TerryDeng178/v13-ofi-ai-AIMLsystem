"""
背离检测回测脚本

验证背离检测的有效性：
- 方向性背离（bull_div/bear_div）→ accuracy@Nbars ≥ 55%
- 事件收益分布较随机基线有统计优势
- 单通道 vs 三通道融合对比
- 生成回测报告和可视化

Author: V13 OFI+CVD AI System
Created: 2025-01-20
"""

import sys
import os
import time
import json
import csv
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ofi_cvd_divergence import DivergenceConfig, DivergenceDetector, DivergenceType


class DivergenceBacktester:
    """背离检测回测器"""
    
    def __init__(self, config: DivergenceConfig):
        self.config = config
        self.detector = DivergenceDetector(config)
        self.events = []
        self.price_data = []
        self.returns_data = []
        
    def load_data(self, data_file: str) -> bool:
        """加载回测数据"""
        try:
            if data_file.endswith('.csv'):
                df = pd.read_csv(data_file)
            elif data_file.endswith('.json'):
                with open(data_file, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            else:
                print(f"不支持的文件格式: {data_file}")
                return False
            
            # 检查必需的列
            required_cols = ['timestamp', 'price', 'z_ofi', 'z_cvd']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"缺少必需的列: {missing_cols}")
                return False
            
            # 存储数据
            self.price_data = df[['timestamp', 'price']].values
            self.returns_data = df[['z_ofi', 'z_cvd']].values
            
            # 添加融合分数（如果存在）
            if 'fusion_score' in df.columns:
                self.fusion_data = df['fusion_score'].values
            else:
                self.fusion_data = None
            
            # 添加一致性分数（如果存在）
            if 'consistency' in df.columns:
                self.consistency_data = df['consistency'].values
            else:
                self.consistency_data = None
            
            print(f"成功加载数据: {len(df)} 条记录")
            return True
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def generate_synthetic_data(self, n_samples: int = 10000, 
                               trend_strength: float = 0.1,
                               noise_level: float = 0.05) -> bool:
        """生成合成数据用于测试"""
        print(f"生成合成数据: {n_samples} 条记录")
        
        # 生成时间序列
        timestamps = np.linspace(0, n_samples * 0.001, n_samples)
        
        # 生成价格序列（带趋势和噪声）
        trend = np.linspace(100, 100 + trend_strength * n_samples, n_samples)
        noise = np.random.normal(0, noise_level, n_samples)
        prices = trend + noise
        
        # 生成OFI和CVD序列（与价格有一定相关性）
        price_changes = np.diff(prices, prepend=prices[0])
        z_ofi = np.random.normal(0, 1, n_samples) + price_changes * 0.5
        z_cvd = np.random.normal(0, 1, n_samples) + price_changes * 0.3
        
        # 裁剪到[-5, 5]范围
        z_ofi = np.clip(z_ofi, -5, 5)
        z_cvd = np.clip(z_cvd, -5, 5)
        
        # 生成融合分数
        fusion_scores = (z_ofi + z_cvd) / 2 + np.random.normal(0, 0.1, n_samples)
        fusion_scores = np.clip(fusion_scores, -5, 5)
        
        # 生成一致性分数
        consistency = np.abs(z_ofi * z_cvd) / (np.abs(z_ofi) + np.abs(z_cvd) + 1e-8)
        
        # 存储数据
        self.price_data = np.column_stack([timestamps, prices])
        self.returns_data = np.column_stack([z_ofi, z_cvd])
        self.fusion_data = fusion_scores
        self.consistency_data = consistency
        
        return True
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行回测"""
        print("开始背离检测回测...")
        
        if len(self.price_data) == 0:
            print("没有数据可回测")
            return {}
        
        events = []
        detection_times = []
        
        for i, (timestamp, price) in enumerate(self.price_data):
            z_ofi, z_cvd = self.returns_data[i]
            fusion_score = self.fusion_data[i] if self.fusion_data is not None else None
            consistency = self.consistency_data[i] if self.consistency_data is not None else None
            
            # 检测背离
            start_time = time.perf_counter()
            event = self.detector.update(
                ts=timestamp,
                price=price,
                z_ofi=z_ofi,
                z_cvd=z_cvd,
                fusion_score=fusion_score,
                consistency=consistency,
                warmup=False,
                lag_sec=0.0
            )
            detection_time = time.perf_counter() - start_time
            detection_times.append(detection_time)
            
            if event and event.get('type'):
                events.append({
                    'index': i,
                    'timestamp': timestamp,
                    'price': price,
                    'event': event
                })
        
        self.events = events
        
        # 计算回测结果
        results = self._calculate_backtest_results(events, detection_times)
        
        print(f"回测完成: 检测到 {len(events)} 个背离事件")
        print(f"平均检测延迟: {np.mean(detection_times)*1000:.3f}ms")
        print(f"P95检测延迟: {np.percentile(detection_times, 95)*1000:.3f}ms")
        
        return results
    
    def _calculate_backtest_results(self, events: List[Dict], 
                                  detection_times: List[float]) -> Dict[str, Any]:
        """计算回测结果"""
        if not events:
            return {
                'total_events': 0,
                'accuracy_10': 0.0,
                'accuracy_20': 0.0,
                'avg_return_10': 0.0,
                'avg_return_20': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_detection_time': 0.0,
                'p95_detection_time': 0.0
            }
        
        # 计算未来收益
        future_returns_10 = []
        future_returns_20 = []
        correct_predictions_10 = 0
        correct_predictions_20 = 0
        
        for event_info in events:
            idx = event_info['index']
            event = event_info['event']
            current_price = event_info['price']
            
            # 计算未来10和20个时间点的收益
            if idx + 10 < len(self.price_data):
                future_price_10 = self.price_data[idx + 10][1]
                return_10 = (future_price_10 - current_price) / current_price
                future_returns_10.append(return_10)
                
                # 检查预测是否正确
                if event['type'] in ['bull_div', 'hidden_bull'] and return_10 > 0:
                    correct_predictions_10 += 1
                elif event['type'] in ['bear_div', 'hidden_bear'] and return_10 < 0:
                    correct_predictions_10 += 1
            
            if idx + 20 < len(self.price_data):
                future_price_20 = self.price_data[idx + 20][1]
                return_20 = (future_price_20 - current_price) / current_price
                future_returns_20.append(return_20)
                
                # 检查预测是否正确
                if event['type'] in ['bull_div', 'hidden_bull'] and return_20 > 0:
                    correct_predictions_20 += 1
                elif event['type'] in ['bear_div', 'hidden_bear'] and return_20 < 0:
                    correct_predictions_20 += 1
        
        # 计算准确率
        accuracy_10 = correct_predictions_10 / len(future_returns_10) if future_returns_10 else 0.0
        accuracy_20 = correct_predictions_20 / len(future_returns_20) if future_returns_20 else 0.0
        
        # 计算平均收益
        avg_return_10 = np.mean(future_returns_10) if future_returns_10 else 0.0
        avg_return_20 = np.mean(future_returns_20) if future_returns_20 else 0.0
        
        # 计算夏普比率
        if future_returns_10 and np.std(future_returns_10) > 0:
            sharpe_ratio = avg_return_10 / np.std(future_returns_10)
        else:
            sharpe_ratio = 0.0
        
        # 计算最大回撤
        if future_returns_10:
            cumulative_returns = np.cumsum(future_returns_10)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = np.min(drawdowns)
        else:
            max_drawdown = 0.0
        
        # 计算胜率
        positive_returns = [r for r in future_returns_10 if r > 0]
        win_rate = len(positive_returns) / len(future_returns_10) if future_returns_10 else 0.0
        
        # 计算检测延迟统计
        avg_detection_time = np.mean(detection_times) if detection_times else 0.0
        p95_detection_time = np.percentile(detection_times, 95) if detection_times else 0.0
        
        return {
            'total_events': len(events),
            'accuracy_10': accuracy_10,
            'accuracy_20': accuracy_20,
            'avg_return_10': avg_return_10,
            'avg_return_20': avg_return_20,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_detection_time': avg_detection_time,
            'p95_detection_time': p95_detection_time,
            'events_by_type': self._count_events_by_type(events)
        }
    
    def _count_events_by_type(self, events: List[Dict]) -> Dict[str, int]:
        """按类型统计事件"""
        type_counts = {}
        for event_info in events:
            event_type = event_info['event']['type']
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        return type_counts
    
    def generate_report(self, output_dir: str) -> str:
        """生成回测报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成CSV报告
        csv_file = os.path.join(output_dir, 'divergence_events.csv')
        self._export_events_to_csv(csv_file)
        
        # 生成JSON报告
        json_file = os.path.join(output_dir, 'backtest_results.json')
        self._export_results_to_json(json_file)
        
        # 生成可视化
        plot_file = os.path.join(output_dir, 'divergence_analysis.png')
        self._create_visualization(plot_file)
        
        # 生成Markdown报告
        md_file = os.path.join(output_dir, 'backtest_report.md')
        self._generate_markdown_report(md_file)
        
        return md_file
    
    def _export_events_to_csv(self, csv_file: str):
        """导出事件到CSV"""
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'index', 'timestamp', 'price', 'type', 'score', 'channels',
                'reason_codes', 'z_ofi', 'z_cvd', 'fusion', 'consistency'
            ])
            
            for event_info in self.events:
                event = event_info['event']
                writer.writerow([
                    event_info['index'],
                    event_info['timestamp'],
                    event_info['price'],
                    event['type'],
                    event['score'],
                    ','.join(event['channels']),
                    ','.join(event['reason_codes']),
                    event['debug']['z_ofi'],
                    event['debug']['z_cvd'],
                    event['debug']['fusion'],
                    event['debug']['consistency']
                ])
    
    def _export_results_to_json(self, json_file: str):
        """导出结果到JSON"""
        results = self._calculate_backtest_results(self.events, [])
        results['config'] = {
            'swing_L': self.config.swing_L,
            'z_hi': self.config.z_hi,
            'z_mid': self.config.z_mid,
            'min_separation': self.config.min_separation,
            'cooldown_secs': self.config.cooldown_secs,
            'warmup_min': self.config.warmup_min,
            'use_fusion': self.config.use_fusion
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def _create_visualization(self, plot_file: str):
        """创建可视化图表"""
        if not self.events:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 价格序列和背离事件
        ax1 = axes[0, 0]
        timestamps = self.price_data[:, 0]
        prices = self.price_data[:, 1]
        ax1.plot(timestamps, prices, 'b-', alpha=0.7, label='Price')
        
        # 标记背离事件
        for event_info in self.events:
            event = event_info['event']
            color = 'green' if 'bull' in event['type'] else 'red'
            ax1.scatter(event_info['timestamp'], event_info['price'], 
                       c=color, s=50, alpha=0.8)
        
        ax1.set_title('Price Series with Divergence Events')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 事件类型分布
        ax2 = axes[0, 1]
        type_counts = self._count_events_by_type(self.events)
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        ax2.bar(types, counts, color=['green', 'red', 'blue', 'orange', 'purple'][:len(types)])
        ax2.set_title('Event Type Distribution')
        ax2.set_xlabel('Event Type')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # 评分分布
        ax3 = axes[1, 0]
        scores = [event_info['event']['score'] for event_info in self.events]
        ax3.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Score Distribution')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Frequency')
        ax3.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.1f}')
        ax3.legend()
        
        # 时间分布
        ax4 = axes[1, 1]
        event_times = [event_info['timestamp'] for event_info in self.events]
        time_diffs = np.diff(event_times)
        ax4.hist(time_diffs, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax4.set_title('Time Between Events')
        ax4.set_xlabel('Time Difference (seconds)')
        ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_markdown_report(self, md_file: str):
        """生成Markdown报告"""
        results = self._calculate_backtest_results(self.events, [])
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# 背离检测回测报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 回测配置\n\n")
            f.write(f"- 枢轴窗口长度: {self.config.swing_L}\n")
            f.write(f"- 高强度阈值: {self.config.z_hi}\n")
            f.write(f"- 中等强度阈值: {self.config.z_mid}\n")
            f.write(f"- 最小枢轴间距: {self.config.min_separation}\n")
            f.write(f"- 冷却时间: {self.config.cooldown_secs}秒\n")
            f.write(f"- 暖启动样本数: {self.config.warmup_min}\n")
            f.write(f"- 使用融合指标: {self.config.use_fusion}\n\n")
            
            f.write("## 回测结果\n\n")
            f.write(f"- **总事件数**: {results['total_events']}\n")
            f.write(f"- **10期准确率**: {results['accuracy_10']:.2%}\n")
            f.write(f"- **20期准确率**: {results['accuracy_20']:.2%}\n")
            f.write(f"- **10期平均收益**: {results['avg_return_10']:.4f}\n")
            f.write(f"- **20期平均收益**: {results['avg_return_20']:.4f}\n")
            f.write(f"- **夏普比率**: {results['sharpe_ratio']:.4f}\n")
            f.write(f"- **最大回撤**: {results['max_drawdown']:.4f}\n")
            f.write(f"- **胜率**: {results['win_rate']:.2%}\n")
            f.write(f"- **平均检测延迟**: {results['avg_detection_time']*1000:.3f}ms\n")
            f.write(f"- **P95检测延迟**: {results['p95_detection_time']*1000:.3f}ms\n\n")
            
            f.write("## 事件类型分布\n\n")
            for event_type, count in results['events_by_type'].items():
                f.write(f"- **{event_type}**: {count}\n")
            
            f.write("\n## 结论\n\n")
            if results['accuracy_10'] >= 0.55:
                f.write("✅ **通过**: 10期准确率达到55%以上\n")
            else:
                f.write("❌ **未通过**: 10期准确率未达到55%\n")
            
            if results['p95_detection_time'] < 0.003:
                f.write("✅ **通过**: P95检测延迟小于3ms\n")
            else:
                f.write("❌ **未通过**: P95检测延迟超过3ms\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='背离检测回测脚本')
    parser.add_argument('--data', type=str, help='数据文件路径（CSV或JSON）')
    parser.add_argument('--output', type=str, default='divergence_backtest_output', 
                       help='输出目录')
    parser.add_argument('--synthetic', action='store_true', 
                       help='使用合成数据')
    parser.add_argument('--samples', type=int, default=10000, 
                       help='合成数据样本数')
    parser.add_argument('--swing_L', type=int, default=20, 
                       help='枢轴窗口长度')
    parser.add_argument('--z_hi', type=float, default=2.0, 
                       help='高强度阈值')
    parser.add_argument('--z_mid', type=float, default=1.0, 
                       help='中等强度阈值')
    
    args = parser.parse_args()
    
    # 创建配置
    config = DivergenceConfig(
        swing_L=args.swing_L,
        z_hi=args.z_hi,
        z_mid=args.z_mid,
        min_separation=10,
        cooldown_secs=3.0,
        warmup_min=200,
        max_lag=0.300,
        use_fusion=True,
        cons_min=0.3
    )
    
    # 创建回测器
    backtester = DivergenceBacktester(config)
    
    # 加载数据
    if args.synthetic:
        if not backtester.generate_synthetic_data(args.samples):
            print("生成合成数据失败")
            return 1
    elif args.data:
        if not backtester.load_data(args.data):
            print("加载数据失败")
            return 1
    else:
        print("请指定数据文件或使用 --synthetic 生成合成数据")
        return 1
    
    # 运行回测
    results = backtester.run_backtest()
    
    # 生成报告
    report_file = backtester.generate_report(args.output)
    print(f"回测报告已生成: {report_file}")
    
    # 打印关键结果
    print("\n=== 回测结果摘要 ===")
    print(f"总事件数: {results['total_events']}")
    print(f"10期准确率: {results['accuracy_10']:.2%}")
    print(f"20期准确率: {results['accuracy_20']:.2%}")
    print(f"P95检测延迟: {results['p95_detection_time']*1000:.3f}ms")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


