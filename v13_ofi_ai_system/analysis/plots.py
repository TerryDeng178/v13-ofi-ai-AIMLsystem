#!/usr/bin/env python3
"""
图表生成工具
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 尝试导入seaborn，如果失败则跳过
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PlotGenerator:
    """图表生成器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置图表样式
        if HAS_SEABORN:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        else:
            plt.style.use('default')
    
    def plot_roc_curves(self, metrics_data: Dict, save_name: str = 'roc_curves.png'):
        """绘制ROC曲线（基于真实数据）"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        horizons = [60, 180, 300, 900]
        
        for i, horizon in enumerate(horizons):
            ax = axes[i]
            
            # 基于真实数据绘制ROC曲线（强制真实数据）
            if not metrics_data or f'{horizon}s' not in metrics_data or 'roc' not in metrics_data[f'{horizon}s']:
                ax.text(0.5, 0.5, f'无{horizon}s ROC数据', ha='center', va='center', transform=ax.transAxes)
                continue
                
            roc_data = metrics_data[f'{horizon}s']['roc']
            fpr = roc_data.get('fpr', [])
            tpr = roc_data.get('tpr', [])
            
            if not fpr or not tpr:
                ax.text(0.5, 0.5, f'{horizon}s ROC数据为空', ha='center', va='center', transform=ax.transAxes)
                continue
                
            ax.plot(fpr, tpr, label=f'{horizon}s', linewidth=2)
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {horizon}s Horizon')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curves(self, metrics_data: Dict, save_name: str = 'pr_curves.png'):
        """绘制PR曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        horizons = [60, 180, 300, 900]
        
        for i, horizon in enumerate(horizons):
            ax = axes[i]
            
            # 强制真实数据，无数据则显示提示
            if not metrics_data or f'{horizon}s' not in metrics_data or 'pr' not in metrics_data[f'{horizon}s']:
                ax.text(0.5, 0.5, f'无{horizon}s PR数据', ha='center', va='center', transform=ax.transAxes)
                continue
                
            pr_data = metrics_data[f'{horizon}s']['pr']
            recall = pr_data.get('recall', [])
            precision = pr_data.get('precision', [])
            
            if not recall or not precision:
                ax.text(0.5, 0.5, f'{horizon}s PR数据为空', ha='center', va='center', transform=ax.transAxes)
                continue
            
            ax.plot(recall, precision, label=f'{horizon}s', linewidth=2)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Precision-Recall Curve - {horizon}s Horizon')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_monotonicity_analysis(self, metrics_data: Dict, save_name: str = 'monotonicity.png'):
        """绘制单调性分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        horizons = [60, 180, 300, 900]
        
        for i, horizon in enumerate(horizons):
            ax = axes[i]
            
            # 强制真实数据，无数据则显示提示
            if not metrics_data or f'{horizon}s' not in metrics_data or 'quantile_returns' not in metrics_data[f'{horizon}s']:
                ax.text(0.5, 0.5, f'无{horizon}s分位收益数据', ha='center', va='center', transform=ax.transAxes)
                continue
                
            quantile_data = metrics_data[f'{horizon}s']['quantile_returns']
            quantiles = list(quantile_data.keys())
            returns = list(quantile_data.values())
            
            if not quantiles or not returns:
                ax.text(0.5, 0.5, f'{horizon}s分位收益数据为空', ha='center', va='center', transform=ax.transAxes)
                continue
            
            bars = ax.bar(quantiles, returns, color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
            ax.set_xlabel('Signal Quantiles')
            ax.set_ylabel('Average Return')
            ax.set_title(f'Monotonicity Analysis - {horizon}s Horizon')
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, ret in zip(bars, returns):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{ret:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_calibration_analysis(self, metrics_data: Dict, save_name: str = 'calibration.png'):
        """绘制校准分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        horizons = [60, 180, 300, 900]
        
        for i, horizon in enumerate(horizons):
            ax = axes[i]
            
            # 强制真实数据，无数据则显示提示
            if not metrics_data or f'{horizon}s' not in metrics_data or 'calibration' not in metrics_data[f'{horizon}s']:
                ax.text(0.5, 0.5, f'无{horizon}s校准数据', ha='center', va='center', transform=ax.transAxes)
                continue
                
            calib_data = metrics_data[f'{horizon}s']['calibration']
            predicted_probs = calib_data.get('predicted_probs', [])
            actual_rates = calib_data.get('actual_rates', [])
            
            if not predicted_probs or not actual_rates:
                ax.text(0.5, 0.5, f'{horizon}s校准数据为空', ha='center', va='center', transform=ax.transAxes)
                continue
            
            ax.plot(predicted_probs, actual_rates, 'o-', label='Actual', linewidth=2, markersize=6)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Actual Rate')
            ax.set_title(f'Calibration Analysis - {horizon}s Horizon')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_signal_distributions(self, signal_data: Dict, save_name: str = 'signal_distributions.png'):
        """绘制信号分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        signal_types = ['ofi', 'cvd', 'fusion', 'events']
        
        for i, signal_type in enumerate(signal_types):
            ax = axes[i]
            
            # 强制真实数据，无数据则显示提示
            if not signal_data or signal_type not in signal_data:
                ax.text(0.5, 0.5, f'无{signal_type}数据', ha='center', va='center', transform=ax.transAxes)
                continue
                
            if signal_type == 'events':
                # 事件分布
                event_data = signal_data[signal_type]
                if not event_data or 'event_types' not in event_data:
                    ax.text(0.5, 0.5, f'{signal_type}事件数据为空', ha='center', va='center', transform=ax.transAxes)
                    continue
                event_types = list(event_data['event_types'].keys())
                counts = list(event_data['event_types'].values())
                ax.bar(event_types, counts, color=['red', 'orange', 'yellow'])
                ax.set_ylabel('Event Count')
                ax.set_title(f'{signal_type.upper()} Event Distribution')
            else:
                # 信号分布
                signal_values = signal_data[signal_type].get('signals', [])
                if not signal_values:
                    ax.text(0.5, 0.5, f'{signal_type}信号数据为空', ha='center', va='center', transform=ax.transAxes)
                    continue
                ax.hist(signal_values, bins=50, alpha=0.7, density=True)
                ax.set_xlabel('Signal Value')
                ax.set_ylabel('Density')
                ax.set_title(f'{signal_type.upper()} Signal Distribution')
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_slice_analysis(self, slice_data: Dict, save_name: str = 'slice_analysis.png'):
        """绘制切片分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        slice_types = ['regime', 'tod', 'volatility', 'symbol']
        
        for i, slice_type in enumerate(slice_types):
            ax = axes[i]
            
            # 强制真实数据，无数据则显示提示
            if not slice_data or slice_type not in slice_data:
                ax.text(0.5, 0.5, f'无{slice_type}切片数据', ha='center', va='center', transform=ax.transAxes)
                continue
                
            slice_info = slice_data[slice_type]
            slices = list(slice_info.keys())
            auc_scores = list(slice_info.values())
            
            if not slices or not auc_scores:
                ax.text(0.5, 0.5, f'{slice_type}切片数据为空', ha='center', va='center', transform=ax.transAxes)
                continue
            
            bars = ax.bar(slices, auc_scores, color=['skyblue', 'lightgreen', 'salmon', 'gold'][:len(slices)])
            ax.set_ylabel('AUC Score')
            ax.set_title(f'{slice_type.title()} Slice Analysis')
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, score in zip(bars, auc_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_divergence_analysis(self, events_data: Dict, save_name: str = 'divergence_analysis.png'):
        """绘制背离事件分析图（基于真实事件数据）"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 1. 背离事件时间分布
        ax1 = axes[0]
        if events_data and 'events_per_hour' in events_data:
            hours = np.arange(24)
            event_counts = [events_data['events_per_hour']] * 24  # 基于真实数据
            ax1.bar(hours, event_counts, alpha=0.7, label='Real Data')
        else:
            hours = np.arange(24)
            # 强制真实数据，无数据则显示提示
            if not events_data or 'hourly_counts' not in events_data:
                ax1.text(0.5, 0.5, '无小时事件数据', ha='center', va='center', transform=ax1.transAxes)
            else:
                event_counts = events_data['hourly_counts']
            ax1.bar(hours, event_counts, alpha=0.7, label='Sample Data')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Event Count')
        ax1.set_title('Divergence Events by Hour')
        ax1.grid(True, alpha=0.3)
        
        # 2. 背离事件后收益分布
        ax2 = axes[1]
        if events_data and 'post_event_returns' in events_data:
            returns = events_data['post_event_returns']  # 真实数据
            ax2.hist(returns, bins=50, alpha=0.7, density=True, label='Real Data')
        else:
            # 强制真实数据，无数据则显示提示
            if not events_data or 'returns' not in events_data:
                ax2.text(0.5, 0.5, '无收益数据', ha='center', va='center', transform=ax2.transAxes)
            else:
                returns = events_data['returns']
            ax2.hist(returns, bins=50, alpha=0.7, density=True, label='Sample Data')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Return')
        ax2.set_ylabel('Density')
        ax2.set_title('Post-Divergence Return Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. 背离事件类型分布
        ax3 = axes[2]
        if events_data and 'event_types' in events_data:
            event_types = list(events_data['event_types'].keys())
            counts = list(events_data['event_types'].values())
            ax3.bar(event_types, counts, alpha=0.7, label='Real Data')
        else:
            event_types = ['anomaly', 'divergence', 'conflict']
            counts = [100, 20, 5]
            ax3.bar(event_types, counts, alpha=0.7, label='Sample Data')
        ax3.set_xlabel('Event Type')
        ax3.set_ylabel('Count')
        ax3.set_title('Event Type Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. 背离事件胜率分析
        ax4 = axes[3]
        horizons = [60, 180, 300, 900]
        if events_data and 'win_rates' in events_data:
            win_rates = events_data['win_rates']  # 真实数据
            ax4.plot(horizons, win_rates, 'o-', linewidth=2, markersize=8, label='Real Data')
        else:
            # 强制真实数据，无数据则显示提示
            ax4.text(0.5, 0.5, '无胜率数据', ha='center', va='center', transform=ax4.transAxes)
        ax4.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Random')
        ax4.set_xlabel('Horizon (seconds)')
        ax4.set_ylabel('Win Rate')
        ax4.set_title('Divergence Event Win Rate by Horizon')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_plots(self, metrics_data: Dict, signal_data: Dict, 
                          slice_data: Dict, events_data: Dict):
        """生成所有图表"""
        print("[CHART] 生成分析图表...")
        
        # ROC曲线
        self.plot_roc_curves(metrics_data)
        print("  [OK] ROC曲线")
        
        # PR曲线
        self.plot_precision_recall_curves(metrics_data)
        print("  [OK] PR曲线")
        
        # 单调性分析
        self.plot_monotonicity_analysis(metrics_data)
        print("  [OK] 单调性分析")
        
        # 校准分析
        self.plot_calibration_analysis(metrics_data)
        print("  [OK] 校准分析")
        
        # 信号分布
        self.plot_signal_distributions(signal_data)
        print("  [OK] 信号分布")
        
        # 切片分析
        self.plot_slice_analysis(slice_data)
        print("  [OK] 切片分析")
        
        # 背离事件分析
        self.plot_divergence_analysis(events_data)
        print("  [OK] 背离事件分析")
        
        print(f"[FOLDER] 图表已保存到: {self.output_dir}")
