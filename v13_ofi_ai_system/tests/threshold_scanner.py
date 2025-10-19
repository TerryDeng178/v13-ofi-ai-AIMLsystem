"""
OFI+CVD融合阈值扫描工具

用于优化融合信号阈值，生成Precision/Recall报告
支持网格扫描和收益分布分析

Author: V13 OFI+CVD AI Trading System
Date: 2025-10-19
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import time
from dataclasses import dataclass
from src.ofi_cvd_fusion import OFI_CVD_Fusion, OFICVDFusionConfig


@dataclass
class ScanResult:
    """扫描结果"""
    threshold: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    signal_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


class ThresholdScanner:
    """阈值扫描器"""
    
    def __init__(self, config: OFICVDFusionConfig = None):
        self.config = config or OFICVDFusionConfig()
        self.results: List[ScanResult] = []
        self.n_samples = 0  # 添加数据点计数
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        生成合成测试数据
        
        Args:
            n_samples: 样本数量
            
        Returns:
            (z_ofi, z_cvd, timestamps, true_signals)
        """
        np.random.seed(42)
        
        # 生成时间序列
        timestamps = np.linspace(0, 3600, n_samples)  # 1小时数据
        
        # 生成OFI和CVD Z-scores
        # 添加一些趋势和噪声
        trend = np.sin(timestamps / 100) * 2
        noise_ofi = np.random.normal(0, 0.5, n_samples)
        noise_cvd = np.random.normal(0, 0.5, n_samples)
        
        z_ofi = trend + noise_ofi
        z_cvd = trend * 0.8 + noise_cvd
        
        # 生成真实信号标签
        true_signals = np.zeros(n_samples, dtype=int)
        
        # 强买入: 两个Z-score都高且一致
        strong_buy_mask = (z_ofi > 2.0) & (z_cvd > 1.5) & (z_ofi * z_cvd > 0)
        true_signals[strong_buy_mask] = 2  # strong_buy
        
        # 买入: 融合得分高
        fusion_scores = 0.6 * z_ofi + 0.4 * z_cvd
        buy_mask = (fusion_scores > 1.0) & (fusion_scores <= 2.0) & (z_ofi * z_cvd > 0)
        true_signals[buy_mask] = 1  # buy
        
        # 强卖出: 两个Z-score都低且一致
        strong_sell_mask = (z_ofi < -2.0) & (z_cvd < -1.5) & (z_ofi * z_cvd > 0)
        true_signals[strong_sell_mask] = -2  # strong_sell
        
        # 卖出: 融合得分低
        sell_mask = (fusion_scores < -1.0) & (fusion_scores >= -2.0) & (z_ofi * z_cvd > 0)
        true_signals[sell_mask] = -1  # sell
        
        return z_ofi, z_cvd, timestamps, true_signals
    
    def scan_thresholds(self, z_ofi: np.ndarray, z_cvd: np.ndarray, 
                       timestamps: np.ndarray, true_signals: np.ndarray,
                       threshold_range: Tuple[float, float] = (0.5, 3.0),
                       step: float = 0.1) -> List[ScanResult]:
        """
        扫描阈值范围
        
        Args:
            z_ofi: OFI Z-scores
            z_cvd: CVD Z-scores
            timestamps: 时间戳
            true_signals: 真实信号标签
            threshold_range: 阈值扫描范围
            step: 扫描步长
            
        Returns:
            扫描结果列表
        """
        results = []
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        self.n_samples = len(z_ofi)  # 记录数据点数量
        
        for threshold in thresholds:
            # 创建融合器
            config = OFICVDFusionConfig(
                fuse_buy=threshold,
                fuse_strong_buy=threshold * 1.5,
                fuse_sell=-threshold,
                fuse_strong_sell=-threshold * 1.5
            )
            fusion = OFI_CVD_Fusion(config)
            
            # 处理所有样本
            predicted_signals = []
            for i in range(len(z_ofi)):
                result = fusion.update(z_ofi[i], z_cvd[i], timestamps[i])
                
                # 转换信号为数值
                signal_map = {
                    'neutral': 0,
                    'buy': 1,
                    'strong_buy': 2,
                    'sell': -1,
                    'strong_sell': -2
                }
                predicted_signals.append(signal_map[result['signal']])
            
            predicted_signals = np.array(predicted_signals)
            
            # 计算指标
            metrics = self._calculate_metrics(true_signals, predicted_signals)
            
            result = ScanResult(
                threshold=threshold,
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                accuracy=metrics['accuracy'],
                signal_count=metrics['signal_count'],
                true_positive=metrics['tp'],
                false_positive=metrics['fp'],
                false_negative=metrics['fn'],
                true_negative=metrics['tn']
            )
            results.append(result)
        
        self.results = results
        return results
    
    def _calculate_metrics(self, true_signals: np.ndarray, 
                          predicted_signals: np.ndarray) -> Dict[str, float]:
        """计算分类指标"""
        # 二分类: 有信号 vs 无信号
        true_binary = (true_signals != 0).astype(int)
        pred_binary = (predicted_signals != 0).astype(int)
        
        tp = np.sum((true_binary == 1) & (pred_binary == 1))
        fp = np.sum((true_binary == 0) & (pred_binary == 1))
        fn = np.sum((true_binary == 1) & (pred_binary == 0))
        tn = np.sum((true_binary == 0) & (pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'signal_count': np.sum(pred_binary),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    def find_optimal_threshold(self, metric: str = 'f1_score') -> ScanResult:
        """找到最优阈值"""
        if not self.results:
            raise ValueError("请先运行扫描")
        
        best_result = max(self.results, key=lambda x: getattr(x, metric))
        return best_result
    
    def generate_report(self, output_file: str = None) -> str:
        """生成扫描报告"""
        if not self.results:
            return "没有扫描结果"
        
        # 找到最优阈值
        best_f1 = self.find_optimal_threshold('f1_score')
        best_precision = self.find_optimal_threshold('precision')
        best_recall = self.find_optimal_threshold('recall')
        
        report = f"""
# OFI+CVD融合阈值扫描报告

## 扫描参数
- 阈值范围: {min(r.threshold for r in self.results):.1f} - {max(r.threshold for r in self.results):.1f}
- 扫描步长: 0.1
- 数据点数量: {self.n_samples}
- 阈值点数量: {len(self.results)}

## 最优阈值推荐

### F1-Score最优
- 阈值: {best_f1.threshold:.2f}
- F1-Score: {best_f1.f1_score:.3f}
- Precision: {best_f1.precision:.3f}
- Recall: {best_f1.recall:.3f}
- Accuracy: {best_f1.accuracy:.3f}
- 信号数量: {best_f1.signal_count}

### Precision最优
- 阈值: {best_precision.threshold:.2f}
- F1-Score: {best_precision.f1_score:.3f}
- Precision: {best_precision.precision:.3f}
- Recall: {best_precision.recall:.3f}

### Recall最优
- 阈值: {best_recall.threshold:.2f}
- F1-Score: {best_recall.f1_score:.3f}
- Precision: {best_recall.precision:.3f}
- Recall: {best_recall.recall:.3f}

## 详细结果
"""
        
        # 添加详细结果表格
        report += "\n| 阈值 | F1-Score | Precision | Recall | Accuracy | 信号数 |\n"
        report += "|------|----------|-----------|--------|----------|--------|\n"
        
        for result in self.results[::5]:  # 每5个显示一个
            report += f"| {result.threshold:.1f} | {result.f1_score:.3f} | {result.precision:.3f} | {result.recall:.3f} | {result.accuracy:.3f} | {result.signal_count} |\n"
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def plot_results(self, output_file: str = None):
        """绘制扫描结果图表"""
        if not self.results:
            print("没有扫描结果")
            return
        
        thresholds = [r.threshold for r in self.results]
        f1_scores = [r.f1_score for r in self.results]
        precisions = [r.precision for r in self.results]
        recalls = [r.recall for r in self.results]
        accuracies = [r.accuracy for r in self.results]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(thresholds, f1_scores, 'b-', label='F1-Score')
        plt.xlabel('阈值')
        plt.ylabel('F1-Score')
        plt.title('F1-Score vs 阈值')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(thresholds, precisions, 'r-', label='Precision')
        plt.plot(thresholds, recalls, 'g-', label='Recall')
        plt.xlabel('阈值')
        plt.ylabel('分数')
        plt.title('Precision & Recall vs 阈值')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(thresholds, accuracies, 'm-', label='Accuracy')
        plt.xlabel('阈值')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs 阈值')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        signal_counts = [r.signal_count for r in self.results]
        plt.plot(thresholds, signal_counts, 'c-', label='信号数量')
        plt.xlabel('阈值')
        plt.ylabel('信号数量')
        plt.title('信号数量 vs 阈值')
        plt.grid(True)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """主函数"""
    print("开始OFI+CVD融合阈值扫描...")
    
    # 创建扫描器
    scanner = ThresholdScanner()
    
    # 生成测试数据
    print("生成测试数据...")
    z_ofi, z_cvd, timestamps, true_signals = scanner.generate_synthetic_data(1000)
    
    # 运行扫描
    print("运行阈值扫描...")
    results = scanner.scan_thresholds(z_ofi, z_cvd, timestamps, true_signals)
    
    # 生成报告
    print("生成扫描报告...")
    report = scanner.generate_report("threshold_scan_report.md")
    print(report)
    
    # 绘制图表
    print("绘制结果图表...")
    scanner.plot_results("threshold_scan_plots.png")
    
    print("扫描完成！")


if __name__ == "__main__":
    main()
