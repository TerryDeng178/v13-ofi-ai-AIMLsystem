"""
OFI+CVD融合指标监控模块

提供Prometheus指标集成，支持Grafana可视化

Author: V13 OFI+CVD AI Trading System
Date: 2025-10-19
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from src.ofi_cvd_fusion import OFI_CVD_Fusion, SignalType


@dataclass
class FusionMetrics:
    """融合指标数据类"""
    fusion_score: float
    signal: str
    consistency: float
    ofi_weight: float
    cvd_weight: float
    reason_codes: list
    components: Dict[str, float]
    warmup: bool
    stats: Dict[str, int]
    timestamp: float


class FusionMetricsCollector:
    """融合指标收集器"""
    
    def __init__(self, fusion: OFI_CVD_Fusion):
        self.fusion = fusion
        self.metrics_history = []
        self.max_history = 1000  # 保留最近1000条记录
    
    def collect_metrics(self, z_ofi: float, z_cvd: float, ts: float,
                       price: Optional[float] = None, lag_sec: float = 0.0) -> FusionMetrics:
        """
        收集融合指标
        
        Args:
            z_ofi: OFI Z-score
            z_cvd: CVD Z-score
            ts: 时间戳
            price: 可选价格
            lag_sec: 时间滞后
            
        Returns:
            融合指标对象
        """
        # 更新融合器
        result = self.fusion.update(z_ofi, z_cvd, ts, price, lag_sec)
        
        # 创建指标对象（添加容错处理）
        metrics = FusionMetrics(
            fusion_score=result['fusion_score'],
            signal=result['signal'],
            consistency=result['consistency'],
            ofi_weight=result['ofi_weight'],
            cvd_weight=result['cvd_weight'],
            reason_codes=result['reason_codes'],
            components=result['components'],
            warmup=result['warmup'],
            stats=result.get('stats', {}),  # 修复：添加容错处理
            timestamp=ts
        )
        
        # 添加到历史记录
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        return metrics
    
    def get_prometheus_metrics(self) -> Dict[str, Any]:
        """
        生成Prometheus格式的指标
        
        Returns:
            Prometheus指标字典
        """
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        
        # 基础指标
        metrics = {
            'fusion_score': latest.fusion_score,
            'consistency': latest.consistency,
            'ofi_weight': latest.ofi_weight,
            'cvd_weight': latest.cvd_weight,
            'warmup': 1 if latest.warmup else 0,
            'timestamp': latest.timestamp
        }
        
        # 信号计数
        signal_counts = {}
        for metrics_item in self.metrics_history:
            signal = metrics_item.signal
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        metrics['signal_counts'] = signal_counts
        
        # 统计信息
        if latest.stats:
            metrics.update({
                'total_updates': latest.stats.get('total_updates', 0),
                'downgrades': latest.stats.get('downgrades', 0),
                'warmup_returns': latest.stats.get('warmup_returns', 0),
                'invalid_inputs': latest.stats.get('invalid_inputs', 0),
                'lag_exceeded': latest.stats.get('lag_exceeded', 0)
            })
            # 保持嵌套结构兼容性
            metrics["stats"] = latest.stats
        
        # 组件得分
        metrics['ofi_component'] = latest.components.get('ofi', 0.0)
        metrics['cvd_component'] = latest.components.get('cvd', 0.0)
        
        return metrics
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        获取汇总统计信息
        
        Returns:
            汇总统计字典
        """
        if not self.metrics_history:
            return {}
        
        # 计算统计信息
        fusion_scores = [m.fusion_score for m in self.metrics_history]
        consistencies = [m.consistency for m in self.metrics_history]
        
        # 信号分布
        signal_distribution = {}
        for metrics_item in self.metrics_history:
            signal = metrics_item.signal
            signal_distribution[signal] = signal_distribution.get(signal, 0) + 1
        
        # 计算百分比
        total_signals = len(self.metrics_history)
        signal_percentages = {
            signal: count / total_signals * 100 
            for signal, count in signal_distribution.items()
        }
        
        return {
            'total_samples': len(self.metrics_history),
            'fusion_score_stats': {
                'mean': sum(fusion_scores) / len(fusion_scores),
                'min': min(fusion_scores),
                'max': max(fusion_scores),
                'std': self._calculate_std(fusion_scores)
            },
            'consistency_stats': {
                'mean': sum(consistencies) / len(consistencies),
                'min': min(consistencies),
                'max': max(consistencies),
                'std': self._calculate_std(consistencies)
            },
            'signal_distribution': signal_distribution,
            'signal_percentages': signal_percentages,
            'warmup_ratio': sum(1 for m in self.metrics_history if m.warmup) / len(self.metrics_history),
            'time_span': {
                'start': self.metrics_history[0].timestamp,
                'end': self.metrics_history[-1].timestamp,
                'duration': self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp
            }
        }
    
    def _calculate_std(self, values: list) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def export_to_json(self, filepath: str) -> None:
        """
        导出指标到JSON文件
        
        Args:
            filepath: 输出文件路径
        """
        import json
        
        data = {
            'summary': self.get_summary_stats(),
            'metrics': [
                {
                    'timestamp': m.timestamp,
                    'fusion_score': m.fusion_score,
                    'signal': m.signal,
                    'consistency': m.consistency,
                    'ofi_weight': m.ofi_weight,
                    'cvd_weight': m.cvd_weight,
                    'reason_codes': m.reason_codes,
                    'components': m.components,
                    'warmup': m.warmup,
                    'stats': m.stats
                }
                for m in self.metrics_history
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def clear_history(self) -> None:
        """清空历史记录"""
        self.metrics_history.clear()


def create_fusion_metrics_collector(config: Optional[Dict[str, Any]] = None) -> FusionMetricsCollector:
    """
    创建融合指标收集器
    
    Args:
        config: 融合器配置
        
    Returns:
        指标收集器实例
    """
    from src.ofi_cvd_fusion import OFICVDFusionConfig
    
    fusion_config = OFICVDFusionConfig(**(config or {}))
    fusion = OFI_CVD_Fusion(fusion_config)
    
    return FusionMetricsCollector(fusion)


# 示例使用
if __name__ == "__main__":
    # 创建指标收集器
    collector = create_fusion_metrics_collector()
    
    # 模拟数据收集
    import random
    import time
    
    print("开始收集融合指标...")
    
    for i in range(100):
        z_ofi = random.gauss(0, 1)
        z_cvd = random.gauss(0, 1)
        ts = time.time()
        
        metrics = collector.collect_metrics(z_ofi, z_cvd, ts)
        
        if i % 20 == 0:
            print(f"样本 {i}: 融合得分={metrics.fusion_score:.3f}, 信号={metrics.signal}, 一致性={metrics.consistency:.3f}")
    
    # 生成报告
    print("\n=== 汇总统计 ===")
    summary = collector.get_summary_stats()
    print(f"总样本数: {summary['total_samples']}")
    print(f"融合得分均值: {summary['fusion_score_stats']['mean']:.3f}")
    print(f"一致性均值: {summary['consistency_stats']['mean']:.3f}")
    print(f"信号分布: {summary['signal_distribution']}")
    print(f"暖启动比例: {summary['warmup_ratio']:.1%}")
    
    # 导出到文件
    collector.export_to_json("fusion_metrics.json")
    print("\n指标已导出到 fusion_metrics.json")
