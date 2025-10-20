"""
OFI+CVD融合Prometheus指标暴露器

将融合指标暴露为真正的Prometheus指标，支持Grafana抓取

Author: V13 OFI+CVD AI Trading System
Date: 2025-10-20
"""

from prometheus_client import Gauge, Counter, Histogram, start_http_server
from typing import Dict, Any, Optional
import time
import threading
from src.fusion_metrics import FusionMetricsCollector


class FusionPrometheusExporter:
    """融合指标Prometheus暴露器"""
    
    def __init__(self, collector: FusionMetricsCollector, port: int = 8005, config_loader=None):
        self.collector = collector
        
        # 从配置加载器获取端口配置
        if config_loader:
            port = config_loader.get('monitoring.fusion_metrics.port', port)
        
        self.port = port
        
        # 定义Prometheus指标
        self.fusion_score_gauge = Gauge(
            'fusion_score',
            'OFI+CVD融合得分',
            ['symbol', 'env']
        )
        
        self.consistency_gauge = Gauge(
            'fusion_consistency',
            '信号一致性',
            ['symbol', 'env']
        )
        
        self.ofi_weight_gauge = Gauge(
            'fusion_ofi_weight',
            'OFI权重',
            ['symbol', 'env']
        )
        
        self.cvd_weight_gauge = Gauge(
            'fusion_cvd_weight',
            'CVD权重',
            ['symbol', 'env']
        )
        
        self.warmup_gauge = Gauge(
            'fusion_warmup',
            '暖启动状态',
            ['symbol', 'env']
        )
        
        self.signal_counter = Counter(
            'fusion_signal_total',
            '信号计数',
            ['signal', 'symbol', 'env']
        )
        
        self.update_duration_histogram = Histogram(
            'fusion_update_duration_seconds',
            '融合更新耗时',
            ['symbol', 'env'],
            buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        )
        
        self.stats_gauge = Gauge(
            'fusion_stats',
            '融合统计信息',
            ['metric', 'symbol', 'env']
        )
        
        # 默认标签
        self.default_labels = {
            'symbol': 'BTCUSDT',
            'env': 'testing'
        }
        
        # 信号计数快照（用于增量更新）
        self._prev_signal_counts = {}
        
        # 更新线程
        self._update_thread = None
        self._stop_event = threading.Event()
    
    def update_metrics(self, symbol: str = None, env: str = None):
        """更新Prometheus指标"""
        labels = {
            'symbol': symbol or self.default_labels['symbol'],
            'env': env or self.default_labels['env']
        }
        
        # 获取最新指标
        prometheus_metrics = self.collector.get_prometheus_metrics()
        
        if not prometheus_metrics:
            return
        
        # 更新基础指标
        self.fusion_score_gauge.labels(**labels).set(prometheus_metrics.get('fusion_score', 0.0))
        self.consistency_gauge.labels(**labels).set(prometheus_metrics.get('consistency', 0.0))
        self.ofi_weight_gauge.labels(**labels).set(prometheus_metrics.get('ofi_weight', 0.0))
        self.cvd_weight_gauge.labels(**labels).set(prometheus_metrics.get('cvd_weight', 0.0))
        self.warmup_gauge.labels(**labels).set(1 if prometheus_metrics.get('warmup', False) else 0)
        
        # 更新信号计数（增量更新）
        signal_counts = prometheus_metrics.get('signal_counts', {})
        for signal, count in signal_counts.items():
            prev = self._prev_signal_counts.get(signal, 0)
            delta = max(0, count - prev)
            if delta:
                self.signal_counter.labels(signal=signal, **labels).inc(delta)
            self._prev_signal_counts[signal] = count
        
        # 更新统计信息
        stats = prometheus_metrics.get('stats', {})
        for metric_name, value in stats.items():
            self.stats_gauge.labels(metric=metric_name, **labels).set(value)
    
    def start_server(self):
        """启动Prometheus HTTP服务器"""
        start_http_server(self.port)
        print(f"Prometheus指标服务器已启动，端口: {self.port}")
        print(f"指标地址: http://localhost:{self.port}/metrics")
    
    def start_auto_update(self, interval: float = 1.0):
        """启动自动更新线程"""
        def update_loop():
            while not self._stop_event.is_set():
                try:
                    # 记录更新耗时
                    t0 = time.perf_counter()
                    self.update_metrics()
                    duration = time.perf_counter() - t0
                    
                    # 记录到histogram
                    labels = self.default_labels.copy()
                    self.update_duration_histogram.labels(**labels).observe(duration)
                    
                    time.sleep(interval)
                except Exception as e:
                    print(f"更新指标时出错: {e}")
        
        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()
        print(f"自动更新线程已启动，间隔: {interval}秒")
    
    def stop_auto_update(self):
        """停止自动更新"""
        if self._update_thread:
            self._stop_event.set()
            self._update_thread.join()
            print("自动更新线程已停止")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        prometheus_metrics = self.collector.get_prometheus_metrics()
        summary_stats = self.collector.get_summary_stats()
        
        return {
            'prometheus_metrics': prometheus_metrics,
            'summary_stats': summary_stats,
            'exporter_status': {
                'port': self.port,
                'auto_update_running': self._update_thread and self._update_thread.is_alive(),
                'metrics_count': len(prometheus_metrics) if prometheus_metrics else 0
            }
        }


def create_fusion_exporter(config: Optional[Dict[str, Any]] = None, 
                          port: int = 8005, 
                          config_loader=None) -> FusionPrometheusExporter:
    """
    创建融合指标暴露器
    
    Args:
        config: 融合器配置
        port: Prometheus服务器端口
        
    Returns:
        指标暴露器实例
    """
    from src.fusion_metrics import create_fusion_metrics_collector
    
    collector = create_fusion_metrics_collector(config)
    exporter = FusionPrometheusExporter(collector, port, config_loader)
    
    return exporter


# 示例使用
if __name__ == "__main__":
    import random
    
    print("启动OFI+CVD融合Prometheus指标暴露器...")
    
    # 创建暴露器
    exporter = create_fusion_exporter(port=8005)
    
    # 启动服务器
    exporter.start_server()
    
    # 启动自动更新
    exporter.start_auto_update(interval=1.0)
    
    # 模拟数据更新
    print("开始模拟数据更新...")
    try:
        for i in range(100):
            z_ofi = random.gauss(0, 1)
            z_cvd = random.gauss(0, 1)
            ts = time.time()
            
            # 收集指标
            metrics = exporter.collector.collect_metrics(z_ofi, z_cvd, ts)
            
            if i % 20 == 0:
                print(f"样本 {i}: 融合得分={metrics.fusion_score:.3f}, 信号={metrics.signal}")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n停止模拟...")
    
    finally:
        # 停止自动更新
        exporter.stop_auto_update()
        
        # 显示最终摘要
        summary = exporter.get_metrics_summary()
        print("\n=== 指标摘要 ===")
        print(f"Prometheus端口: {summary['exporter_status']['port']}")
        print(f"指标数量: {summary['exporter_status']['metrics_count']}")
        print(f"总样本数: {summary['summary_stats'].get('total_samples', 0)}")
        
        print("\n指标暴露器已停止")
