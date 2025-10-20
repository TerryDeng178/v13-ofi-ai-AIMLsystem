"""
背离检测Prometheus指标集成

提供背离检测模块的Prometheus指标收集和暴露功能：
- 背离事件计数（按类型和通道）
- 背离评分分布
- 最后事件时间戳
- 抑制事件统计
- 枢轴检测统计

Author: V13 OFI+CVD AI System
Created: 2025-01-20
"""

import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry


class DivergenceMetricsCollector:
    """背离检测指标收集器"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
        self._latest_events = {}  # 存储最新事件信息
    
    def _setup_metrics(self):
        """设置Prometheus指标"""
        # 背离事件计数器
        self.divergence_events_total = Counter(
            'divergence_events_total',
            'Total number of divergence events detected',
            ['type', 'channel', 'env'],
            registry=self.registry
        )
        
        # 背离评分分布
        self.divergence_score = Gauge(
            'divergence_score',
            'Latest divergence event score',
            ['type', 'channel', 'env'],
            registry=self.registry
        )
        
        # 最后事件时间戳
        self.divergence_last_ts = Gauge(
            'divergence_last_ts',
            'Timestamp of last divergence event',
            ['type', 'channel', 'env'],
            registry=self.registry
        )
        
        # 抑制事件统计
        self.divergence_suppressed_total = Counter(
            'divergence_suppressed_total',
            'Total number of suppressed divergence events',
            ['reason', 'env'],
            registry=self.registry
        )
        
        # 枢轴检测统计
        self.divergence_pivots_detected = Counter(
            'divergence_pivots_detected_total',
            'Total number of pivots detected',
            ['indicator', 'env'],
            registry=self.registry
        )
        
        # 背离检测延迟
        self.divergence_detection_duration = Histogram(
            'divergence_detection_duration_seconds',
            'Time spent on divergence detection',
            ['env'],
            buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
            registry=self.registry
        )
        
        # 背离检测频率
        self.divergence_detection_rate = Gauge(
            'divergence_detection_rate_per_second',
            'Current divergence detection rate',
            ['env'],
            registry=self.registry
        )
        
        # 背离类型分布（改为Counter）
        self.divergence_type_distribution = Counter(
            'divergence_type_distribution_total',
            'Total count of divergence types',
            ['type', 'env'],
            registry=self.registry
        )
        
        # 背离评分分布（新增Histogram）
        self.divergence_score_histogram = Histogram(
            'divergence_score_histogram',
            'Distribution of divergence scores',
            ['type', 'env'],
            buckets=[0, 20, 40, 50, 60, 70, 80, 90, 100],
            registry=self.registry
        )
    
    def record_divergence_event(self, event: Dict[str, Any], env: str = "testing"):
        """记录背离事件"""
        if not event or event.get('type') is None:
            return
        
        event_type = event['type']
        channels = event.get('channels', [])
        score = event.get('score', 0.0)
        ts = event.get('ts', time.time())
        
        # 记录事件计数
        for channel in channels:
            self.divergence_events_total.labels(
                type=event_type,
                channel=channel,
                env=env
            ).inc()
            
            # 更新评分
            self.divergence_score.labels(
                type=event_type,
                channel=channel,
                env=env
            ).set(score)
            
            # 更新时间戳
            self.divergence_last_ts.labels(
                type=event_type,
                channel=channel,
                env=env
            ).set(ts)
        
        # 更新背离类型分布
        self.divergence_type_distribution.labels(
            type=event_type,
            env=env
        ).inc()
        
        # 记录评分分布
        self.divergence_score_histogram.labels(
            type=event_type,
            env=env
        ).observe(score)
        
        # 存储最新事件信息
        self._latest_events[event_type] = {
            'score': score,
            'channels': channels,
            'ts': ts,
            'env': env
        }
    
    def record_suppressed_event(self, reason: str, env: str = "testing"):
        """记录被抑制的事件"""
        self.divergence_suppressed_total.labels(
            reason=reason,
            env=env
        ).inc()
    
    def record_pivots_detected(self, indicator: str, count: int, env: str = "testing"):
        """记录检测到的枢轴数量"""
        self.divergence_pivots_detected.labels(
            indicator=indicator,
            env=env
        ).inc(count)
    
    def record_detection_duration(self, duration: float, env: str = "testing"):
        """记录检测延迟"""
        self.divergence_detection_duration.labels(env=env).observe(duration)
    
    def update_detection_rate(self, rate: float, env: str = "testing"):
        """更新检测频率"""
        self.divergence_detection_rate.labels(env=env).set(rate)
    
    def get_prometheus_metrics(self) -> str:
        """获取Prometheus格式的指标"""
        from prometheus_client import generate_latest
        return generate_latest(self.registry).decode('utf-8')
    
    def get_latest_events(self) -> Dict[str, Any]:
        """获取最新事件信息"""
        return self._latest_events.copy()
    
    def reset_metrics(self):
        """重置所有指标（注意：Counter类型不应重置，这里只清空最新事件）"""
        # 注意：Prometheus Counter应该只增不减，这里不重置Counter
        # 如果需要重置，应该创建新的CollectorRegistry
        
        # 清空最新事件
        self._latest_events.clear()


class DivergencePrometheusExporter:
    """背离检测Prometheus导出器"""
    
    def __init__(self, port: int = 8004, env: str = "testing", config_loader=None):
        # 从配置加载器获取端口配置
        if config_loader:
            port = config_loader.get('monitoring.divergence_metrics.port', port)
            env = config_loader.get('monitoring.divergence_metrics.env', env)
        
        self.port = port
        self.env = env
        self.metrics_collector = DivergenceMetricsCollector()
        self.detector = None
        self._running = False
        self._start_time = None
        self._event_count = 0
        # 新增：增量统计水位
        self._last_suppressed = {}
        self._last_pivots = {'ofi': 0, 'cvd': 0, 'fusion': 0}
    
    def set_detector(self, detector):
        """设置背离检测器"""
        self.detector = detector
    
    def start_auto_update(self, update_interval: float = 1.0):
        """启动自动更新"""
        import threading
        import time
        
        self._running = True
        self._start_time = time.time()
        
        def update_loop():
            while self._running:
                try:
                    self._update_metrics()
                    time.sleep(update_interval)
                except Exception as e:
                    print(f"Error updating divergence metrics: {e}")
        
        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()
    
    def stop_auto_update(self):
        """停止自动更新"""
        self._running = False
        if hasattr(self, '_update_thread'):
            self._update_thread.join(timeout=1.0)
    
    def _update_metrics(self):
        """更新指标"""
        if not self.detector:
            return
        
        # 获取检测器统计信息
        stats = self.detector.get_stats()
        
        # 1) 枢轴：分通道增量统计
        by_ch = stats.get('pivots_by_channel', {})
        for ch in ('ofi', 'cvd', 'fusion'):
            cur = by_ch.get(ch, 0)
            inc = max(0, cur - self._last_pivots.get(ch, 0))
            if inc > 0:
                self.metrics_collector.record_pivots_detected(ch, inc, self.env)
                self._last_pivots[ch] = cur
        
        # 2) 抑制原因：按增量统计
        cur_sup = stats.get('suppressed_by_reason', {})
        for reason, count in cur_sup.items():
            last = self._last_suppressed.get(reason, 0)
            inc = max(0, count - last)
            if inc > 0:
                # 按增量次数累加
                for _ in range(inc):
                    self.metrics_collector.record_suppressed_event(reason, self.env)
                self._last_suppressed[reason] = count
        
        # 更新检测频率
        if self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                rate = self._event_count / elapsed
                self.metrics_collector.update_detection_rate(rate, self.env)
    
    def record_event(self, event: Dict[str, Any]):
        """记录背离事件"""
        if event and event.get('type'):
            self.metrics_collector.record_divergence_event(event, self.env)
            self._event_count += 1
    
    def record_detection_duration(self, duration: float):
        """记录检测延迟"""
        self.metrics_collector.record_detection_duration(duration, self.env)
    
    def get_metrics(self) -> str:
        """获取指标"""
        return self.metrics_collector.get_prometheus_metrics()
    
    def start_http_server(self):
        """启动HTTP服务器"""
        from prometheus_client import start_http_server
        start_http_server(self.port, registry=self.metrics_collector.registry)
        print(f"Divergence metrics server started on port {self.port}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            'running': self._running,
            'port': self.port,
            'env': self.env,
            'event_count': self._event_count,
            'uptime': time.time() - self._start_time if self._start_time else 0,
            'latest_events': self.metrics_collector.get_latest_events()
        }
