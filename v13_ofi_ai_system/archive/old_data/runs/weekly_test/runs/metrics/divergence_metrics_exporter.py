#!/usr/bin/env python3
"""
背离检测Prometheus指标导出器
"""

import time
import sys
from pathlib import Path
from prometheus_client import Counter, Histogram, Summary, Info, start_http_server
import threading
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.ofi_cvd_divergence import DivergenceDetector, DivergenceConfig


class DivergenceMetricsExporter:
    """背离检测指标导出器"""
    
    def __init__(self, port: int = 8003):
        self.port = port
        
        # 定义Prometheus指标
        self.events_total = Counter(
            'divergence_events_total',
            'Total number of divergence events detected',
            ['source', 'side', 'kind']
        )
        
        self.detection_latency = Histogram(
            'divergence_detection_latency_seconds',
            'Time taken to detect divergence events',
            ['source'],
            buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        )
        
        self.score_bucket = Histogram(
            'divergence_score_bucket',
            'Distribution of divergence scores',
            ['source'],
            buckets=[0, 20, 40, 60, 70, 80, 85, 90, 95, 100]
        )
        
        self.pairing_gap = Histogram(
            'divergence_pairing_gap_bars',
            'Gap between pivot pairs in bars',
            ['source'],
            buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        )
        
        self.forward_return = Summary(
            'divergence_forward_return',
            'Forward returns after divergence events',
            ['horizon', 'source']
        )
        
        self.config_info = Info(
            'divergence_active_config_info',
            'Current active configuration parameters'
        )
        
        # 创建检测器
        self.config = DivergenceConfig()
        self.detector = DivergenceDetector(self.config)
        
        # 设置配置信息
        self.config_info.info({
            'swing_L': str(self.config.swing_L),
            'z_hi': str(self.config.z_hi),
            'z_mid': str(self.config.z_mid),
            'version': 'v1.0'
        })
    
    def start_server(self):
        """启动Prometheus服务器"""
        start_http_server(self.port)
        print(f"🚀 Prometheus指标服务器已启动: http://localhost:{self.port}/metrics")
    
    def simulate_events(self):
        """模拟事件生成（用于测试）"""
        import random
        import numpy as np
        
        sources = ['OFI', 'CVD', 'FUSION']
        sides = ['bull', 'bear']
        kinds = ['regular', 'hidden']
        
        while True:
            # 模拟检测延迟
            start_time = time.time()
            
            # 模拟检测过程
            time.sleep(random.uniform(0.001, 0.005))
            
            # 记录延迟
            source = random.choice(sources)
            self.detection_latency.labels(source=source).observe(time.time() - start_time)
            
            # 模拟事件生成
            if random.random() < 0.1:  # 10%概率生成事件
                side = random.choice(sides)
                kind = random.choice(kinds)
                score = random.uniform(0, 100)
                
                # 记录事件
                self.events_total.labels(
                    source=source,
                    side=side,
                    kind=kind
                ).inc()
                
                # 记录分数
                self.score_bucket.labels(source=source).observe(score)
                
                # 记录配对间隔
                gap = random.randint(1, 100)
                self.pairing_gap.labels(source=source).observe(gap)
                
                # 记录前瞻收益
                horizon = random.choice(['10', '20'])
                return_val = random.uniform(-0.05, 0.05)
                self.forward_return.labels(
                    horizon=horizon,
                    source=source
                ).observe(return_val)
            
            time.sleep(1)  # 每秒检查一次


def main():
    exporter = DivergenceMetricsExporter()
    
    # 启动服务器
    exporter.start_server()
    
    # 启动模拟线程
    simulation_thread = threading.Thread(target=exporter.simulate_events)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 指标导出器已停止")


if __name__ == "__main__":
    main()
