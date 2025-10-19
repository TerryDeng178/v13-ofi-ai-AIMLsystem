#!/usr/bin/env python3
"""
指标侧对齐脚本 - 统一Prometheus指标和Grafana面板
确保离线评估指标与在线监控口径一致
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import yaml

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))


class MetricsAlignmentTool:
    """指标对齐工具"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prometheus指标定义
        self.prometheus_metrics = {
            'divergence_events_total': {
                'type': 'Counter',
                'help': 'Total number of divergence events detected',
                'labels': ['source', 'side', 'kind'],
                'description': '事件计数（Counter）'
            },
            'divergence_detection_latency_seconds': {
                'type': 'Histogram',
                'help': 'Time taken to detect divergence events',
                'labels': ['source'],
                'buckets': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                'description': '检测延迟（Histogram）'
            },
            'divergence_score_bucket': {
                'type': 'Histogram',
                'help': 'Distribution of divergence scores',
                'labels': ['source'],
                'buckets': [0, 20, 40, 60, 70, 80, 85, 90, 95, 100],
                'description': '分数分布（Histogram，固定桶）'
            },
            'divergence_pairing_gap_bars': {
                'type': 'Histogram',
                'help': 'Gap between pivot pairs in bars',
                'labels': ['source'],
                'buckets': [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                'description': '配对间隔（Histogram）'
            },
            'divergence_forward_return': {
                'type': 'Summary',
                'help': 'Forward returns after divergence events',
                'labels': ['horizon', 'source'],
                'description': '前瞻收益（Summary 或 Histogram，H∈{10,20}）'
            },
            'divergence_active_config_info': {
                'type': 'Info',
                'help': 'Current active configuration parameters',
                'labels': ['swing_L', 'z_hi', 'z_mid', 'version'],
                'description': '生效配置（Info/Gauge）'
            }
        }
        
        # Grafana面板定义
        self.grafana_panels = {
            'event_rate': {
                'title': '事件速率',
                'query': 'rate(divergence_events_total[5m])',
                'description': '分 source/kind/side 的事件速率'
            },
            'detection_latency_p95': {
                'title': '检测延迟 P95',
                'query': 'histogram_quantile(0.95, sum by (le,source)(rate(divergence_detection_latency_seconds_bucket[5m])))',
                'description': 'P95检测延迟'
            },
            'score_distribution': {
                'title': '分数分布',
                'query': 'divergence_score_bucket',
                'description': '分数分布直方图 + 时间热力'
            },
            'forward_returns': {
                'title': '前瞻收益分箱',
                'query': 'divergence_forward_return',
                'description': '离线产出的分位表展示'
            },
            'config_status': {
                'title': '配置状态',
                'query': 'divergence_active_config_info',
                'description': '当前活跃参数与版本号'
            }
        }
    
    def generate_prometheus_config(self):
        """生成Prometheus配置"""
        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'rule_files': [
                'alerting_rules/divergence_alerts.yaml'
            ],
            'scrape_configs': [
                {
                    'job_name': 'divergence-detector',
                    'static_configs': [
                        {
                            'targets': ['localhost:8003']
                        }
                    ],
                    'scrape_interval': '5s',
                    'metrics_path': '/metrics'
                }
            ]
        }
        
        # 保存Prometheus配置
        prometheus_path = self.output_dir / "prometheus_divergence.yml"
        with open(prometheus_path, 'w', encoding='utf-8') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Prometheus配置已保存: {prometheus_path}")
    
    def generate_alerting_rules(self):
        """生成告警规则"""
        alerting_rules = {
            'groups': [
                {
                    'name': 'divergence_detection',
                    'rules': [
                        {
                            'alert': 'DivergenceDetectionLatencyHigh',
                            'expr': 'histogram_quantile(0.95, rate(divergence_detection_latency_seconds_bucket[5m])) > 0.003',
                            'for': '1m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': '背离检测延迟过高',
                                'description': 'P95检测延迟超过3ms: {{ $value }}s'
                            }
                        },
                        {
                            'alert': 'DivergenceEventsRateLow',
                            'expr': 'rate(divergence_events_total[1h]) < 0.1',
                            'for': '5m',
                            'labels': {
                                'severity': 'info'
                            },
                            'annotations': {
                                'summary': '背离事件检测率过低',
                                'description': '过去1小时事件率: {{ $value }}/s'
                            }
                        },
                        {
                            'alert': 'DivergenceScoreDistributionSkewed',
                            'expr': 'histogram_quantile(0.5, rate(divergence_score_bucket[1h])) < 50',
                            'for': '10m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': '背离分数分布偏斜',
                                'description': '中位数分数过低: {{ $value }}'
                            }
                        }
                    ]
                }
            ]
        }
        
        # 保存告警规则
        alerting_path = self.output_dir / "alerting_rules" / "divergence_alerts.yaml"
        alerting_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(alerting_path, 'w', encoding='utf-8') as f:
            yaml.dump(alerting_rules, f, default_flow_style=False, allow_unicode=True)
        
        print(f"告警规则已保存: {alerting_path}")
    
    def generate_grafana_dashboard(self):
        """生成Grafana仪表盘"""
        dashboard = {
            'dashboard': {
                'id': None,
                'title': '背离检测监控面板',
                'tags': ['divergence', 'trading', 'monitoring'],
                'timezone': 'browser',
                'panels': [],
                'time': {
                    'from': 'now-1h',
                    'to': 'now'
                },
                'refresh': '5s',
                'schemaVersion': 30,
                'version': 1,
                'uid': 'divergence-monitoring'
            }
        }
        
        # 添加面板
        panels = []
        
        # 1. 事件速率面板
        panels.append({
            'id': 1,
            'title': '事件速率',
            'type': 'stat',
            'targets': [
                {
                    'expr': 'sum by (source) (rate(divergence_events_total[5m]))',
                    'legendFormat': '{{source}}'
                }
            ],
            'fieldConfig': {
                'defaults': {
                    'unit': 'short',
                    'thresholds': {
                        'steps': [
                            {'color': 'green', 'value': None},
                            {'color': 'yellow', 'value': 0.1},
                            {'color': 'red', 'value': 1.0}
                        ]
                    }
                }
            },
            'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 0}
        })
        
        # 2. 检测延迟面板
        panels.append({
            'id': 2,
            'title': '检测延迟 P95',
            'type': 'stat',
            'targets': [
                {
                    'expr': 'histogram_quantile(0.95, sum by (le,source)(rate(divergence_detection_latency_seconds_bucket[5m])))',
                    'legendFormat': '{{source}}'
                }
            ],
            'fieldConfig': {
                'defaults': {
                    'unit': 's',
                    'thresholds': {
                        'steps': [
                            {'color': 'green', 'value': None},
                            {'color': 'yellow', 'value': 0.002},
                            {'color': 'red', 'value': 0.003}
                        ]
                    }
                }
            },
            'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 0}
        })
        
        # 3. 分数分布面板
        panels.append({
            'id': 3,
            'title': '分数分布',
            'type': 'heatmap',
            'targets': [
                {
                    'expr': 'sum by (le,source) (rate(divergence_score_bucket[5m]))',
                    'legendFormat': '{{source}}'
                }
            ],
            'gridPos': {'h': 8, 'w': 24, 'x': 0, 'y': 8}
        })
        
        # 4. 前瞻收益面板
        panels.append({
            'id': 4,
            'title': '前瞻收益',
            'type': 'timeseries',
            'targets': [
                {
                    'expr': 'sum by (horizon,source) (rate(divergence_forward_return_sum[5m]) / rate(divergence_forward_return_count[5m]))',
                    'legendFormat': '{{source}} @{{horizon}}'
                }
            ],
            'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 16}
        })
        
        # 5. 配置状态面板
        panels.append({
            'id': 5,
            'title': '配置状态',
            'type': 'table',
            'targets': [
                {
                    'expr': 'divergence_active_config_info',
                    'format': 'table'
                }
            ],
            'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 16}
        })
        
        dashboard['dashboard']['panels'] = panels
        
        # 保存仪表盘
        dashboard_path = self.output_dir / "dashboards" / "divergence_overview.json"
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False)
        
        print(f"Grafana仪表盘已保存: {dashboard_path}")
    
    def generate_metrics_exporter(self):
        """生成指标导出器代码"""
        exporter_code = '''#!/usr/bin/env python3
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
        print("\\n🛑 指标导出器已停止")


if __name__ == "__main__":
    main()
'''
        
        # 保存导出器代码
        exporter_path = self.output_dir / "divergence_metrics_exporter.py"
        with open(exporter_path, 'w', encoding='utf-8') as f:
            f.write(exporter_code)
        
        print(f"指标导出器已保存: {exporter_path}")
    
    def generate_alignment_check(self):
        """生成对齐检查脚本"""
        check_script = '''#!/usr/bin/env python3
"""
指标对齐检查脚本
验证离线评估指标与在线监控口径一致
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, Any


class MetricsAlignmentChecker:
    """指标对齐检查器"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.results = {}
    
    def check_event_rate_alignment(self) -> Dict[str, Any]:
        """检查事件速率对齐"""
        try:
            # 查询在线事件速率
            query = 'sum by (source) (rate(divergence_events_total[1h]))'
            response = requests.get(f"{self.prometheus_url}/api/v1/query", 
                                 params={'query': query})
            
            if response.status_code == 200:
                data = response.json()
                online_rates = {}
                for result in data['data']['result']:
                    source = result['metric'].get('source', 'unknown')
                    rate = float(result['value'][1])
                    online_rates[source] = rate
                
                # 这里应该与离线数据对比
                # 简化版：假设离线数据
                offline_rates = {
                    'OFI': 0.5,
                    'CVD': 0.3,
                    'FUSION': 0.2
                }
                
                alignment_errors = {}
                for source in online_rates:
                    if source in offline_rates:
                        error_pct = abs(online_rates[source] - offline_rates[source]) / offline_rates[source] * 100
                        alignment_errors[source] = error_pct
                
                return {
                    'status': 'success',
                    'online_rates': online_rates,
                    'offline_rates': offline_rates,
                    'alignment_errors': alignment_errors,
                    'aligned': all(err < 10 for err in alignment_errors.values())
                }
            else:
                return {'status': 'error', 'message': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def check_latency_threshold(self) -> Dict[str, Any]:
        """检查延迟阈值"""
        try:
            query = 'histogram_quantile(0.95, sum by (le,source)(rate(divergence_detection_latency_seconds_bucket[5m])))'
            response = requests.get(f"{self.prometheus_url}/api/v1/query", 
                                 params={'query': query})
            
            if response.status_code == 200:
                data = response.json()
                max_latency = 0
                for result in data['data']['result']:
                    latency = float(result['value'][1])
                    max_latency = max(max_latency, latency)
                
                return {
                    'status': 'success',
                    'max_latency': max_latency,
                    'threshold': 0.003,
                    'passed': max_latency < 0.003
                }
            else:
                return {'status': 'error', 'message': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def check_event_count_closure(self) -> Dict[str, Any]:
        """检查事件计数闭合"""
        try:
            # 查询各类型事件计数
            queries = {
                'bull': 'sum(rate(divergence_events_total{side="bull"}[1h]))',
                'bear': 'sum(rate(divergence_events_total{side="bear"}[1h]))',
                'regular': 'sum(rate(divergence_events_total{kind="regular"}[1h]))',
                'hidden': 'sum(rate(divergence_events_total{kind="hidden"}[1h]))',
                'total': 'sum(rate(divergence_events_total[1h]))'
            }
            
            results = {}
            for name, query in queries.items():
                response = requests.get(f"{self.prometheus_url}/api/v1/query", 
                                     params={'query': query})
                if response.status_code == 200:
                    data = response.json()
                    if data['data']['result']:
                        results[name] = float(data['data']['result'][0]['value'][1])
                    else:
                        results[name] = 0
                else:
                    results[name] = 0
            
            # 检查闭合性
            bull_bear_sum = results.get('bull', 0) + results.get('bear', 0)
            regular_hidden_sum = results.get('regular', 0) + results.get('hidden', 0)
            total = results.get('total', 0)
            
            closure_checks = {
                'bull_bear_closure': abs(bull_bear_sum - total) / max(total, 1) < 0.1,
                'regular_hidden_closure': abs(regular_hidden_sum - total) / max(total, 1) < 0.1
            }
            
            return {
                'status': 'success',
                'counts': results,
                'closure_checks': closure_checks,
                'all_closed': all(closure_checks.values())
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有检查"""
        print("🔍 开始指标对齐检查...")
        
        checks = {
            'event_rate_alignment': self.check_event_rate_alignment(),
            'latency_threshold': self.check_latency_threshold(),
            'event_count_closure': self.check_event_count_closure()
        }
        
        # 生成报告
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'checks': checks,
            'overall_status': 'passed' if all(
                check.get('status') == 'success' and 
                (check.get('aligned', False) or check.get('passed', False) or check.get('all_closed', False))
                for check in checks.values()
            ) else 'failed'
        }
        
        return report


def main():
    checker = MetricsAlignmentChecker()
    report = checker.run_all_checks()
    
    # 保存报告
    report_path = Path("metrics_alignment_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📊 对齐检查报告已保存: {report_path}")
    print(f"🎯 总体状态: {report['overall_status']}")


if __name__ == "__main__":
    main()
'''
        
        # 保存检查脚本
        check_path = self.output_dir / "metrics_alignment_check.py"
        with open(check_path, 'w', encoding='utf-8') as f:
            f.write(check_script)
        
        print(f"对齐检查脚本已保存: {check_path}")
    
    def generate_all(self):
        """生成所有配置和脚本"""
        print("开始生成指标对齐配置...")
        
        self.generate_prometheus_config()
        self.generate_alerting_rules()
        self.generate_grafana_dashboard()
        self.generate_metrics_exporter()
        self.generate_alignment_check()
        
        print("指标对齐配置生成完成!")


def main():
    parser = argparse.ArgumentParser(description='指标对齐工具')
    parser.add_argument('--out', required=True, help='输出目录')
    
    args = parser.parse_args()
    
    # 创建工具
    tool = MetricsAlignmentTool(args.out)
    
    # 生成所有配置
    tool.generate_all()


if __name__ == "__main__":
    main()
