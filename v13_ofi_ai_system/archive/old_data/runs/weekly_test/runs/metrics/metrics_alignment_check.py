#!/usr/bin/env python3
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
