#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出配置相关指标到 Prometheus 格式

指标：
- config_fingerprint{service=...}
- legacy_conflict_total{key=...}
- deprecation_warning_total{key=...}
- reload_latency_ms
- reload_qps
- reload_success_ratio
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config.enhanced_config_loader import EnhancedConfigLoader
except ImportError:
    print("# ERROR: EnhancedConfigLoader not available, falling back to base loader", file=sys.stderr)
    from config.unified_config_loader import UnifiedConfigLoader as EnhancedConfigLoader


def export_prometheus_metrics():
    """导出 Prometheus 格式的 metrics"""
    # 禁用生产环境护栏（测试/导出环境）
    loader = EnhancedConfigLoader(
        enable_observability=True,
        enable_production_guard=False  # 测试时禁用
    )
    
    # 如果使用增强版加载器，获取指标
    if hasattr(loader, 'get_metrics'):
        metrics = loader.get_metrics()
    else:
        # 降级到基础指标
        metrics = {
            "config_fingerprint": {
                "value": "unknown",
                "labels": {"service": "v13_ofi_system"}
            }
        }
    
    print("# HELP config_fingerprint Configuration fingerprint (SHA256 hash)")
    print("# TYPE config_fingerprint gauge")
    fp_label = ",".join([f'{k}="{v}"' for k, v in metrics["config_fingerprint"]["labels"].items()])
    
    # 清洗指纹：确保只包含十六进制字符 [0-9a-f]
    fingerprint = metrics["config_fingerprint"]["value"]
    if isinstance(fingerprint, str):
        # 移除非十六进制字符
        import re
        fingerprint_clean = re.sub(r'[^0-9a-f]', '', fingerprint.lower())
        if len(fingerprint_clean) != len(fingerprint):
            print(f"# WARNING: Fingerprint contained non-hex characters, cleaned: {fingerprint} -> {fingerprint_clean}", file=sys.stderr)
    else:
        fingerprint_clean = str(fingerprint)
    
    print(f'config_fingerprint{{{fp_label}}} "{fingerprint_clean}"')
    
    # Legacy conflict metrics
    print("# HELP legacy_conflict_total Total number of legacy key conflicts detected")
    print("# TYPE legacy_conflict_total counter")
    for key, count in metrics.get("legacy_conflict_total", {}).items():
        print(f'legacy_conflict_total{{key="{key}"}} {count}')
    
    # Deprecation warning metrics
    print("# HELP deprecation_warning_total Total number of deprecation warnings issued")
    print("# TYPE deprecation_warning_total counter")
    for key, count in metrics.get("deprecation_warning_total", {}).items():
        print(f'deprecation_warning_total{{key="{key}"}} {count}')
    
    # Reload metrics
    if "reload_total" in metrics:
        print("# HELP reload_total Total number of reload attempts")
        print("# TYPE reload_total counter")
        print(f"reload_total {metrics['reload_total']}")
        
        print("# HELP reload_success Number of successful reloads")
        print("# TYPE reload_success counter")
        print(f"reload_success {metrics['reload_success']}")
        
        print("# HELP reload_failed Number of failed reloads")
        print("# TYPE reload_failed counter")
        print(f"reload_failed {metrics.get('reload_failed', 0)}")
        
        print("# HELP reload_throttled Number of throttled reloads")
        print("# TYPE reload_throttled counter")
        print(f"reload_throttled {metrics.get('reload_throttled', 0)}")
        
        print("# HELP reload_qps Reload rate (reloads per second)")
        print("# TYPE reload_qps gauge")
        print(f"reload_qps {metrics.get('reload_qps', 0.0)}")
        
        print("# HELP reload_success_ratio Ratio of successful reloads")
        print("# TYPE reload_success_ratio gauge")
        print(f"reload_success_ratio {metrics.get('reload_success_ratio', 0.0)}")
    
    # Reload latency metrics
    for metric_name in ["reload_latency_p50_ms", "reload_latency_p95_ms", "reload_latency_p99_ms"]:
        if metric_name in metrics:
            print(f"# HELP {metric_name} Reload latency percentile ({metric_name})")
            print(f"# TYPE {metric_name} gauge")
            print(f"{metric_name} {metrics[metric_name]}")


if __name__ == "__main__":
    export_prometheus_metrics()

