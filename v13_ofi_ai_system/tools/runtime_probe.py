#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行时探针工具：冒烟测试 + 热更新抗抖测试
- 60s 冒烟测试（检查 ERROR、异常栈）
- 5× 连续热更新抗抖（验证 reload 无异常、无半配置状态）
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.unified_config_loader import UnifiedConfigLoader


class RuntimeProbe:
    """运行时探针"""
    
    def __init__(self):
        self.loader = UnifiedConfigLoader()
        self.error_logs = deque(maxlen=1000)
        self.reload_times = []
        self.reload_latencies = []
        self.start_time = time.time()
    
    def smoke_test(self, duration_secs: int = 60) -> Dict[str, Any]:
        """
        冒烟测试：运行指定时长，检查 ERROR 和异常栈
        
        Args:
            duration_secs: 测试时长（秒）
        
        Returns:
            测试结果字典
        """
        print(f"[SMOKE] 开始 {duration_secs}s 冒烟测试...")
        
        start_time = time.time()
        errors = []
        tracebacks = []
        
        # 模拟运行期间检查配置加载
        check_interval = 5  # 每5秒检查一次
        checks = 0
        
        while time.time() - start_time < duration_secs:
            try:
                # 检查配置加载是否正常
                config = self.loader.get()
                if not isinstance(config, dict):
                    errors.append(f"配置加载异常：返回类型错误 {type(config)}")
                
                # 检查关键配置是否存在
                required_keys = [
                    "fusion_metrics",
                    "divergence_detection",
                    "strategy_mode",
                    "monitoring"
                ]
                missing_keys = [k for k in required_keys if k not in config]
                if missing_keys:
                    errors.append(f"缺少关键配置键: {missing_keys}")
                
                checks += 1
                time.sleep(check_interval)
                
            except Exception as e:
                tracebacks.append({
                    "time": time.time() - start_time,
                    "error": str(e),
                    "type": type(e).__name__
                })
                errors.append(f"配置检查异常: {e}")
        
        elapsed = time.time() - start_time
        
        result = {
            "duration_secs": elapsed,
            "checks_count": checks,
            "errors": errors,
            "tracebacks": tracebacks,
            "error_count": len(errors),
            "traceback_count": len(tracebacks),
            "passed": len(errors) == 0 and len(tracebacks) == 0
        }
        
        print(f"[SMOKE] 冒烟测试完成: {checks} 次检查, {len(errors)} 个错误, {len(tracebacks)} 个异常栈")
        
        return result
    
    def stress_reload(self, count: int = 5) -> Dict[str, Any]:
        """
        压力热更新测试：连续 reload 指定次数，验证抗抖
        
        Args:
            count: 连续 reload 次数
        
        Returns:
            测试结果字典
        """
        print(f"[STRESS] 开始 {count}× 连续热更新测试...")
        
        reload_results = []
        latencies = []
        errors = []
        half_config_states = []
        
        for i in range(count):
            try:
                # 记录 reload 前配置快照
                config_before = self.loader.get().get("fusion_metrics", {}).get("thresholds", {})
                fuse_buy_before = config_before.get("fuse_buy", None)
                
                # 执行 reload
                reload_start = time.time()
                self.loader.reload()
                reload_latency = (time.time() - reload_start) * 1000  # 毫秒
                latencies.append(reload_latency)
                
                # 记录 reload 后配置
                config_after = self.loader.get().get("fusion_metrics", {}).get("thresholds", {})
                fuse_buy_after = config_after.get("fuse_buy", None)
                
                # 检查是否有半配置状态（配置在 reload 过程中不一致）
                if fuse_buy_before is None or fuse_buy_after is None:
                    half_config_states.append({
                        "reload_index": i,
                        "issue": "配置值在 reload 过程中丢失",
                        "before": fuse_buy_before,
                        "after": fuse_buy_after
                    })
                
                # 等待短暂时间再继续下一次 reload（模拟真实场景）
                if i < count - 1:
                    time.sleep(0.1)
                
                reload_results.append({
                    "index": i,
                    "latency_ms": reload_latency,
                    "success": True
                })
                
            except Exception as e:
                errors.append({
                    "reload_index": i,
                    "error": str(e),
                    "type": type(e).__name__
                })
                reload_results.append({
                    "index": i,
                    "latency_ms": None,
                    "success": False
                })
        
        # 计算时延分位数
        sorted_latencies = sorted([l for l in latencies if l is not None])
        n = len(sorted_latencies)
        
        def percentile(values, p):
            if not values:
                return None
            k = (len(values) - 1) * p
            f = int(k)
            c = k - f
            if f + 1 < len(values):
                return values[f] + c * (values[f + 1] - values[f])
            return values[f]
        
        result = {
            "reload_count": count,
            "reload_results": reload_results,
            "latencies_ms": latencies,
            "reload_latency_p50_ms": percentile(sorted_latencies, 0.50) if sorted_latencies else None,
            "reload_latency_p95_ms": percentile(sorted_latencies, 0.95) if sorted_latencies else None,
            "reload_latency_p99_ms": percentile(sorted_latencies, 0.99) if sorted_latencies else None,
            "errors": errors,
            "half_config_states": half_config_states,
            "error_count": len(errors),
            "half_config_count": len(half_config_states),
            "passed": len(errors) == 0 and len(half_config_states) == 0
        }
        
        print(f"[STRESS] 热更新测试完成: {count} 次 reload, {len(errors)} 个错误, {len(half_config_states)} 个半配置状态")
        if sorted_latencies:
            print(f"[STRESS] 时延分位: p50={result['reload_latency_p50_ms']:.2f}ms, "
                  f"p95={result['reload_latency_p95_ms']:.2f}ms, "
                  f"p99={result['reload_latency_p99_ms']:.2f}ms")
        
        return result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行时探针：冒烟测试 + 热更新抗抖")
    parser.add_argument(
        "--smoke-secs",
        type=int,
        default=60,
        help="冒烟测试时长（秒，默认: 60）"
    )
    parser.add_argument(
        "--stress-reload",
        type=int,
        default=5,
        help="热更新压力测试次数（默认: 5）"
    )
    
    args = parser.parse_args()
    
    probe = RuntimeProbe()
    
    # 执行冒烟测试
    smoke_result = probe.smoke_test(args.smoke_secs)
    
    # 执行热更新压力测试
    reload_result = probe.stress_reload(args.stress_reload)
    
    # 汇总结果
    overall_result = {
        "smoke_test": smoke_result,
        "stress_reload": reload_result,
        "overall_pass": smoke_result["passed"] and reload_result["passed"],
        "timestamp": time.time()
    }
    
    # 输出到 reports/ 目录
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "runtime_probe_report.json"
    
    try:
        with open(report_path, "w", encoding="utf-8", newline="") as f:
            json.dump(overall_result, f, indent=2, ensure_ascii=False)
        print(f"\n[REPORT] 报告已保存到: {report_path}")
    except Exception as e:
        print(f"\n[WARN] 无法写入报告文件: {e}", file=sys.stderr)
    
    # 退出码
    sys.exit(0 if overall_result["overall_pass"] else 1)


if __name__ == "__main__":
    main()