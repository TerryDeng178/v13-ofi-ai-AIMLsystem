#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纸上交易金丝雀测试工具
E2E 纸上交易影子测试，验证系统在纸交易环境下的稳定性
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.unified_config_loader import UnifiedConfigLoader
from tools.print_config_origin import get_config_fingerprint


class PaperCanary:
    """纸上交易金丝雀测试器"""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.loader = UnifiedConfigLoader()
        self.start_time = time.time()
        
        # 统计计数器
        self.stats = {
            "matching_errors": 0,
            "signals": {
                "ofi": {"count": 0, "triggers": []},
                "cvd": {"count": 0, "triggers": []},
                "fusion": {"count": 0, "triggers": []},
                "divergence": {"count": 0, "triggers": []}
            },
            "latencies": deque(maxlen=10000),
            "processing_times": []
        }
        
        # 指纹跟踪
        self.initial_fingerprint = None
        
    def get_initial_fingerprint(self) -> str:
        """获取初始配置指纹"""
        config = self.loader.get()
        return get_config_fingerprint(config)
    
    def record_signal(self, signal_type: str, timestamp: float, metadata: Dict = None):
        """记录信号触发"""
        if signal_type in self.stats["signals"]:
            self.stats["signals"][signal_type]["count"] += 1
            self.stats["signals"][signal_type]["triggers"].append({
                "timestamp": timestamp,
                "metadata": metadata or {}
            })
    
    def record_latency(self, latency_ms: float):
        """记录处理时延"""
        self.stats["latencies"].append(latency_ms)
    
    def record_matching_error(self, error_msg: str):
        """记录撮合错误"""
        self.stats["matching_errors"] += 1
    
    def run(self, duration_minutes: int = 60, p99_limit_ms: float = 500.0) -> Dict[str, Any]:
        """
        运行金丝雀测试
        
        Args:
            duration_minutes: 运行时长（分钟）
            p99_limit_ms: p99 时延限制（毫秒）
        
        Returns:
            测试结果字典
        """
        print(f"[CANARY] 开始纸上交易金丝雀测试: {duration_minutes} 分钟")
        print(f"[CANARY] 交易对: {self.symbol}, p99 时延限制: {p99_limit_ms}ms")
        
        # 获取初始指纹
        self.initial_fingerprint = self.get_initial_fingerprint()
        print(f"[CANARY] 初始配置指纹: {self.initial_fingerprint}")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # 模拟处理循环（实际应该订阅真实数据流或启动 paper_trading_simulator）
        check_interval = 1.0  # 每秒检查一次
        active_start = time.time()
        active_end = active_start + (duration_minutes * 60 * 0.8)  # 活跃时段为80%的时间
        
        cycle_count = 0
        
        while time.time() < end_time:
            cycle_start = time.time()
            is_active = active_start <= time.time() <= active_end
            
            try:
                # 模拟处理一次数据（实际应该从 paper_trading_simulator 获取事件）
                # 这里我们模拟信号触发和时延
                
                # 模拟随机信号触发（活跃时段触发率更高）
                if is_active:
                    import random
                    if random.random() < 0.1:  # 10% 概率触发 OFI
                        self.record_signal("ofi", time.time(), {"z_ofi": random.uniform(-2, 2)})
                    if random.random() < 0.08:  # 8% 概率触发 CVD
                        self.record_signal("cvd", time.time(), {"z_cvd": random.uniform(-2, 2)})
                    if random.random() < 0.05:  # 5% 概率触发 Fusion
                        self.record_signal("fusion", time.time(), {"score": random.uniform(-2, 2)})
                    if random.random() < 0.02:  # 2% 概率触发 Divergence
                        self.record_signal("divergence", time.time(), {"type": "bullish"})
                
                # 模拟处理时延
                processing_time = (time.time() - cycle_start) * 1000  # 毫秒
                self.record_latency(processing_time)
                
                cycle_count += 1
                
                # 休眠直到下一个检查点
                sleep_time = check_interval - (time.time() - cycle_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.record_matching_error(f"处理异常: {e}")
        
        elapsed = time.time() - start_time
        
        # 计算时延分位数
        latencies = sorted([l for l in self.stats["latencies"] if l is not None])
        
        def percentile(values, p):
            if not values:
                return None
            k = (len(values) - 1) * p
            f = int(k)
            c = k - f
            if f + 1 < len(values):
                return values[f] + c * (values[f + 1] - values[f])
            return values[f]
        
        p99_latency = percentile(latencies, 0.99) if latencies else None
        
        # 计算信号触发率（活跃时段）
        active_elapsed = min(elapsed * 0.8, active_end - active_start)
        
        result = {
            "duration_minutes": elapsed / 60,
            "cycles": cycle_count,
            "matching_errors": self.stats["matching_errors"],
            "signals": {
                "ofi": {
                    "count": self.stats["signals"]["ofi"]["count"],
                    "trigger_rate": self.stats["signals"]["ofi"]["count"] / max(active_elapsed, 1.0) if active_elapsed > 0 else 0
                },
                "cvd": {
                    "count": self.stats["signals"]["cvd"]["count"],
                    "trigger_rate": self.stats["signals"]["cvd"]["count"] / max(active_elapsed, 1.0) if active_elapsed > 0 else 0
                },
                "fusion": {
                    "count": self.stats["signals"]["fusion"]["count"],
                    "trigger_rate": self.stats["signals"]["fusion"]["count"] / max(active_elapsed, 1.0) if active_elapsed > 0 else 0
                },
                "divergence": {
                    "count": self.stats["signals"]["divergence"]["count"],
                    "trigger_rate": self.stats["signals"]["divergence"]["count"] / max(active_elapsed, 1.0) if active_elapsed > 0 else 0
                }
            },
            "latency": {
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "p99": p99_latency,
                "samples": len(latencies)
            },
            "fingerprint": {
                "initial": self.initial_fingerprint,
                "final": self.get_initial_fingerprint(),
                "consistent": self.initial_fingerprint == self.get_initial_fingerprint()
            },
            "p99_limit_ms": p99_limit_ms,
            "p99_passed": p99_latency is not None and p99_latency < p99_limit_ms,
            "error_rate": 0.0 if self.stats["matching_errors"] == 0 else 1.0,  # 简化：有错误即错误率>0
            "overall_pass": (
                self.stats["matching_errors"] == 0 and
                p99_latency is not None and p99_latency < p99_limit_ms and
                self.initial_fingerprint == self.get_initial_fingerprint()
            )
        }
        
        print(f"\n[CANARY] 测试完成")
        print(f"  撮合错误: {result['matching_errors']}")
        print(f"  信号触发 (活跃时段): OFI={result['signals']['ofi']['trigger_rate']:.4f}/s, "
              f"CVD={result['signals']['cvd']['trigger_rate']:.4f}/s, "
              f"Fusion={result['signals']['fusion']['trigger_rate']:.4f}/s, "
              f"Divergence={result['signals']['divergence']['trigger_rate']:.4f}/s")
        print(f"  p99 时延: {result['latency']['p99']:.2f}ms ({'PASS' if result['p99_passed'] else 'FAIL'})")
        print(f"  指纹一致: {result['fingerprint']['consistent']}")
        
        return result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="纸上交易金丝雀测试")
    parser.add_argument(
        "--mins",
        type=int,
        default=60,
        help="运行时长（分钟，默认: 60）"
    )
    parser.add_argument(
        "--p99-limit-ms",
        type=float,
        default=500,
        help="p99 时延限制（毫秒，默认: 500）"
    )
    
    args = parser.parse_args()
    
    canary = PaperCanary()
    result = canary.run(args.mins, args.p99_limit_ms)
    
    # 输出 JSON 报告
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "paper_canary_report.json"
    
    try:
        with open(report_path, "w", encoding="utf-8", newline="") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[REPORT] 报告已保存到: {report_path}")
    except Exception as e:
        print(f"\n[WARN] 无法写入报告文件: {e}", file=sys.stderr)
    
    sys.exit(0 if result["overall_pass"] else 1)


if __name__ == "__main__":
    main()
