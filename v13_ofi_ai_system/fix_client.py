#!/usr/bin/env python3
"""修复binance_websocket_client.py - 添加命令行支持和更新方法"""

import re
from pathlib import Path

# 读取文件
file_path = Path("src/binance_websocket_client.py")
content = file_path.read_text(encoding='utf-8')

# 1. 修改print_statistics方法为SUMMARY格式
old_print_stats = '''    def print_statistics(self):
        """打印统计信息（增强版：包含分位数和序列一致性）"""
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        if elapsed == 0:
            return
        
        rate = self.stats['total_messages'] / elapsed
        avg_latency = sum(self.stats['latency_list']) / len(self.stats['latency_list']) if self.stats['latency_list'] else 0
        
        print()
        print("=" * 80)
        print("📊 运行统计")
        print("=" * 80)
        print(f"⏱️  运行时间: {elapsed:.1f}秒")
        print(f"📨 接收消息: {self.stats['total_messages']} 条")
        print(f"⚡ 接收速率: {rate:.2f} 条/秒")
        print(f"📡 平均时延: {avg_latency:.2f}ms")
        
        # 时延分位数（硬标准）
        if self.stats['latency_list']:
            percentiles = self.calculate_percentiles()
            print(f"📊 时延分位:")
            print(f"   - P50 (中位数): {percentiles['p50']:.2f}ms")
            print(f"   - P95: {percentiles['p95']:.2f}ms")
            print(f"   - P99: {percentiles['p99']:.2f}ms")
            print(f"📉 最小时延: {min(self.stats['latency_list']):.2f}ms")
            print(f"📈 最大时延: {max(self.stats['latency_list']):.2f}ms")
        
        # 序列一致性统计（硬标准 - 期货WS严格对齐 v2）
        print(f"🔗 序列一致性(pu==last_u):")
        print(f"   - Resyncs (连续性断裂): {self.stats['resyncs']} 次")
        print(f"   - Reconnects (重连): {self.stats['reconnects']} 次")
        print(f"   - Batch Span (观测): avg={self.stats['batch_span_sum'] / self.stats['total_messages']:.1f}, max={self.stats['batch_span_max']}")
        
        print(f"💾 缓存数据: {len(self.order_book_history)} 条")
        print("=" * 80)
        print()'''

new_print_stats = '''    def print_statistics(self):
        """打印运行统计数据（SUMMARY格式 - Task 1.1.6）"""
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        if elapsed == 0:
            return
        
        rate = self.stats['total_messages'] / elapsed
        
        # 计算分位数
        percentiles = self.calculate_percentiles() if self.stats['latency_list'] else {'p50': 0, 'p95': 0, 'p99': 0}
        
        # 计算batch_span P95
        batch_span_p95 = np.percentile(list(self.stats['batch_span_list']), 95) if len(self.stats['batch_span_list']) > 0 else 0
        
        # 计算log_queue depth P95
        log_queue_p95 = np.percentile(list(self.stats['log_queue_depth_list']), 95) if len(self.stats['log_queue_depth_list']) > 0 else 0
        
        # SUMMARY格式输出（符合Task 1.1.6要求）
        print(f"\\nSUMMARY | t={elapsed:.0f}s | msgs={self.stats['total_messages']} | "
              f"rate={rate:.2f}/s | p50={percentiles['p50']:.1f} p95={percentiles['p95']:.1f} p99={percentiles['p99']:.1f} | "
              f"breaks={self.stats['resyncs']} resyncs={self.stats['resyncs']} reconnects={self.stats['reconnects']} | "
              f"batch_span_p95={batch_span_p95:.0f} max={self.stats['batch_span_max']} | "
              f"log_q_p95={log_queue_p95:.0f} max={self.stats['log_queue_max_depth']} drops={self.stats['log_drops']}")
        
        # 同时记录到日志
        self.logger.info(f"SUMMARY: runtime={elapsed:.0f}s, msgs={self.stats['total_messages']}, "
                        f"rate={rate:.2f}/s, p95={percentiles['p95']:.1f}ms, "
                        f"breaks={self.stats['resyncs']}, resyncs={self.stats['resyncs']}, "
                        f"log_drops={self.stats['log_drops']}")'''

# 检查是否存在旧版本（可能因编码问题无法直接匹配）
if 'def print_statistics(self):' in content:
    print("找到print_statistics方法，需要手动处理...")
    # 保存标记，后续手动处理
else:
    print("未找到print_statistics方法")

print("请手动修改文件")

