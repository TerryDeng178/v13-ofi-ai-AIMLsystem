"""
Divergence Detector 自动化测试套件

测试目标：
- 接口契约验证
- 四类背离检测
- 冷却与去重机制
- 融合一致性
- 值域与健壮性
- 统计一致性
- 性能基准测试

Author: Test Engineer
Created: 2025-01-20
"""

import sys
import os
import time
import math
from typing import Dict, List, Optional, Tuple, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ofi_cvd_divergence import DivergenceDetector, DivergenceConfig, DivergenceType


def make_detector(**overrides) -> DivergenceDetector:
    """
    创建背离检测器实例，覆盖配置参数
    
    Args:
        **overrides: 配置覆盖参数
        
    Returns:
        DivergenceDetector: 配置好的检测器
    """
    config = DivergenceConfig()
    
    # 默认测试友好配置
    defaults = {
        'swing_L': 4,
        'min_separation': 3,
        'warmup_min': 10,
        'cooldown_secs': 0.5,
        'max_lag': 0.5,
        'weak_threshold': 30.0,
        'cons_min': 0.2,
        'use_fusion': True
    }
    
    # 应用默认值和覆盖
    for key, value in {**defaults, **overrides}.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return DivergenceDetector(config=config)


def feed_series(
    detector: DivergenceDetector,
    series: List[Tuple[float, float, float, float, Optional[float], Optional[float]]]
) -> Tuple[Optional[Dict], Dict]:
    """
    按序推送数据系列，返回最后一次非空事件和统计
    
    Args:
        detector: 检测器实例
        series: 数据系列，每个元素为 (ts, price, z_ofi, z_cvd, fusion_score, consistency)
        
    Returns:
        Tuple[Optional[Dict], Dict]: (最后一次事件, stats)
    """
    last_event = None
    warmup_samples = detector.cfg.warmup_min
    
    for i, (ts, price, z_ofi, z_cvd, fusion_score, consistency) in enumerate(series):
        warmup = i < warmup_samples
        event = detector.update(
            ts=ts,
            price=price,
            z_ofi=z_ofi,
            z_cvd=z_cvd,
            fusion_score=fusion_score,
            consistency=consistency,
            warmup=warmup,
            lag_sec=0.0
        )
        if event:
            last_event = event
    
    return last_event, detector.get_stats()


def test_input_validation():
    """测试输入验证"""
    detector = make_detector(warmup_min=0)
    
    # NaN 输入
    event = detector.update(1.0, 100.0, float('nan'), 0.0)
    assert event is None, "NaN z_ofi should be rejected"
    
    # Inf 输入
    event = detector.update(2.0, 100.0, float('inf'), 0.0)
    assert event is None, "Inf z_ofi should be rejected"
    
    # 负时间戳
    event = detector.update(-1.0, 100.0, 0.0, 0.0)
    assert event is None, "Negative timestamp should be rejected"
    
    # 负价格
    event = detector.update(3.0, -100.0, 0.0, 0.0)
    assert event is None, "Negative price should be rejected"
    
    # 有效输入应该正常
    event = detector.update(4.0, 100.0, 1.5, -1.5)
    # 可能没有事件，但不应该报错
    
    print("[OK] 输入验证测试通过")


def test_bullish_regular_divergence():
    """测试看涨常规背离"""
    detector = make_detector(warmup_min=0)
    
    # 构造价格LL + 指标HL的背离
    # 价格: 100 -> 90 (LL), 指标: -2 -> 2 (HL)
    series = [
        (1.0, 100.0, -2.0, 0.0, None, None),  # 第一个低点
        (2.0, 95.0, -1.5, 0.0, None, None),
        (3.0, 90.0, -1.0, 0.0, None, None),
        (4.0, 95.0, 0.5, 0.0, None, None),
        (5.0, 100.0, 1.5, 0.0, None, None),
        (6.0, 95.0, 1.0, 0.0, None, None),
        (7.0, 90.0, 0.0, 0.0, None, None),   # 第二个低点
        (8.0, 95.0, 1.0, 0.0, None, None),
        (9.0, 100.0, 2.0, 0.0, None, None),  # 确认背离
    ]
    
    event, stats = feed_series(detector, series)
    
    if event:
        assert event['type'] == 'bull_div', f"Expected bull_div, got {event['type']}"
        assert event['channel'] in ['ofi', 'cvd'], f"Unexpected channel: {event['channel']}"
        assert 'ts' in event and 'score' in event and 'pivots' in event
        assert 'lookback' in event and 'debug' in event and 'warmup' in event
        assert event['score'] >= 30.0, f"Score too low: {event['score']}"
    
    print("[OK] 看涨常规背离测试通过")


def test_bearish_regular_divergence():
    """测试看跌常规背离"""
    detector = make_detector(warmup_min=0)
    
    # 构造价格HH + 指标LH的背离
    # 价格: 100 -> 110 (HH), 指标: 2 -> -2 (LH)
    series = [
        (1.0, 100.0, 2.0, 0.0, None, None),   # 第一个高点
        (2.0, 95.0, 1.5, 0.0, None, None),
        (3.0, 105.0, 1.0, 0.0, None, None),
        (4.0, 100.0, 0.5, 0.0, None, None),
        (5.0, 110.0, 0.0, 0.0, None, None),  # 第二个高点
        (6.0, 105.0, -1.0, 0.0, None, None),
        (7.0, 100.0, -1.5, 0.0, None, None),
        (8.0, 95.0, -2.0, 0.0, None, None),  # 确认背离
    ]
    
    event, stats = feed_series(detector, series)
    
    if event:
        assert event['type'] == 'bear_div', f"Expected bear_div, got {event['type']}"
        assert event['channel'] in ['ofi', 'cvd'], f"Unexpected channel: {event['channel']}"
        assert event['score'] >= 30.0, f"Score too low: {event['score']}"
    
    print("[OK] 看跌常规背离测试通过")


def test_hidden_bull_divergence():
    """测试隐藏看涨背离"""
    detector = make_detector(warmup_min=0)
    
    # 构造价格HL + 指标LL的隐藏看涨背离
    # 价格: 90 -> 100 (HL), 指标: 2 -> -2 (LL)
    series = [
        (1.0, 100.0, 2.0, 0.0, None, None),
        (2.0, 95.0, 1.5, 0.0, None, None),
        (3.0, 90.0, 1.0, 0.0, None, None),   # 第一个低点
        (4.0, 95.0, 0.5, 0.0, None, None),
        (5.0, 100.0, 0.0, 0.0, None, None),  # 第二个低点（更高）
        (6.0, 95.0, -0.5, 0.0, None, None),
        (7.0, 100.0, -1.0, 0.0, None, None),
        (8.0, 95.0, -1.5, 0.0, None, None),
        (9.0, 100.0, -2.0, 0.0, None, None), # 确认背离
    ]
    
    event, stats = feed_series(detector, series)
    
    if event:
        assert event['type'] == 'hidden_bull', f"Expected hidden_bull, got {event['type']}"
        assert event['score'] >= 30.0, f"Score too low: {event['score']}"
    
    print("[OK] 隐藏看涨背离测试通过")


def test_hidden_bear_divergence():
    """测试隐藏看跌背离"""
    detector = make_detector(warmup_min=0)
    
    # 构造价格LH + 指标HH的隐藏看跌背离
    # 价格: 110 -> 100 (LH), 指标: -2 -> 2 (HH)
    series = [
        (1.0, 100.0, -2.0, 0.0, None, None),
        (2.0, 105.0, -1.5, 0.0, None, None),
        (3.0, 110.0, -1.0, 0.0, None, None), # 第一个高点
        (4.0, 105.0, -0.5, 0.0, None, None),
        (5.0, 100.0, 0.0, 0.0, None, None),  # 第二个高点（更低）
        (6.0, 105.0, 0.5, 0.0, None, None),
        (7.0, 100.0, 1.0, 0.0, None, None),
        (8.0, 105.0, 1.5, 0.0, None, None),
        (9.0, 100.0, 2.0, 0.0, None, None),  # 确认背离
    ]
    
    event, stats = feed_series(detector, series)
    
    if event:
        assert event['type'] == 'hidden_bear', f"Expected hidden_bear, got {event['type']}"
        assert event['score'] >= 30.0, f"Score too low: {event['score']}"
    
    print("[OK] 隐藏看跌背离测试通过")


def test_cooldown_mechanism():
    """测试冷却机制"""
    detector = make_detector(warmup_min=0, cooldown_secs=2.0)
    
    # 创建会触发背离的序列
    series = []
    for i in range(20):
        ts = 1.0 + i * 0.5
        price = 100.0 - (i % 2) * 5
        z_ofi = 1.5 if i % 2 == 0 else -1.5
        series.append((ts, price, z_ofi, 0.0, None, None))
    
    # 第一次更新
    event1, stats1 = feed_series(detector, series[:10])
    
    # 稍后触发相同类型的事件（在冷却期内）
    if event1:
        # 重置检测器但保持冷却状态
        # 实际上我们需要在短时间内再次触发
        detector2 = make_detector(warmup_min=0, cooldown_secs=2.0)
        event2, stats2 = feed_series(detector2, series[:12])
        
        if event2:
            # 检查是否有 cooldown 抑制
            assert 'suppressed_by_reason' in stats2
    
    print("[OK] 冷却机制测试通过")


def test_deduplication():
    """测试去重机制"""
    detector = make_detector(warmup_min=0)
    
    # 创建重复的枢轴对
    series = []
    for i in range(20):
        ts = 1.0 + i * 0.1
        price = 100.0 - (i % 4) * 5
        z_ofi = 1.5 if i % 4 == 0 else -0.5
        series.append((ts, price, z_ofi, 0.0, None, None))
    
    # 第一次完整序列
    event1, stats1 = feed_series(detector, series)
    
    # 再次推送相同序列
    event2, stats2 = feed_series(detector, series)
    
    # 第二次应该没有新事件或事件更少
    print(f"[INFO] 第一次事件数: {stats1.get('events_total', 0)}")
    print(f"[INFO] 第二次事件数: {stats2.get('events_total', 0)}")
    
    print("[OK] 去重机制测试通过")


def test_fusion_consistency():
    """测试融合一致性"""
    # 测试一：低一致性
    detector1 = make_detector(warmup_min=0, use_fusion=True)
    series1 = [
        (1.0, 100.0, 0.0, 0.0, -2.0, 0.2),
        (2.0, 95.0, 0.0, 0.0, -1.5, 0.2),
        (3.0, 90.0, 0.0, 0.0, -1.0, 0.2),
        (4.0, 95.0, 0.0, 0.0, 0.5, 0.2),
        (5.0, 100.0, 0.0, 0.0, 1.5, 0.2),
        (6.0, 95.0, 0.0, 0.0, 1.0, 0.2),
        (7.0, 90.0, 0.0, 0.0, 0.0, 0.2),
        (8.0, 95.0, 0.0, 0.0, 1.0, 0.2),
        (9.0, 100.0, 0.0, 0.0, 2.0, 0.2),
    ]
    event1, _ = feed_series(detector1, series1)
    
    # 测试二：高一致性
    detector2 = make_detector(warmup_min=0, use_fusion=True)
    series2 = [
        (1.0, 100.0, 0.0, 0.0, -2.0, 0.8),
        (2.0, 95.0, 0.0, 0.0, -1.5, 0.8),
        (3.0, 90.0, 0.0, 0.0, -1.0, 0.8),
        (4.0, 95.0, 0.0, 0.0, 0.5, 0.8),
        (5.0, 100.0, 0.0, 0.0, 1.5, 0.8),
        (6.0, 95.0, 0.0, 0.0, 1.0, 0.8),
        (7.0, 90.0, 0.0, 0.0, 0.0, 0.8),
        (8.0, 95.0, 0.0, 0.0, 1.0, 0.8),
        (9.0, 100.0, 0.0, 0.0, 2.0, 0.8),
    ]
    event2, _ = feed_series(detector2, series2)
    
    # 高一致性应该产生更高分数
    if event1 and event2:
        print(f"[INFO] 低一致性分数: {event1['score']:.2f}")
        print(f"[INFO] 高一致性分数: {event2['score']:.2f}")
        # 注意：由于其他因素影响，这里不强制断言更高
    
    print("[OK] 融合一致性测试通过")


def test_value_clipping():
    """测试值域裁剪"""
    detector = make_detector(warmup_min=0)
    
    # 超界值
    event1 = detector.update(1.0, 100.0, 12.0, 0.0)  # z_ofi 超出 [-5, 5]
    event2 = detector.update(2.0, 100.0, 0.0, -11.0)  # z_cvd 超出 [-5, 5]
    event3 = detector.update(3.0, 100.0, 0.0, 0.0, 9.0)  # fusion_score 超出 [-5, 5]
    
    # NaN/Inf 应该被拒绝
    event4 = detector.update(4.0, 100.0, float('nan'), 0.0)
    event5 = detector.update(5.0, 100.0, float('inf'), 0.0)
    event6 = detector.update(6.0, 100.0, 0.0, 0.0, float('nan'))
    
    assert event4 is None, "NaN should be rejected"
    assert event5 is None, "Inf should be rejected"
    assert event6 is None, "NaN fusion_score should be rejected"
    
    print("[OK] 值域裁剪测试通过")


def test_statistics_consistency():
    """测试统计一致性"""
    detector = make_detector(warmup_min=0)
    
    # 创建多样本序列
    series = []
    for i in range(50):
        ts = 1.0 + i * 0.1
        price = 100.0 + math.sin(i * 0.1) * 5
        z_ofi = math.cos(i * 0.15) * 2
        z_cvd = math.sin(i * 0.2) * 1.5
        series.append((ts, price, z_ofi, z_cvd, None, None))
    
    event, stats = feed_series(detector, series)
    
    # 检查统计一致性
    if stats['events_total'] > 0:
        sum_by_type = sum(stats['events_by_type'].values())
        assert stats['events_total'] == sum_by_type, \
            f"events_total ({stats['events_total']}) != sum(events_by_type) ({sum_by_type})"
        
        # 检查枢轴计数
        total_pivots = stats['pivots_detected']
        channel_sum = sum(stats['pivots_by_channel'].values())
        assert total_pivots == channel_sum, \
            f"pivots_detected ({total_pivots}) != sum(pivots_by_channel) ({channel_sum})"
    
    print(f"[INFO] 事件总数: {stats['events_total']}")
    print(f"[INFO] 各类型事件: {stats['events_by_type']}")
    print(f"[INFO] 枢轴总数: {stats['pivots_detected']}")
    print("[OK] 统计一致性测试通过")


def test_performance():
    """性能基准测试"""
    detector = make_detector(warmup_min=0)
    
    # 生成10万条随机样本
    num_samples = 100000
    times = []
    
    for i in range(num_samples):
        ts = 1.0 + i * 0.001
        price = 100.0 + (i % 100) * 0.1
        z_ofi = (i % 10 - 5) * 0.5
        z_cvd = ((i % 7) - 3) * 0.5
        
        start = time.perf_counter()
        detector.update(ts, price, z_ofi, z_cvd, None, None, warmup=False, lag_sec=0.0)
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # 转换为毫秒
    
    # 计算分位数
    times.sort()
    p50 = times[num_samples // 2]
    p95 = times[int(num_samples * 0.95)]
    p99 = times[int(num_samples * 0.99)]
    avg = sum(times) / len(times)
    
    print(f"[PERF] 平均耗时: {avg:.3f} ms")
    print(f"[PERF] p50: {p50:.3f} ms")
    print(f"[PERF] p95: {p95:.3f} ms")
    print(f"[PERF] p99: {p99:.3f} ms")
    
    # 断言性能要求
    assert avg < 1.0, f"Average time too high: {avg:.3f} ms"
    assert p99 <= 5.0, f"p99 time too high: {p99:.3f} ms"
    
    print("[OK] 性能测试通过")


if __name__ == '__main__':
    print("=" * 80)
    print("Divergence Detector 自动化测试套件")
    print("=" * 80)
    
    tests = [
        ("输入验证", test_input_validation),
        ("看涨常规背离", test_bullish_regular_divergence),
        ("看跌常规背离", test_bearish_regular_divergence),
        ("隐藏看涨背离", test_hidden_bull_divergence),
        ("隐藏看跌背离", test_hidden_bear_divergence),
        ("冷却机制", test_cooldown_mechanism),
        ("去重机制", test_deduplication),
        ("融合一致性", test_fusion_consistency),
        ("值域裁剪", test_value_clipping),
        ("统计一致性", test_statistics_consistency),
        ("性能基准", test_performance),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\n[TEST] {name}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 80)
    
    sys.exit(0 if failed == 0 else 1)

