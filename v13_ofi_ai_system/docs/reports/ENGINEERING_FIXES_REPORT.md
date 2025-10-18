# 工程层面关键修复报告

## 📊 执行摘要

**修复时间**: 2025-10-19 01:12  
**修复目标**: 解决长时间运行队列丢弃问题  
**修复范围**: 工程通道层面（run_realtime_cvd.py）+ 算法层面（real_cvd_calculator.py）

## 🎯 问题诊断

**核心问题**: 不是CVD算法本身的问题，而是长时间运行下的**数据通道/队列策略在代码层面有缺陷**，导致背压没被正确消化，出现高丢弃率→样本失真→Z质量看起来"退化"。

**关键证据**:
- 5分钟测试：0%丢弃率 ✅
- 35分钟测试：40.35%丢弃率 ❌
- Step 1.6配置：已正确加载 ✅
- ID健康/解析/重连：全部为0/绿 ✅

## 🔧 高优先级修复（必须改）

### 1. ✅ 分析模式改为阻塞不丢

**位置**: `ws_consume()`

**问题**: 
```python
if queue.full():
    metrics.queue_dropped += 1
    queue.get_nowait()
    await queue.put(..)
```
这会不断丢旧消息，直接导致长测样本失真。

**修复**:
```python
# 分析模式：阻塞不丢（实时灰度可通过DROP_OLD=true启用丢旧策略）
DROP_OLD = os.getenv("DROP_OLD", "false").lower() == "true"
if DROP_OLD and queue.full():
    metrics.queue_dropped += 1
    try:
        _ = queue.get_nowait()
    except asyncio.QueueEmpty:
        pass
await queue.put((time.time(), msg))  # 分析模式默认阻塞
```

### 2. ✅ 移除undefined的flush_records_batch()调用

**位置**: `processor()`

**问题**: 调用未定义的`flush_records_batch()`，会导致NameError

**修复**: 移除所有`flush_records_batch()`调用，保持原有的集中导出逻辑

### 3. ✅ 修复watermark定时flush分支的时间戳

**位置**: `processor()` 的 `force_flush_timeout` 分支

**问题**: 使用了过期的接收时间戳`ts_recv`，导致延迟统计偏差

**修复**:
```python
record = CVDRecord(
    timestamp=time.time(),  # 使用当前时间，不复用旧的ts_recv
    ...
)
```

### 4. ✅ 修复默认参数为Step 1.6基线

**位置**: `processor()` 中构造 `CVDConfig`

**问题**: 默认值不符合Step 1.6基线
- `z_mode` 默认 "level" → 应为 "delta"
- `scale_mode` 默认 "ewma" → 应为 "hybrid"
- `freeze_min` 默认 50 → 应为 80
- 其他参数也不符合Step 1.6

**修复**:
```python
cfg = CVDConfig(
    z_mode=os.getenv("CVD_Z_MODE", "delta"),  # Step 1.6: delta模式
    freeze_min=int(os.getenv("FREEZE_MIN", "80")),  # Step 1.6: 80
    scale_mode=os.getenv("SCALE_MODE", "hybrid"),  # Step 1.6: hybrid模式
    scale_fast_weight=float(os.getenv("SCALE_FAST_WEIGHT", "0.35")),  # Step 1.6: 0.35
    scale_slow_weight=float(os.getenv("SCALE_SLOW_WEIGHT", "0.65")),  # Step 1.6: 0.65
    mad_multiplier=float(os.getenv("MAD_MULTIPLIER", "1.45")),  # Step 1.6: 1.45
    ...
)
```

## 🔧 中优先级修复（已完成）

### 5. ✅ 添加z_mode到有效配置打印

**位置**: `RealCVDCalculator._print_effective_config()`

**问题**: 有效配置未打印z_mode，难以及时发现误配置

**修复**:
```python
print(f"[CVD] Effective config for {self.symbol}:")
print(f"  Z_MODE={self.cfg.z_mode}")  # 防止误配置
```

### 6. ✅ 混合尺度权重强制归一化

**位置**: `RealCVDCalculator._z_delta()` 和 `_peek_delta_z()`

**问题**: 权重未归一化，若配置错误会改变期望量纲

**修复**:
```python
# 权重归一化：防止配置错误
w_fast = max(0.0, min(1.0, self.cfg.scale_fast_weight))
w_slow = max(0.0, min(1.0, self.cfg.scale_slow_weight))
w_sum = w_fast + w_slow
if w_sum <= 1e-9:
    w_fast, w_slow = 0.5, 0.5
else:
    w_fast, w_slow = w_fast / w_sum, w_slow / w_sum

ewma_mix = (w_fast * self._ewma_abs_fast + 
           w_slow * self._ewma_abs_delta)
```

## 📋 修复文件清单

### run_realtime_cvd.py
1. ✅ 队列策略：分析模式阻塞不丢，实时灰度可选
2. ✅ 移除flush_records_batch()调用
3. ✅ 修复watermark定时flush时间戳
4. ✅ 修复默认参数为Step 1.6基线

### real_cvd_calculator.py
1. ✅ 添加Z_MODE到有效配置打印
2. ✅ 混合尺度权重归一化（_z_delta和_peek_delta_z）

## 🎯 预期效果

修复后的预期效果：

1. **队列丢弃率**: 40.35% → 0% ✅
2. **数据量**: 恢复正常（不再偏少）✅
3. **Z-score质量**: 可以准确测量P(|Z|>2)和P(|Z|>3) ✅
4. **长时间稳定性**: 35-40分钟测试稳定运行 ✅
5. **配置正确性**: 默认值对齐Step 1.6基线 ✅

## 🚀 下一步行动

1. **运行修复版测试**: 35-40分钟干净金测
2. **验证丢弃率**: 目标0%丢弃率
3. **测量Z质量**: P(|Z|>2)≤8%, P(|Z|>3)≤2%
4. **生成最终报告**: 验证修复效果

---
*报告生成时间: 2025-10-19 01:12*  
*修复执行者: V13 OFI+CVD+AI System*
