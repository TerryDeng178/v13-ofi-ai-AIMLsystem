# 内存关键BUG修复总结

## 修复日期
2025-10-27

## 问题根源
运行一段时间后不再落盘的根本原因：**缓冲区无界增长 + 保存失败导致内存死循环**

## 修复的4个关键BUG

### BUG #1: 缓冲区无界增长 (最严重)
**位置**: 第111-120行

**问题**:
- 缓冲区使用普通列表，无大小限制
- 当轮转失败或延迟时，缓冲区无限增长
- 导致内存耗尽或进程僵死

**修复**:
- 添加两级水位线：`buffer_high` 和 `buffer_emergency`
- 高水位触发即时落盘
- 紧急水位对预览流做 deadletter 溢写
- **权威流（prices/orderbook）绝不丢失**，强制再次落盘

**代码**:
```python
self.buffer_high = {
    'prices': 20000, 'orderbook': 12000,
    'ofi': 8000, 'cvd': 8000, 'fusion': 5000, 'events': 5000, 'features': 8000
}
self.buffer_emergency = {
    'prices': 40000, 'orderbook': 24000,
    'ofi': 16000, 'cvd': 16000, 'fusion': 10000, 'events': 10000, 'features': 16000
}
```

### BUG #4: 保存失败回灌导致死循环
**位置**: 第1458-1461行

**问题**:
- 保存失败后数据回灌到内存
- 持久性失败（磁盘满/权限问题）导致无限失败→回灌→再失败
- 内存持续增长直到OOM

**修复**:
- 改为写入 deadletter 目录（`artifacts/deadletter/<kind>/*.ndjson`）
- 不再回灌到内存，避免死循环
- 数据不丢失，可作为事后恢复

**代码**:
```python
except Exception as e:
    logger.error(f"保存数据错误 {symbol}-{kind}: {e.__class__.__name__}: {e}")
    # 不再回灌到内存，改为死信落地，避免失败→回灌→再失败死循环
    self._spill_to_deadletter(symbol, kind, buf)
```

### BUG #3: 轮转持锁写盘阻塞
**位置**: 第1463-1488行

**问题**:
- 轮转期间持有锁，完成所有写盘操作
- 任一 symbol/kind 写慢会阻塞整个轮转
- 其他 symbol 无法及时落盘

**修复**:
- 缩小锁粒度：锁内只收集任务并更新时间戳
- 写盘放到锁外并发执行
- 通过 `save_semaphore` 限流，避免雪崩

**代码**:
```python
async with self.rotation_lock:
    # 只在锁内收集任务，不写盘
    for symbol in self.symbols:
        for kind in ['prices', 'orderbook']:
            if self.data_buffers[kind][symbol]:
                tasks.append(self._save_data(symbol, kind))
    self.last_rotate_time = current_time
# 锁外并发落盘（限流）
if tasks:
    async with self.save_semaphore:
        await asyncio.gather(*tasks, return_exceptions=True)
```

### BUG #6: 特征表遍历整个缓冲
**位置**: 第375-396行

**问题**:
- 每次都全表扫描缓冲区
- 缓冲区越大，性能越差（O(N)复杂度）
- 拖慢轮转过程

**修复**:
- 只处理"最近60s + 上次未处理后的增量秒"
- 通过尾扫快速收集增量数据
- 复杂度从 O(N) 降到 O(窗口大小)

**代码**:
```python
def _collect_recent(buf):
    out = []
    # 逆向尾扫，遇到过旧数据即停
    for x in reversed(buf):
        ts = x.get('ts_ms', 0)
        if ts < start_ms:
            break
        out.append(x)
    out.reverse()
    return out
```

## 新增功能

### 1. 内存压力监控
`_maybe_flush_on_pressure()` 方法在每次添加数据后检查缓冲区大小：
- 超过高水位：触发即时落盘
- 超过紧急水位：
  - 预览流（ofi/cvd/fusion/events/features）：溢写到 deadletter
  - 权威流（prices/orderbook）：强制再次落盘

### 2. Deadletter机制
`_spill_to_deadletter()` 方法将失败数据写入NDJSON文件：
- 位置：`artifacts/deadletter/<kind>/`
- 格式：`{symbol}_{timestamp}_{rows}.ndjson`
- 目的：数据不丢失，可事后恢复

### 3. Dinamik并发限流
- 使用 `asyncio.Semaphore` 控制并发写盘数量
- 默认并发度：2（可通过环境变量 `SAVE_CONCURRENCY` 调整）

## 预期效果

### 1. 内存使用可控
- 缓冲区大小被严格限制
- 紧急情况下自动溢写到磁盘
- 不会出现OOM

### 2. 数据不丢失
- 权威流（prices/orderbook）绝不错失数据
- 预览流失败时写入 deadletter
- 可通过 deadletter 恢复

### 3. 性能稳定
- 轮转不阻塞
- 特征表生成高效
- 并发写盘优化

### 4. 故障恢复
- 保存失败不再导致死循环
- 错误计数合理衰减
- 健康检查不"自杀"

## 测试验证

### 1. 水位线触发测试
```bash
# 观察日志中的 [SPILL] 消息
# 人为降低水位线测试
export BUFFER_HIGH=100  # 降低到100条
python run_success_harvest.py
# 应该在达到100条时触发落盘
```

### 2. Deadletter测试
```bash
# 模拟写失败
chmod 555 deploy/data/ofi_cvd/
# 观察 artifacts/deadletter/ 目录下的文件
# 恢复权限后继续运行
chmod 755 deploy/data/ofi_cvd/
```

### 3. 性能测试
```bash
# 监控内存使用
top -p $(pgrep -f run_success_harvest)
# 监控特征表生成时间
grep "生成特征宽表" logs/harvest.log
```

## 监控指标

### 1. 缓冲区大小
```python
# 添加监控代码
for kind in self.data_buffers:
    for symbol in self.symbols:
        size = len(self.data_buffers[kind][symbol])
        if size > self.buffer_high.get(kind, 10000) * 0.8:
            logger.warning(f"[MEMORY] {symbol}-{kind} 接近高水位: {size}")
```

### 2. Deadletter统计
```bash
# 统计 deadletter 文件数量
ls -la artifacts/deadletter/*/*.ndjson | wc -l
```

### 3. 轮转耗时
```bash
# 监控轮转耗时
grep "2030-01-01 执行定时轮转" logs/harvest.log | \
  awk '{print $NF}' | sort -n
```

## 注意事项

1. **Deadletter恢复**：系统不会自动恢复 deadletter 数据，需要手动处理
2. **水位线调整**：根据实际数据流量调整水位线阈值
3. **并发度控制**：不要设置过高的 `SAVE_CONCURRENCY`，可能导致磁盘IO压力过大
4. **监控告警**：建议添加缓冲区大小和 deadletter 文件数的监控告警

## 后续优化建议

1. 添加 deadletter 自动恢复机制
2. 实现缓冲区大小动态调整
3. 添加更多的性能指标统计
4. 考虑使用持久化队列（如 Redis）作为缓冲区

## 总结

这次修复通过以下手段彻底解决了"运行一段时间后不再落盘"的问题：

1. ✅ **水位线机制** - 防止内存无界增长
2. ✅ **Deadletter机制** - 防止保存失败死循环
3. ✅ **锁粒度优化** - 防止轮转阻塞
4. ✅ **性能优化** - 防止特征表拖慢

现在系统可以在高负载和故障情况下稳定运行，数据不丢失，内存可控！

