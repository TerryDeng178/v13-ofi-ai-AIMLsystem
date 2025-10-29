# run_success_harvest.py 潜在BUGS分析报告

## 问题：运行一段时间后不再继续落盘文件

经过代码审查，发现以下潜在问题：

---

## 🐛 BUG #1: 数据缓冲区无限增长

**位置**: 第111-120行（初始化），以及各处 `.append()` 操作

**问题描述**:
```python
# 数据缓冲区（初始化）
self.data_buffers = {
    'prices': {symbol: [] for symbol in self.symbols},
    'ofi': {symbol: [] for symbol in self.symbols},
    # ... 其他缓冲区都是普通列表，没有大小限制
}
```

**严重性**: 🔴 高

**原因**:
- 缓冲区使用普通 Python 列表 `[]`，没有大小限制
- 当 `_check_and_rotate_data()` 轮转失败或触发不及时时，缓冲区会无限增长
- 导致内存占用不断上升，最终可能触发 OOM

**证据**:
- `orderbook_buf` 使用了 `deque(maxlen=256)` 有限制
- 但 `data_buffers` 中的列表完全没有限制
- 没有监控缓冲区大小的逻辑

---

## 🐛 BUG #2: 跨日期分区可能导致目录结构问题

**位置**: 第1383-1410行（_save_data 方法）

**问题描述**:
```python
# 按事件时间分区，避免跨日错桶（使用UTC时间确保一致性）
df['date'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True).dt.strftime('%Y-%m-%d')

# 按日期分组保存
for date_str, date_group in df.groupby('date'):
    # ...
    filepath = base_dir / f"date={date_str}" / f"symbol={symbol}" / f"kind={kind}" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
```

**严重性**: 🟡 中

**原因**:
- 如果缓冲区积累了大量跨日期的数据，一次轮转会写入多个日期目录
- 目录创建 `mkdir(parents=True, exist_ok=True)` 可能失败（权限问题）
- 没有异常处理，如果目录创建失败会导致整个轮转失败

**证据**:
- 只有在保存时才创建目录
- 没有提前验证目录可写性

---

## 🐛 BUG #3: 轮转锁可能导致阻塞

**位置**: 第1424-1446行（_check_and_rotate_data 方法）

**问题描述**:
```python
async def _check_and_rotate_data(self):
    current_time = datetime.now().timestamp()
    if current_time - self.last_rotate_time >= self.parquet_rotate_sec:
        async with self.rotation_lock:
            # 双重检查
            if current_time - self.last_rotate_time >= self.parquet_rotate_sec:
                # 批量保存所有缓冲区的数据
                for symbol in self.symbols:
                    # ...
                    for kind in ['prices', 'orderbook']:
                        if self.data_buffers[kind][symbol]:
                            await self._save_data(symbol, kind)
```

**严重性**: 🟡 中

**原因**:
- 轮转过程中持有锁，如果保存某个文件耗时过长，会阻塞所有其他操作
- 每次轮转都要为每个 symbol × 每个 kind 执行保存操作
- 如果某个文件保存失败（磁盘满、权限问题），可能导致整个轮转过程中断
- 没有超时机制

**证据**:
- 使用 `async with self.rotation_lock` 会完全阻塞
- 没有 try-finally 保证锁一定会释放

---

## 🐛 BUG #4: 异常恢复机制不完整

**位置**: 第1416-1419行（_save_data 异常处理）

**问题描述**:
```python
except Exception as e:
    logger.error(f"保存数据错误 {symbol}-{kind}: {e}")
    # 回灌：避免本次快照丢失
    self.data_buffers[kind][symbol] = buf + self.data_buffers[kind][symbol]
```

**严重性**: 🔴 高

**原因**:
- 如果保存失败，数据会回灌到缓冲区
- 但如果下次轮转又失败，数据会无限积累
- 没有记录失败次数或跳过机制
- 如果磁盘满了，会一直失败→回灌→再失败的死循环

**证据**:
- 回灌逻辑可能导致数据重复
- 没有最大重试次数限制
- 没有降级策略

---

## 🐛 BUG #5: 健康检查可能导致进程意外退出

**位置**: 第268-292行（_check_health 方法），第1630-1645行（_health_check_loop）

**问题描述**:
```python
def _check_health(self):
    # ...
    if self.connection_errors > self.max_connection_errors:
        logger.error(f"[HEALTH] 连接错误过多: {self.connection_errors}")
        return False

async def _health_check_loop(self):
    while self.running:
        if not self._check_health():
            logger.error("[HEALTH] 健康检查失败，准备重启")
            self.running = False
            break
```

**严重性**: 🟡 中

**原因**:
- 健康检查失败会直接设置 `self.running = False`
- 但此时可能还有数据在缓冲区未保存
- 进程退出前会执行 finally 保存，但如果轮转失败就不会保存

**证据**:
- 健康检查过于激进
- 没有区分"数据流中断"和"数据缓冲正常"的情况

---

## 🐛 BUG #6: 特征表生成可能有性能问题

**位置**: 第353-468行（_generate_features_table 方法）

**问题描述**:
```python
def _generate_features_table(self, symbol: str):
    """生成特征对齐宽表（按秒聚合）"""
    try:
        # 获取最近N秒的数据进行聚合
        current_time = datetime.now().timestamp()
        lookback_seconds = 60  # 聚合最近60秒的数据
        cutoff_time = current_time - lookback_seconds
        
        # 收集各类型数据
        prices_data = []
        ofi_data = []
        cvd_data = []
        fusion_data = []
        
        # 从缓冲区收集数据
        for data_point in self.data_buffers['prices'][symbol]:
            if data_point['ts_ms'] / 1000 >= cutoff_time:
                prices_data.append(data_point)
```

**严重性**: 🟠 中高

**原因**:
- 每次都遍历整个缓冲区来过滤数据
- 如果缓冲区很大（几万条），性能会很低
- 没有索引或优化

**证据**:
- 使用简单的列表遍历和 if 判断
- 每次轮转都要执行这个操作

---

## 🐛 BUG #7: 轮转触发频率过高

**位置**: 第1577-1578行、第1617-1618行（外交部调用）

**问题描述**:
```python
await self._process_trade_data(symbol, trade_data)

# 检查是否需要定时轮转
await self._check_and_rotate_data()
```

**严重性**: 🟡 中

**原因**:
- 每收到一条交易数据就检查一次轮转
- 每收到一条订单簿数据也检查一次轮转
- 在数据量大时（每秒几千条），检查频率过高
- `_check_and_rotate_data` 内部有时间检查，但还是有开销

**证据**:
- 交易流和订单簿流同时运行
- 两个流都频繁调用检查函数

---

## 🐛 BUG #8: 初始化时创建目录可能失败

**位置**: 第470-487行（_create_directory_structure 方法）

**问题描述**:
```python
def _create_directory_structure(self):
Morning    # 使用UTC时间创建目录，确保与数据分区一致
    today_utc = datetime.utcnow().strftime("%Y-%m-%d")
    
    for symbol in self.symbols:
        # 权威库：只保留 raw（prices/orderbook）
        for kind in ['prices', 'orderbook']:
            dir_path = self.output_dir / f"date={today_utc}" / f"symbol={symbol}" / f"kind={kind}"
            dir_path.mkdir(parents=True, exist_ok=True)
```

**严重性**: 🟢 低

**原因**:
- 启动时只创建"今天"的目录
- 如果进程跨日运行，跨日数据在保存时才创建目录
- 此时可能因为权限问题失败

**证据**:
- 只在初始化时创建一次目录
- 保存时再创建可能出错

---

## 🎯 最可能导致"停止落盘"的BUG组合

**组合1**: BUG#1 + BUG#4
- 缓冲区无限增长 → 内存不足 → 保存异常 → 数据回灌 → 更慢 → 死循环
- **概率**: 🔴 高

**组合2**: BUG#3 + BUG#4
- 某个文件保存失败 → 持有轮转锁 → 阻塞其他操作 → 缓冲区积累 → 超时
- **概率**: 🟡 中

**组合3**: BUG#2 + BUG#4
- 跨日目录创建失败 → 保存失败 → 数据回灌 → 重复失败
- **概率**: 🟡 中

---

## 🔍 建议的监控和诊断

1. **添加缓冲区大小监控**
   ```python
   buffer_sizes = {k: len(v) for k, v in self.data_buffers.items()}
   logger.info(f"缓冲区大小: {buffer_sizes}")
   ```

2. **添加保存失败计数器**
   ```python
   self.save_failures = {symbol: 0 for symbol in self.symbols}
   ```

3. **添加轮转时长监控**
   ```python
   rotation_start = time.time()
   # ... 保存操作
   rotation_duration = time.time() - rotation_start
   if rotation_duration > 10:
       logger.warning(f"轮转耗时过长: {rotation_duration:.1f}秒")
   ```

4. **添加内存使用监控**
   ```python
   import psutil
   memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
   ```

---

## 📋 总结

最可能的原因是 **BUG#1（缓冲区无限增长）+ BUG#4（异常恢复机制不完整）** 的组合，导致：
1. 运行一段时间后内存不足
2. 保存操作变慢或失败
3. 数据回灌导致缓冲区更大
4. 形成死循环，最终进程崩溃或僵死

建议优先修复这两个BUG。

