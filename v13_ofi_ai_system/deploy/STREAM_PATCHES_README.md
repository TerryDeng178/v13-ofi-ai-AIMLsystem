# 交易流卡死修复补丁说明

## 问题描述

原始实现中存在"交易流卡死但连接仍存活"的问题：
- 交易流使用 `async for message in websocket` 的阻塞式读取
- 当服务器长时间不推送消息时，循环会一直卡住
- 订单簿流仍在正常工作，掩盖了交易流的问题
- 健康检查只维护全局数据时间戳，无法区分交易流和订单簿流状态

## 解决方案

### 补丁A：流超时Watchdog

**核心改进**：将阻塞式读取改为带超时的单次接收

```python
# 原始代码（会卡死）
async for message in websocket:
    # 处理消息

# 修复后代码（带超时）
while self.running:
    try:
        message = await asyncio.wait_for(websocket.recv(), timeout=self.stream_idle_sec)
        # 处理消息
    except asyncio.TimeoutError:
        logger.warning(f"[TRADE] {self.stream_idle_sec}s 未收到消息，触发重连")
        raise  # 触发外层重连
```

**环境变量配置**：
- `STREAM_IDLE_SEC`: 流空闲超时秒数（默认120秒）
- `TRADE_TIMEOUT`: 交易流超时秒数（默认150秒，确保早于健康告警）
- `ORDERBOOK_TIMEOUT`: 订单簿流超时秒数（默认180秒，确保早于健康告警）

### 补丁B：分流监控

**核心改进**：健康检查改为分别监控交易流和订单簿流

```python
# 新增分流时间戳
self.last_trade_time = {symbol: self._mono() for symbol in self.symbols}
self.last_ob_time = {symbol: self._mono() for symbol in self.symbols}

# 分流健康检查
def _check_health(self):
    for symbol in self.symbols:
        if now - self.last_trade_time[symbol] > self.trade_timeout:
            logger.warning(f"[HEALTH][TRADE] {symbol} 交易流超时")
        if now - self.last_ob_time[symbol] > self.orderbook_timeout:
            logger.warning(f"[HEALTH][OB] {symbol} 订单簿流超时")
```

**环境变量配置**：
- `HEALTH_CHECK_INTERVAL`: 健康检查间隔（默认25秒，更快发现软性停滞）

## 新增优化功能

### 补丁C：批量落盘并发优化

**核心改进**：提高保存吞吐量，避免串行保存瓶颈

```python
# 原始代码（串行保存）
async with self.save_semaphore:
    await asyncio.gather(*tasks, return_exceptions=True)

# 优化后代码（并发保存）
async def save_with_semaphore(task):
    async with self.save_semaphore:
        return await task

wrapped_tasks = [save_with_semaphore(task) for task in tasks]
await asyncio.gather(*wrapped_tasks, return_exceptions=True)
```

**环境变量配置**：
- `SAVE_CONCURRENCY`: 保存并发度（默认2）

### 补丁D：极端流量保护

**核心改进**：动态调整轮转间隔，应对极端交易流量

```python
# 极端流量检测
if max_prices_buffer >= self.extreme_traffic_threshold:
    self.extreme_traffic_mode = True
    self.parquet_rotate_sec = self.extreme_rotate_sec  # 30秒轮转
```

**环境变量配置**：
- `EXTREME_TRAFFIC_THRESHOLD`: 极端流量阈值（默认30000）
- `EXTREME_ROTATE_SEC`: 极端流量轮转间隔（默认30秒）

## 使用方法

### 1. 基本运行

```bash
# 使用默认配置
python run_success_harvest.py

# 自定义超时配置
export STREAM_IDLE_SEC=90
export TRADE_TIMEOUT=120  # 确保早于健康告警
export ORDERBOOK_TIMEOUT=150  # 确保早于健康告警
export HEALTH_CHECK_INTERVAL=20  # 更快检测
export SAVE_CONCURRENCY=3  # 提高并发度
export EXTREME_TRAFFIC_THRESHOLD=25000  # 自定义极端流量阈值
python run_success_harvest.py
```

### 2. 测试补丁

```bash
# 运行测试脚本
python test_stream_patches.py
```

### 3. 监控日志

**正常重连日志**：
```
[TRADE] 120s 未收到消息，触发重连
统一交易流连接错误: ...
连接统一交易流: 6个symbol
```

**健康检查日志**：
```
[HEALTH][TRADE] BTCUSDT 交易流超时 185.2s
[HEALTH][OB] ETHUSDT 订单簿流超时 305.1s
```

**轮转统计日志**：
```
[ROTATE] BTCUSDT: trade_delta=45.2s, ob_delta=12.1s, buffers={'prices': 150, 'ofi': 80}, reconnect_count=2, mode=NORMAL
```

**极端流量保护日志**：
```
[EXTREME_TRAFFIC] 进入极端流量模式: max_prices_buffer=32000, 轮转间隔调整为30秒
[EXTREME_TRAFFIC] 退出极端流量模式: max_prices_buffer=18000, 轮转间隔恢复为60秒
```

## 配置建议

### 生产环境推荐配置

```bash
# 保守配置（适合稳定环境）
export STREAM_IDLE_SEC=120
export TRADE_TIMEOUT=150  # 确保早于健康告警
export ORDERBOOK_TIMEOUT=180  # 确保早于健康告警
export HEALTH_CHECK_INTERVAL=25
export SAVE_CONCURRENCY=2
export EXTREME_TRAFFIC_THRESHOLD=30000

# 激进配置（适合快速检测）
export STREAM_IDLE_SEC=60
export TRADE_TIMEOUT=90   # 确保早于健康告警
export ORDERBOOK_TIMEOUT=120  # 确保早于健康告警
export HEALTH_CHECK_INTERVAL=15
export SAVE_CONCURRENCY=4
export EXTREME_TRAFFIC_THRESHOLD=20000
```

### 测试环境配置

```bash
# 快速测试配置
export STREAM_IDLE_SEC=10
export TRADE_TIMEOUT=15
export ORDERBOOK_TIMEOUT=20
export HEALTH_CHECK_INTERVAL=5
export SAVE_CONCURRENCY=3
export EXTREME_TRAFFIC_THRESHOLD=1000
export RUN_HOURS=0.1
```

## 验证方法

### 1. 观察重连日志

正常运行时应该看到：
- 定期轮转日志
- 偶尔的重连日志（正常现象）
- 健康检查警告（当流确实超时时）

### 2. 检查数据完整性

- 价格数据应该持续写入
- 重连期间不应该有长时间的数据中断
- 缓冲区大小应该保持稳定

### 3. 性能监控

- `reconnect_count` 应该合理（不是0，但也不应该过高）
- 缓冲区大小应该稳定
- 延迟时间应该合理

## 故障排除

### 1. 频繁重连

**症状**：`reconnect_count` 增长过快
**原因**：超时设置过短
**解决**：增加 `STREAM_IDLE_SEC` 值

### 2. 数据中断

**症状**：长时间没有新数据
**原因**：超时设置过长
**解决**：减少 `STREAM_IDLE_SEC` 值

### 3. 健康检查误报

**症状**：频繁的健康检查警告
**原因**：超时阈值设置不当
**解决**：调整 `TRADE_TIMEOUT` 和 `ORDERBOOK_TIMEOUT`

## 技术细节

### 1. 超时机制

- `asyncio.wait_for()` 提供超时控制
- 超时后抛出 `TimeoutError`
- 异常传播到外层触发重连

### 2. 时间戳管理

- 使用 `time.monotonic()` 避免NTP回拨
- 分流时间戳独立更新
- 全局时间戳保持兼容性

### 3. 并发安全

- 使用 `asyncio.Lock()` 保证轮转安全
- 信号量控制保存并发数
- 原子操作避免数据竞争

## 总结

这两个补丁解决了原始实现中的关键问题：
1. **补丁A**：防止流卡死，确保自动重连
2. **补丁B**：提供分流监控，便于问题诊断

通过合理的配置，系统现在能够：
- 自动检测和处理流中断
- 提供详细的监控信息
- 保持数据采集的连续性
- 支持生产环境的稳定运行
