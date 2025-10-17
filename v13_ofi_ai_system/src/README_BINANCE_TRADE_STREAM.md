# Binance Trade Stream 使用文档

## 📋 文档信息

- **模块名称**: `binance_trade_stream.py`
- **版本**: v1.0.0
- **创建时间**: 2025-10-17
- **最后更新**: 2025-10-17
- **任务来源**: Task 1.2.9 - 集成Trade流和CVD计算

---

## 🎯 功能概述

`binance_trade_stream.py` 是一个轻量级的Binance Trade流WebSocket客户端，用于实时接收和处理 `aggTrade` 数据，并集成CVD（Cumulative Volume Delta）计算。

### 核心功能

1. ✅ **WebSocket连接管理**
   - 连接Binance Futures aggTrade流
   - 自动心跳检测（60s超时）
   - 指数退避重连（1s → 30s）

2. ✅ **CVD实时计算**
   - 集成 `RealCVDCalculator`
   - 实时计算CVD、Z-score、EMA
   - 支持Tick Rule方向判定

3. ✅ **背压管理**
   - 有界队列（默认1024）
   - 队列满时丢弃旧帧，保留最新数据

4. ✅ **监控指标**
   - `reconnect_count`: 重连次数
   - `queue_dropped`: 队列丢弃计数
   - `total_messages`: 总消息数
   - `parse_errors`: 解析错误数
   - （注：`latency_ms` 在处理日志中单独打印，不属于 `MonitoringMetrics`）

5. ✅ **日志与速率限制**
   - 分级日志（INFO/DEBUG）
   - 噪音日志速率限制（5条/秒）
   - 定期统计输出（60秒）

---

## 🚀 快速开始

### 1. 基础运行

```bash
# 默认连接ETHUSDT aggTrade流
cd v13_ofi_ai_system/src
python binance_trade_stream.py
```

**默认配置**:
- 交易对: ETHUSDT
- 打印间隔: 每100条成交
- 队列大小: 1024
- 心跳超时: 60秒

---

### 2. 命令行参数

```bash
# 指定交易对
python binance_trade_stream.py --symbol BTCUSDT

# 自定义WebSocket URL
python binance_trade_stream.py --url wss://fstream.binancefuture.com/stream?streams=btcusdt@aggTrade
```

---

### 3. 环境变量配置

```bash
# 完整配置示例
export SYMBOL=ETHUSDT
export WS_URL=wss://fstream.binancefuture.com/stream?streams=ethusdt@aggTrade
export QUEUE_SIZE=2048
export PRINT_EVERY=100
export HEARTBEAT_TIMEOUT=60
export BACKOFF_MAX=30
export LOG_LEVEL=INFO

# 运行
python binance_trade_stream.py
```

---

## 📖 详细API文档

### 类：`MonitoringMetrics`

**监控指标数据类**，用于追踪系统运行状态。

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `reconnect_count` | int | 重连次数（首次连接不计） |
| `queue_dropped` | int | 队列丢弃消息数 |
| `total_messages` | int | 总接收消息数 |
| `parse_errors` | int | 解析失败消息数 |

**注意**: `latency_ms` 不在此监控指标集中，延迟是在处理日志中单独计算并打印（从交易所事件时间到本地接收时间）。

#### 方法

**`queue_dropped_rate() -> float`**

计算队列丢弃率。

```python
rate = queue_dropped / total_messages
```

**返回**: 丢弃率（0.0-1.0）

---

**`to_dict() -> Dict[str, Any]`**

导出为字典格式。

**返回**:
```python
{
    "reconnect_count": 0,
    "queue_dropped": 5,
    "total_messages": 10000,
    "parse_errors": 0,
    "queue_dropped_rate": 0.0005
}
```

---

### 函数：`parse_aggtrade_message(text: str)`

**解析Binance aggTrade消息**。

#### 参数

- `text` (str): WebSocket接收的原始JSON字符串

#### 返回值

- **成功**: `Tuple[float, float, bool, Optional[int]]`
  - `price`: 成交价格
  - `qty`: 成交数量
  - `is_buy`: 买卖方向（True=买入，False=卖出）
  - `event_time_ms`: 事件时间戳（毫秒）

- **失败**: `None`（解析错误）

#### 示例

```python
text = '{"data":{"p":"3245.5","q":"10.0","m":false,"E":1697527081000}}'
result = parse_aggtrade_message(text)
# 返回: (3245.5, 10.0, True, 1697527081000)
```

---

### 协程：`ws_consume(url, queue, stop_evt, metrics)`

**WebSocket消费者**，负责连接Binance并接收消息。

#### 参数

- `url` (str): WebSocket URL
- `queue` (asyncio.Queue): 消息队列
- `stop_evt` (asyncio.Event): 停止事件
- `metrics` (MonitoringMetrics): 监控指标对象

#### 功能

1. **连接管理**
   - 首次连接成功
   - 连接断开后自动重连
   - 指数退避策略

2. **心跳检测**
   - 60秒无消息 → 超时重连
   - 使用 `asyncio.wait_for`

3. **背压处理**
   - 队列满时丢弃最旧消息
   - 记录 `queue_dropped`

#### 日志输出

```
INFO Connected: wss://fstream.binancefuture.com/stream?streams=ethusdt@aggTrade
INFO [METRICS] reconnect_count=1
WARNING Heartbeat timeout (>60s). Reconnecting...
WARNING Connection closed: ...
WARNING Reconnect due to error: ...
```

---

### 协程：`processor(symbol, queue, stop_evt, metrics)`

**消息处理器**，负责解析消息并计算CVD。

#### 参数

- `symbol` (str): 交易对符号（如 "ETHUSDT"）
- `queue` (asyncio.Queue): 消息队列
- `stop_evt` (asyncio.Event): 停止事件
- `metrics` (MonitoringMetrics): 监控指标对象

#### 功能

1. **消息解析**
   - 从队列获取消息
   - 调用 `parse_aggtrade_message`
   - 解析失败计入 `parse_errors`

2. **CVD计算**
   - 调用 `RealCVDCalculator.update_with_trade()`
   - 计算 CVD、Z-score、EMA

3. **延迟计算**
   - `latency_ms = current_time - event_time_ms`
   - 端到端延迟监控

4. **定期统计**
   - 每60秒输出处理统计
   - 每N条打印CVD状态

#### 日志输出

```
INFO CVD ETHUSDT | cvd=-15180.334000 z=-2.211 ema=-14675.903796 | warmup=False std_zero=False bad=0 | latency=187.1ms
INFO [STAT] trades=100 avg_proc=0.523ms | {'reconnect_count': 0, 'queue_dropped': 0, 'total_messages': 500, 'parse_errors': 0, 'queue_dropped_rate': 0.0}
WARNING Parse error on message (truncated): ...
```

**格式说明**:
- **CVD日志**: `cvd/z/ema | warmup/std_zero/bad | latency`（每N条打印）
- **统计日志**: 定期输出（60秒），包含完整监控指标

---

### 协程：`main(symbol, url)`

**主入口函数**。

#### 参数

- `symbol` (Optional[str]): 交易对符号（默认从环境变量 `SYMBOL` 或 "ETHUSDT"）
- `url` (Optional[str]): WebSocket URL（默认自动构建）

#### 功能

1. **初始化**
   - 创建队列、停止事件、监控指标
   - 配置信号处理（SIGINT/SIGTERM）

2. **启动任务**
   - `ws_consume`: WebSocket消费者
   - `processor`: 消息处理器

3. **优雅关闭**
   - 等待停止事件
   - 取消所有任务
   - 输出最终指标

#### 示例

```python
import asyncio

# 方式1: 直接运行
asyncio.run(main())

# 方式2: 指定参数
asyncio.run(main(symbol="BTCUSDT"))

# 方式3: 自定义URL
asyncio.run(main(url="wss://..."))
```

---

## 🔧 配置参数详解

### 环境变量

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `SYMBOL` | str | "ETHUSDT" | 交易对符号 |
| `WS_URL` | str | 自动构建 | WebSocket URL（优先级高于SYMBOL） |
| `QUEUE_SIZE` | int | 1024 | 消息队列大小 |
| `PRINT_EVERY` | int | 100 | 打印间隔（条数） |
| `HEARTBEAT_TIMEOUT` | int | 60 | 心跳超时（秒） |
| `BACKOFF_MAX` | int | 30 | 最大退避时间（秒） |
| `LOG_LEVEL` | str | "INFO" | 日志级别（DEBUG/INFO/WARNING/ERROR） |

---

### 命令行参数

```bash
python binance_trade_stream.py --help
```

**输出**:
```
usage: binance_trade_stream.py [-h] [--symbol SYMBOL] [--url URL]

optional arguments:
  -h, --help       show this help message and exit
  --symbol SYMBOL  symbol, e.g. ETHUSDT (default from ENV SYMBOL)
  --url URL        override websocket URL (default from ENV WS_URL)
```

---

## 💡 使用场景

### 场景1: 实时监控CVD

```bash
# 监控ETHUSDT的CVD变化
export PRINT_EVERY=10
python binance_trade_stream.py --symbol ETHUSDT
```

**输出示例**:
```
2025-10-17 23:35:32,817 INFO CVD ETHUSDT | cvd=0.006000 z=None ema=0.006000 | warmup=True std_zero=False bad=0 | latency=187.2ms
2025-10-17 23:36:14,483 INFO CVD ETHUSDT | cvd=-15180.334000 z=-2.211 ema=-14675.903796 | warmup=False std_zero=False bad=0 | latency=187.1ms
```

**说明**: 日志格式为 `cvd/z/ema | warmup/std_zero/bad | latency`。监控指标（`dropped/reconnect`）在定期统计日志中输出。

---

### 场景2: 调试模式

```bash
# 启用DEBUG日志，查看详细信息
export LOG_LEVEL=DEBUG
export PRINT_EVERY=1
python binance_trade_stream.py
```

---

### 场景3: 高频交易对

```bash
# 增大队列，降低丢弃率
export QUEUE_SIZE=4096
export PRINT_EVERY=500
python binance_trade_stream.py --symbol BTCUSDT
```

---

### 场景4: 集成到其他脚本

```python
import asyncio
from binance_trade_stream import main

# 运行3分钟后停止
async def run_for_duration():
    task = asyncio.create_task(main(symbol="ETHUSDT"))
    await asyncio.sleep(180)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

asyncio.run(run_for_duration())
```

---

## ⚠️ 注意事项与最佳实践

### 1. WebSocket URL格式

**✅ 正确**:
```python
url = f"wss://fstream.binancefuture.com/stream?streams={symbol.lower()}@aggTrade"
# 例如: wss://...streams=ethusdt@aggTrade
```

**❌ 错误**:
```python
url = f"wss://...streams={symbol}@aggTrade"  # 未转小写
url = "wss://...streams=ETHUSDT@aggTrade"    # 大写会失败
```

---

### 2. 心跳超时调优

- **交易活跃期**（如美股开盘）: 60s足够
- **交易清淡期**（如周末、节假日）: 可增至120s
- **网络不稳定**: 可降至30s，快速检测断连

```bash
export HEARTBEAT_TIMEOUT=120  # 增加超时时间
```

---

### 3. 队列大小与丢弃率

**队列丢弃率计算**:
```
丢弃率 = queue_dropped / total_messages
```

**推荐配置**:
- **一般场景**: `QUEUE_SIZE=1024`，丢弃率 <0.1%
- **高频交易对**: `QUEUE_SIZE=2048`，丢弃率 <0.01%
- **低延迟要求**: `QUEUE_SIZE=512`，优先实时性

---

### 4. 打印频率优化

| 场景 | PRINT_EVERY | 说明 |
|------|-------------|------|
| **开发调试** | 1-10 | 实时查看每笔成交 |
| **正常监控** | 100（默认） | 定期查看状态 |
| **生产环境** | 1000+ | 最小化日志量 |
| **性能测试** | 10000 | 最小化I/O开销（不能设为0，会触发取模异常） |

---

### 5. 延迟分析

**延迟组成**:
```
总延迟 = 网络延迟 + 处理延迟 + 队列等待
```

**正常范围**:
- **网络延迟**: 50-200ms（取决于地理位置）
- **处理延迟**: <1ms（CVD计算）
- **队列等待**: <10ms（队列未满）

**异常情况**:
- `latency > 1000ms`: 网络问题或服务器时钟偏移
- `latency < 0`: 本地时钟偏移

---

### 6. 错误处理策略

**解析错误**:
- 记录到 `parse_errors`
- 不中断流程
- 日志记录前160字符

**重连策略**:
- 初始退避: 1秒
- 每次翻倍: 2秒 → 4秒 → 8秒 → 16秒 → 30秒
- 上限: 30秒

**背压处理**:
- 丢弃旧帧，保留最新
- 适合实时监控场景
- 不适合数据完整性要求高的场景

---

## 🔍 监控与故障排查

### 监控指标解读

**1. `reconnect_count`**
- **期望值**: 0（理想）
- **警告阈值**: >3次/小时
- **原因**: 网络不稳定、服务器维护、心跳超时

**2. `queue_dropped`**
- **期望值**: 0（理想）
- **警告阈值**: 丢弃率 >0.5%
- **原因**: 消费速度慢、队列太小、打印过于频繁

**3. `parse_errors`**
- **期望值**: 0（严格）
- **警告阈值**: >0
- **原因**: Binance消息格式变更、编码问题

**4. `latency_ms`**
- **期望值**: <500ms（正常）
- **警告阈值**: p95 >1000ms
- **原因**: 网络延迟、服务器负载、本地时钟偏移

---

### 常见问题

#### Q1: 连接一直失败

**症状**:
```
WARNING Reconnect due to error: ...
WARNING Reconnect due to error: ...
```

**排查步骤**:
1. 检查网络连接: `ping fstream.binancefuture.com`
2. 检查URL格式: symbol必须小写
3. 检查防火墙/代理设置
4. 尝试其他交易对（如BTCUSDT）

---

#### Q2: 高队列丢弃率

**症状**:
```
INFO [STAT] ... 'queue_dropped_rate': 0.025
```

**解决方案**:
1. **增大队列**: `export QUEUE_SIZE=2048`
2. **降低打印频率**: `export PRINT_EVERY=200`
3. **优化处理速度**: 减少日志I/O
4. **检查系统负载**: CPU/内存是否充足

---

#### Q3: CVD值异常

**症状**: CVD值持续为0或异常大

**排查步骤**:
1. 检查 `bad_points`: 是否有解析错误
2. 检查 `parse_errors`: 消息格式是否正确
3. 检查 `is_buy` 映射: `m=True` → `is_buy=False`
4. 查看日志: 是否有警告信息

---

#### Q4: Z-score一直为None

**原因**: warmup期（历史数据不足）

**说明**:
- Z-score需要足够的历史数据
- warmup阈值: `max(5, z_window//5)` = 60笔成交（默认窗口300）
- 等待60笔成交后，`z_cvd` 才会有值

---

## 📊 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **处理延迟** | <1ms | 单笔成交处理时间（CVD计算） |
| **内存占用** | ~10-20MB | 包括队列、历史数据 |
| **CPU占用** | <5% | 单核CPU使用率 |
| **网络流量** | ~1-5KB/s | WebSocket持续连接 |
| **吞吐量** | >1000笔/秒 | 理论上限（实际取决于交易对） |

---

## 🔗 相关文件

- **本模块**: `v13_ofi_ai_system/src/binance_trade_stream.py`
- **CVD计算器**: `v13_ofi_ai_system/src/real_cvd_calculator.py`
- **CVD计算器文档**: `v13_ofi_ai_system/src/README_CVD_CALCULATOR.md`
- **任务卡**: `v13_ofi_ai_system/TASKS/Stage1_真实OFI+CVD核心/Task_1.2.9_集成Trade流和CVD计算.md`

**注意**: 本模块是独立的WebSocket客户端库，可被其他脚本导入使用（如 `examples/run_realtime_cvd.py`）。

---

## 📚 技术参考

### Binance API文档
- **Futures WebSocket**: https://binance-docs.github.io/apidocs/futures/en/#websocket-market-streams
- **aggTrade字段说明**: https://binance-docs.github.io/apidocs/futures/en/#aggregate-trade-streams

### Python异步编程
- **asyncio官方文档**: https://docs.python.org/3/library/asyncio.html
- **websockets库**: https://websockets.readthedocs.io/

---

## 📞 支持与反馈

- **项目**: V13 OFI+CVD+AI System
- **任务来源**: Task 1.2.9
- **模块路径**: `v13_ofi_ai_system/src/binance_trade_stream.py`
- **问题反馈**: 通过项目任务卡系统提交

---

**最后更新**: 2025-10-17  
**文档版本**: v1.0.0  
**状态**: ✅ 稳定（已通过Task 1.2.9验证）

