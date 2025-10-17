# Binance WebSocket Client 使用规范

## 📋 文档版本信息
- **版本**: v1.0 (稳定版)
- **最后验证**: 2025-10-17
- **验证测试**: 30分钟稳态测试通过
- **文件路径**: `v13_ofi_ai_system/src/binance_websocket_client.py`

---

## ✅ 文件状态说明

### **重要**: 此文件已完成开发，无需修改！

**验证数据** (30分钟稳态测试 - 完整证据):
- ✅ **运行时长**: 1800秒 (30分钟完整)
- ✅ **接收消息**: 3540条
- ✅ **接收速率**: 1.97条/秒
- ✅ **延迟p50**: 80ms (阈值: <100ms)
- ✅ **延迟p95**: 82ms (阈值: <150ms)
- ✅ **延迟p99**: 93ms (阈值: <200ms)
- ✅ **连续性**: breaks=0, resyncs=0 (阈值: ==0)
- ✅ **稳定性**: reconnects=0 (阈值: <=1)
- ✅ **日志质量**: drops=0 (阈值: ==0)

**测试证据位置**:
- 📄 日志文件: `v13_ofi_ai_system/logs/ethusdt_20251017.log*` (7个轮转切片)
- 📄 指标快照1: `v13_ofi_ai_system/data/order_book/metrics_snapshot_t1789s.json`
- 📄 指标快照2: `v13_ofi_ai_system/data/order_book/metrics_snapshot_t1800s.json`
- 📄 NDJSON数据: `v13_ofi_ai_system/data/order_book/ethusdt_depth.ndjson.gz` (~1.2MB)
- 📄 验收报告: `v13_ofi_ai_system/TASKS/Stage1_真实OFI核心/✅Task_1.1.6_测试和验证.md`

**最终SUMMARY输出**:
```
SUMMARY | t=1789s | msgs=3534 | rate=1.97/s | 
         p50=80.0 p95=82.0 p99=93.3 | 
         breaks=0 resyncs=0 reconnects=0 | 
         batch_span_p95=786 max=1536 | 
         log_q_p95=2 max=7 drops=0
```

---

## 🔒 端点/路径规范（统一标准）

### Binance Futures 官方端点
**⚠️ 必须使用以下端点，不准随意更改！**

| 类型 | 端点URL | 说明 |
|------|---------|------|
| **REST API** | `https://fapi.binance.com/fapi/v1/depth` | 订单簿快照 |
| **WebSocket** | `wss://fstream.binancefuture.com/stream` | 实时订单簿增量 |

### 本地存储路径规范
**⚠️ 所有路径相对于 `v13_ofi_ai_system/` 根目录！**

| 文件类型 | 路径模板 | 示例 |
|---------|---------|------|
| **NDJSON数据** | `data/order_book/{symbol}_depth.ndjson.gz` | `data/order_book/ethusdt_depth.ndjson.gz` |
| **实时指标** | `data/order_book/metrics.json` | `data/order_book/metrics.json` |
| **日志文件** | `logs/{symbol}_{date}.log` | `logs/ethusdt_20251017.log` |
| **日志轮转** | `logs/{symbol}_{date}.log.{N}` | `logs/ethusdt_20251017.log.1` |
| **验收报告** | `reports/Task_{X}_{Y}_{Z}_validation.json` | `reports/Task_1_1_6_validation.json` |

### WebSocket Stream 格式
```
wss://fstream.binancefuture.com/stream?streams={symbol}@depth@{update_speed}

参数:
- symbol: 小写交易对，如 ethusdt, btcusdt
- update_speed: 100ms (固定，不准改)

完整示例:
wss://fstream.binancefuture.com/stream?streams=ethusdt@depth@100ms
```

### REST API 查询格式
```
https://fapi.binance.com/fapi/v1/depth?symbol={SYMBOL}&limit={LIMIT}

参数:
- SYMBOL: 大写交易对，如 ETHUSDT, BTCUSDT
- limit: 1000 (固定，不准改)

完整示例:
https://fapi.binance.com/fapi/v1/depth?symbol=ETHUSDT&limit=1000
```

---

## 📏 验收阈值标准（统一化）

### 核心指标阈值表
**⚠️ 以下阈值为硬性标准，任何测试必须通过！**

| 指标 | 阈值 | 验证方式 | 不通过=失败 |
|------|------|----------|------------|
| **运行时长** | `>= 1800s (30分钟)` | 实际运行时间 | ✅ 是 |
| **接收速率** | `>= 1.0 msg/s` | `rate` 字段 | ✅ 是 |
| **延迟p50** | `< 100ms` | `p50` 字段 | ⚠️ 警告 |
| **延迟p95** | `< 150ms` | `p95` 字段 | ⚠️ 警告 |
| **延迟p99** | `< 200ms` | `p99` 字段 | ⚠️ 警告 |
| **连续性断裂** | `== 0` | `breaks` 字段 | ✅ 是 |
| **重同步次数** | `== 0` | `resyncs` 字段 | ✅ 是 |
| **重连次数** | `<= 1` | `reconnects` 字段 | ⚠️ 警告 |
| **日志丢失** | `== 0` | `drops` 字段 | ✅ 是 |
| **批次跨度p95** | `< 1000` | `batch_span_p95` 字段 | ⚠️ 观测 |
| **日志队列深度** | `< 100` | `log_q_p95` 字段 | ⚠️ 观测 |

### 阈值检查脚本示例
```python
import json
from pathlib import Path

# 读取最终metrics.json
metrics_file = Path("v13_ofi_ai_system/data/order_book/metrics.json")
with open(metrics_file, "r", encoding="utf-8") as f:
    metrics = json.load(f)

# 硬性阈值检查
assert metrics["runtime_seconds"] >= 1800, f"运行时长不足: {metrics['runtime_seconds']}s < 1800s"
assert metrics["recv_rate"] >= 1.0, f"接收速率过低: {metrics['recv_rate']:.2f} < 1.0"
assert metrics["continuity"]["breaks"] == 0, f"连续性断裂: {metrics['continuity']['breaks']} != 0"
assert metrics["continuity"]["resyncs"] == 0, f"重同步异常: {metrics['continuity']['resyncs']} != 0"
assert metrics["log_queue"]["drops"] == 0, f"日志丢失: {metrics['log_queue']['drops']} != 0"

# 警告级别检查
if metrics["latency_ms"]["p95"] >= 150:
    print(f"⚠️ 警告: p95延迟过高 {metrics['latency_ms']['p95']:.1f}ms")
if metrics["continuity"]["reconnects"] > 1:
    print(f"⚠️ 警告: 重连次数过多 {metrics['continuity']['reconnects']}")

print("✅ 所有硬性阈值检查通过！")
```

### 验收交付物清单
**⚠️ 每次测试必须提交以下所有文件！**

- [ ] 📄 最终 `metrics.json` 文件
- [ ] 📄 两份 `metrics` 快照（相隔>=10s）
- [ ] 📄 日志文件切片列表（证明轮转生效）
- [ ] 📄 最后2条 `SUMMARY` 控制台输出
- [ ] 📄 NDJSON数据文件（含文件大小）
- [ ] 📄 验收报告 `reports/Task_*_validation.json`
- [ ] 📄 阈值检查脚本执行结果

---

## ⛔ 不准乱改的硬规矩

### 🔴 绝对禁止修改的内容

#### 1. 核心算法逻辑
```python
# ❌ 禁止修改: 连续性判断逻辑
if pu is not None and self.last_u is not None and int(pu) != int(self.last_u):
    self.stats['resyncs'] += 1
    self.logger.warning(f"Continuity break: pu={pu} != last_u={self.last_u}")

# ❌ 禁止修改: 首条消息处理逻辑
if not self.synced:
    self.synced = True
    self.last_u = int(u)
    self.logger.info(f"Started streaming from first message: u={u}")
```

#### 2. 数据格式规范
```python
# ❌ 禁止修改: NDJSON字段名和类型
ndjson_record = {
    "timestamp": str,      # ISO 8601格式
    "symbol": str,         # 交易对符号
    "ts_recv": float,      # 毫秒时间戳
    "E": int,              # 事件时间
    "U": int,              # 首个更新ID
    "u": int,              # 最后更新ID
    "pu": int,             # 上一批次最后ID
    "latency_event_ms": float,    # 事件延迟
    "latency_pipeline_ms": float  # 处理延迟
}
```

#### 3. 端点URL
```python
# ❌ 禁止修改: 官方端点
REST_URL = "https://fapi.binance.com/fapi/v1/depth"
WS_URL = "wss://fstream.binancefuture.com/stream?streams={symbol}@depth@100ms"
```

#### 4. 日志格式
```python
# ❌ 禁止修改: SUMMARY格式
SUMMARY | t={runtime}s | msgs={total} | rate={rate}/s | 
         p50={p50} p95={p95} p99={p99} | 
         breaks={breaks} resyncs={resyncs} reconnects={reconnects} | 
         batch_span_p95={span_p95} max={span_max} | 
         log_q_p95={q_p95} max={q_max} drops={drops}
```

### 🟡 允许修改的参数配置

#### 可通过命令行参数调整
```bash
# ✅ 允许修改: 交易对
--symbol BTCUSDT  # 默认: ETHUSDT

# ✅ 允许修改: 订单簿深度
--depth 10  # 默认: 5

# ✅ 允许修改: 日志轮转方式
--rotate interval  # 或 size

# ✅ 允许修改: 轮转间隔
--rotate-sec 3600  # 默认: 60

# ✅ 允许修改: 日志文件大小
--max-bytes 10000000  # 默认: 5000000

# ✅ 允许修改: 备份数量
--backups 24  # 默认: 7

# ✅ 允许修改: 打印间隔
--print-interval 60  # 默认: 10

# ✅ 允许修改: 运行时长
--run-minutes 60  # 默认: None (无限)
```

#### 可通过代码调整（需充分理由）
```python
# ✅ 允许修改: 队列大小（如果drops>0）
queue_max = 10000  # 默认: 10000

# ✅ 允许修改: 日志级别（生产环境建议INFO）
level = logging.INFO  # 默认: DEBUG

# ✅ 允许修改: 统计窗口大小
self.latency_window = deque(maxlen=1000)  # 默认: 1000
```

### 🔴 修改前必须满足的条件

**如果确实需要修改核心代码，必须**:

1. ✅ **充分理由**: 文档说明为什么必须修改
2. ✅ **最小补丁**: 修改行数 <= 60行
3. ✅ **不破坏接口**: NDJSON字段、metrics.json结构保持兼容
4. ✅ **保留旧逻辑**: 通过注释标记旧代码
5. ✅ **完整测试**: 重新运行30分钟稳态测试
6. ✅ **通过阈值**: 所有硬性阈值必须通过
7. ✅ **更新文档**: 同步更新本使用规范和任务卡
8. ✅ **版本记录**: 在版本历史中记录修改原因

### ⚠️ 违反规矩的后果

| 违规行为 | 后果 |
|---------|------|
| 修改核心算法未记录 | ❌ 代码回滚 |
| 修改端点URL | ❌ 连接失败，代码回滚 |
| 修改NDJSON字段 | ❌ 数据不兼容，代码回滚 |
| 修改未通过30分钟测试 | ❌ 代码回滚 |
| 修改未更新文档 | ⚠️ 警告，要求补充文档 |
| 修改超过60行 | ⚠️ 警告，要求拆分为多个任务 |

### ✅ 正确的修改流程

```
1. 提出修改需求 → 2. 评估必要性 → 3. 设计最小补丁 → 
4. 更新相关文档 → 5. 实施修改 → 6. 运行30分钟测试 → 
7. 验证所有阈值 → 8. 提交验收报告 → 9. 更新版本历史
```

---

## 🚀 基本使用

### 方式1: 命令行直接运行

```bash
# 最简单用法 (默认ETHUSDT, 无限运行)
python binance_websocket_client.py

# 指定交易对
python binance_websocket_client.py --symbol BTCUSDT

# 运行指定时间 (分钟)
python binance_websocket_client.py --symbol ETHUSDT --run-minutes 30

# 完整参数示例
python binance_websocket_client.py \
    --symbol ETHUSDT \
    --depth 5 \
    --rotate interval \
    --rotate-sec 60 \
    --max-bytes 5000000 \
    --backups 7 \
    --print-interval 10 \
    --run-minutes 30
```

### 方式2: Python代码调用

```python
from pathlib import Path
from binance_websocket_client import BinanceOrderBookStream

# 创建客户端
client = BinanceOrderBookStream(
    symbol="ETHUSDT",           # 交易对
    depth_levels=5,             # 订单簿深度
    rotate="interval",          # 日志轮转方式: "interval" 或 "size"
    rotate_sec=60,              # 轮转间隔(秒)
    max_bytes=5_000_000,        # 日志文件最大字节数
    backups=7,                  # 保留的备份文件数
    print_interval=10,          # SUMMARY打印间隔(秒)
    base_dir=Path("v13_ofi_ai_system")  # 基础目录
)

# 运行 (阻塞式)
client.run(reconnect=True)

# 或在后台线程运行
import threading
import time

t = threading.Thread(target=client.run, kwargs={"reconnect": True}, daemon=True)
t.start()

# 运行30分钟后停止
time.sleep(30 * 60)
if client.ws:
    client.ws.close()
client.listener.stop()
```

---

## 📂 输出文件说明

### 1. NDJSON数据文件
**路径**: `v13_ofi_ai_system/data/order_book/{symbol}_depth.ndjson.gz`

**格式**: 每行一条JSON记录 (NDJSON)
```json
{
  "timestamp": "2025-10-17T08:18:01.000Z",
  "symbol": "ETHUSDT",
  "ts_recv": 1697527081000.0,
  "E": 1697527081000,
  "U": 76585007743,
  "u": 76585007745,
  "pu": 76585006694,
  "latency_event_ms": 79.0,
  "latency_pipeline_ms": 0.5
}
```

**字段说明**:
- `timestamp`: 事件UTC时间 (ISO格式)
- `symbol`: 交易对符号
- `ts_recv`: 接收时间戳 (毫秒)
- `E`: 事件时间戳 (毫秒)
- `U`: 本批次第一个更新ID
- `u`: 本批次最后一个更新ID
- `pu`: 上一批次最后一个更新ID
- `latency_event_ms`: 事件延迟 (毫秒)
- `latency_pipeline_ms`: 处理延迟 (毫秒)

**用途**: 
- 历史数据回放
- OFI计算输入
- 数据分析和研究

### 2. 实时指标文件
**路径**: `v13_ofi_ai_system/data/order_book/metrics.json`

**格式**: JSON对象 (每10秒刷新)
```json
{
  "timestamp": "2025-10-17T08:38:01.509",
  "window_sec": 10,
  "runtime_seconds": 1200.0,
  "total_messages": 2639,
  "recv_rate": 2.20,
  "latency_ms": {
    "avg_ms": 80.5,
    "min_ms": 75.0,
    "max_ms": 150.0,
    "p50": 79.0,
    "p95": 81.0,
    "p99": 98.0
  },
  "continuity": {
    "breaks": 0,
    "resyncs": 0,
    "reconnects": 0
  },
  "batch_span": {
    "p95": 795,
    "max": 1536
  },
  "log_queue": {
    "depth_p95": 2,
    "depth_max": 7,
    "drops": 0
  },
  "symbol": "ETHUSDT"
}
```

**用途**:
- 实时监控
- 性能评估
- 告警触发

### 3. 日志文件
**路径**: `v13_ofi_ai_system/logs/{symbol}_{date}.log`

**轮转规则**:
- **时间轮转**: 每N秒创建新文件 (默认60秒)
- **大小轮转**: 文件超过N字节创建新文件 (默认5MB)
- **备份保留**: 保留最近N个备份 (默认7个)

**日志级别**:
- `INFO`: 关键事件 (连接、SUMMARY)
- `DEBUG`: 详细调试信息 (消息接收)
- `WARNING`: 连续性警告
- `ERROR`: 错误信息

---

## 📊 实时监控

### SUMMARY输出格式
```
SUMMARY | t=1200s | msgs=2639 | rate=2.20/s | 
         p50=79.0 p95=81.0 p99=98.0 | 
         breaks=0 resyncs=0 reconnects=0 | 
         batch_span_p95=795 max=1536 | 
         log_q_p95=2 max=7 drops=0
```

**字段解释**:
- `t`: 运行时长 (秒)
- `msgs`: 总消息数
- `rate`: 接收速率 (条/秒)
- `p50/p95/p99`: 延迟分位数 (毫秒)
- `breaks`: 连续性断裂次数 (应为0)
- `resyncs`: 重同步次数 (应为0)
- `reconnects`: 重连次数
- `batch_span_p95`: 批次跨度95分位
- `log_q_p95`: 日志队列深度95分位
- `drops`: 日志丢失次数 (应为0)

### 监控指标阈值

| 指标 | 正常范围 | 警告阈值 | 异常阈值 |
|------|----------|----------|----------|
| **rate** | 1.0-3.0/s | <0.5/s | <0.1/s |
| **p50** | 50-100ms | >200ms | >500ms |
| **p95** | 70-150ms | >300ms | >1000ms |
| **p99** | 80-200ms | >500ms | >2000ms |
| **breaks** | 0 | 0 | >0 |
| **resyncs** | 0 | 1-2 | >3 |
| **reconnects** | 0-1 | 2-3 | >5 |
| **drops** | 0 | 0 | >0 |

---

## 🔧 命令行参数详解

```bash
python binance_websocket_client.py [OPTIONS]

OPTIONS:
  --symbol TEXT           交易对符号 (默认: ETHUSDT)
  --depth INTEGER         订单簿深度档位 (默认: 5)
  --rotate TEXT           日志轮转方式 "interval"|"size" (默认: interval)
  --rotate-sec INTEGER    轮转间隔秒数 (默认: 60)
  --max-bytes INTEGER     日志文件最大字节数 (默认: 5000000)
  --backups INTEGER       保留备份数量 (默认: 7)
  --print-interval INTEGER SUMMARY打印间隔秒数 (默认: 10)
  --run-minutes INTEGER   运行时长分钟数 (默认: None, 无限运行)
```

---

## ⚠️ 重要说明

### 1. REST vs WebSocket 序列号
**问题**: Binance Futures的REST API和WebSocket API使用**不同的序列号系统**

**当前解决方案**:
- ✅ 直接从第一条WebSocket消息开始处理
- ✅ 使用 `pu == last_u` 验证消息间连续性
- ✅ **不依赖REST快照对齐**

**影响**:
- ⚠️ 初始几秒的订单簿状态可能不完整
- ✅ 事件统计和延迟测量不受影响
- ✅ 连续性追踪完全可靠

**补偿措施**:
- 用于延迟监控和数据收集 (当前目的): **无影响**
- 用于完整订单簿重建: **建议等待60秒后再开始OFI计算**

### 2. 依赖项
**必需依赖**:
- `websocket-client`: WebSocket连接 (已在requirements.txt)

**无需新增依赖**:
- ✅ 异步日志使用标准库 `logging.handlers`
- ✅ 分位数计算使用纯Python实现
- ✅ JSON处理使用标准库 `json`

### 3. Windows特殊性
- 控制台编码默认GBK，emoji可能显示异常
- 文件锁定：需强制终止Python进程才能删除日志文件
- 路径分隔符：代码已兼容，使用 `Path` 对象

---

## 🎯 后续任务集成指南

### Task 1.2.x: OFI计算模块

**推荐架构**:
```python
# 1. 启动WebSocket客户端 (后台运行)
from binance_websocket_client import BinanceOrderBookStream
import threading

ws_client = BinanceOrderBookStream(symbol="ETHUSDT")
ws_thread = threading.Thread(target=ws_client.run, daemon=True)
ws_thread.start()

# 2. 读取NDJSON文件计算OFI (独立模块)
from real_ofi_calculator import RealOFICalculator
import gzip
import json

ofi_calc = RealOFICalculator()

with gzip.open("v13_ofi_ai_system/data/order_book/ethusdt_depth.ndjson.gz", "rt") as f:
    for line in f:
        data = json.loads(line)
        # 提取订单簿数据
        # ofi_value = ofi_calc.calculate(bids, asks)
        pass
```

**为什么这样设计**:
- ✅ **职责分离**: WebSocket专注数据接收，OFI专注计算
- ✅ **解耦**: 两者独立运行，互不影响
- ✅ **可测试**: NDJSON文件可用于回放测试
- ✅ **可扩展**: 轻松添加实时回调接口 (如需要)

### 如果需要实时回调 (可选)

**仅当需要毫秒级实时计算时才修改** (约10行代码):

```python
# 在 binance_websocket_client.py 的 __init__ 添加:
self.on_orderbook_callback: Optional[Callable] = None

# 在 on_message 处理完成后调用:
if self.on_orderbook_callback:
    # 解析bid/ask数据
    bids = [[float(p), float(q)] for p, q in data.get('b', [])]
    asks = [[float(p), float(q)] for p, q in data.get('a', [])]
    self.on_orderbook_callback(bids, asks, E, U, u)
```

**使用方式**:
```python
def on_orderbook_update(bids, asks, timestamp, U, u):
    ofi_value = ofi_calc.calculate(bids, asks)
    print(f"OFI: {ofi_value}")

ws_client.on_orderbook_callback = on_orderbook_update
ws_client.run()
```

**⚠️ 注意**: 回调函数必须非常快 (<1ms)，否则会阻塞WebSocket接收！

---

## 🐛 故障排查

### 问题1: WebSocket连接失败
```
ConnectionRefusedError: [WinError 10060]
```

**解决**:
- ✅ 检查网络连接
- ✅ 确认URL正确: `wss://fstream.binancefuture.com/stream?streams=...`
- ✅ 检查防火墙设置

### 问题2: 无SUMMARY输出
```
WebSocket opened
Loaded REST snapshot lastUpdateId=...
(然后无输出)
```

**已修复**: 此问题在v1.0已解决 (REST/WS序列号不匹配)

**验证**: 运行1分钟应该看到至少6条SUMMARY

### 问题3: resyncs > 0
```
SUMMARY | ... | resyncs=5 | ...
```

**原因**: 消息连续性断裂 (`pu != last_u`)

**排查**:
- 检查网络稳定性
- 查看日志文件中的 `WARNING` 信息
- 如果频繁发生 (>10次/小时): 考虑网络优化

### 问题4: drops > 0
```
SUMMARY | ... | drops=10 | ...
```

**原因**: 日志队列满，消息被丢弃

**解决**:
- 增加队列大小: 修改 `queue_max=10000` (在代码中)
- 减少日志级别: 改为 `INFO` (当前为 `DEBUG`)

---

## 📌 最佳实践

### 1. 生产环境部署
```bash
# 建议配置
python binance_websocket_client.py \
    --symbol ETHUSDT \
    --rotate interval \
    --rotate-sec 3600 \     # 每小时轮转
    --backups 24 \          # 保留24小时
    --print-interval 60     # 每分钟打印一次
```

### 2. 开发测试
```bash
# 快速测试 (5分钟)
python binance_websocket_client.py \
    --symbol ETHUSDT \
    --print-interval 10 \
    --run-minutes 5
```

### 3. 长期监控
- 使用 `systemd` 或 `supervisor` 管理进程
- 定期检查 `metrics.json`
- 设置告警: `breaks > 0` 或 `drops > 0`
- 定期清理旧日志文件

### 4. 数据分析
```python
# 读取所有历史数据
import gzip
import json
from pathlib import Path

ndjson_file = Path("v13_ofi_ai_system/data/order_book/ethusdt_depth.ndjson.gz")

latencies = []
with gzip.open(ndjson_file, "rt", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        latencies.append(data["latency_event_ms"])

# 统计分析
import statistics
print(f"平均延迟: {statistics.mean(latencies):.2f}ms")
print(f"最大延迟: {max(latencies):.2f}ms")
```

---

## 📚 相关文档

- **任务卡**: `v13_ofi_ai_system/TASKS/Stage1_真实OFI核心/✅Task_1.1.6_测试和验证.md`
- **项目规则**: `v13_ofi_ai_system/📜PROJECT_RULES.md`
- **异步日志模块**: `v13_ofi_ai_system/src/utils/async_logging.py`

---

## 🎓 学习建议 (给技术小白)

### 这个程序是做什么的？
就像一个"数据收集器"，专门负责：
1. 📡 连接到币安交易所
2. 👂 实时监听ETHUSDT的价格变化
3. 💾 把所有变化记录到文件里
4. 📊 每10秒告诉你系统运行状况

### 什么时候需要用到它？
- **现在**: 收集真实市场数据
- **未来**: 为OFI策略提供数据源

### 它和OFI计算什么关系？
- **这个程序**: 负责"收集原材料" (订单簿数据)
- **OFI计算器**: 负责"加工原材料" (计算买卖压力)
- 两者**分工明确**，互不干扰

### 我需要修改这个文件吗？
**99%不需要！** 除非：
- 想监控其他交易对 (改 `--symbol` 参数即可)
- 需要实时回调 (按上面指南添加10行代码)

---

## ✅ 版本历史

### v1.0 (2025-10-17) - 稳定版
- ✅ 解决Binance Futures REST/WS序列号不匹配问题
- ✅ 实现异步非阻塞日志
- ✅ 实现日志轮转和保留
- ✅ NDJSON数据持久化
- ✅ 实时metrics.json刷新
- ✅ 完整的序列一致性检查
- ✅ 30分钟稳态测试验证通过

**测试结果**: 所有指标完美通过，生产级质量

---

**最后更新**: 2025-10-17  
**维护状态**: ✅ 稳定，无需维护  
**下一步**: Task 1.2.1 - 创建OFI计算器

