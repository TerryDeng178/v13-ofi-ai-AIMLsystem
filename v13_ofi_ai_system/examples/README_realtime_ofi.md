# Realtime OFI Calculator

实时OFI（Order Flow Imbalance）计算示例，集成WebSocket数据流和OFI计算器。

## 🚀 快速开始

### 1. DEMO模式（本地仿真）

无需外部依赖，立即运行：

```bash
cd v13_ofi_ai_system/examples
python run_realtime_ofi.py --demo
```

**输出示例**：
```
[INFO] Signal handlers configured (Windows mode: SIGINT only)
[INFO] OFI Calculator initialized: symbol=DEMO-USD, K=5, z_window=300, ema_alpha=0.2
[INFO] Running in DEMO mode (local synthetic orderbook, 50 Hz)
DEMO-USD OFI=+0.08767  Z=None  EMA=+0.08767  warmup=True  std_zero=False
DEMO-USD OFI=-0.51964  Z=-0.968  EMA=-0.02848  warmup=False  std_zero=False
```

### 2. 真实WebSocket模式

需要安装 `websockets`：

```bash
pip install websockets
```

**设置环境变量**：

```bash
# Linux/Mac
export WS_URL="wss://your-websocket-endpoint"
export SYMBOL="BTCUSDT"
export K_LEVELS="5"
export Z_WINDOW="300"
export EMA_ALPHA="0.2"

python run_realtime_ofi.py
```

```powershell
# Windows PowerShell
$env:WS_URL="wss://your-websocket-endpoint"
$env:SYMBOL="BTCUSDT"
$env:K_LEVELS="5"
$env:Z_WINDOW="300"
$env:EMA_ALPHA="0.2"

python run_realtime_ofi.py
```

---

## ⚙️ 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `WS_URL` | `""` | WebSocket URL（空则使用DEMO模式） |
| `SYMBOL` | `"DEMO-USD"` | 交易对符号 |
| `K_LEVELS` | `5` | 订单簿档位数（前K档） |
| `Z_WINDOW` | `300` | Z-score滚动窗口大小 |
| `EMA_ALPHA` | `0.2` | EMA平滑系数（0-1） |

### 输出字段说明

每条输出包含：
- **symbol**: 交易对符号
- **OFI**: Order Flow Imbalance值（-∞ ~ +∞）
  - 正值：买入压力 > 卖出压力
  - 负值：卖出压力 > 买入压力
- **Z**: Z-score标准化后的OFI
  - `None`: warmup期间
  - 典型范围：-3 ~ +3
  - |Z| > 2：强信号
- **EMA**: 指数移动平均平滑后的OFI
- **warmup**: 是否在warmup期（Z-score未就绪）
- **std_zero**: 标准差是否为0（低波动期）

### 性能指标（每60秒输出）

```
[STAT] window=60.0s processed=3000 p50=0.123ms p95=0.456ms dropped=0 parse_errors=0 queue_depth=0
```

- **window**: 统计窗口时长
- **processed**: 处理的消息数
- **p50/p95**: 处理延迟的50th/95th百分位
- **dropped**: 背压保护丢弃的消息数
- **parse_errors**: 解析错误数
- **queue_depth**: 当前队列深度

---

## 🛠️ 排障指南

### 常见错误1: 无法导入 `real_ofi_calculator`

**错误信息**：
```
Cannot import real_ofi_calculator. Ensure it exists in project src or same directory.
```

**解决方案**：
1. 确认文件结构：
   ```
   v13_ofi_ai_system/
   ├── src/
   │   └── real_ofi_calculator.py
   └── examples/
       └── run_realtime_ofi.py
   ```
2. 从 `v13_ofi_ai_system/examples/` 目录运行

### 常见错误2: websockets 未安装

**错误信息**：
```
[ERROR] websockets not installed. Use: pip install websockets  (or run with --demo)
```

**解决方案**：
```bash
pip install websockets
```
或使用 `--demo` 模式测试

### 常见错误3: 连接超时/重连循环

**现象**：
```
[WARN] WS disconnected: TimeoutError; reconnect in 1s
[WARN] WS disconnected: TimeoutError; reconnect in 2s
```

**可能原因**：
1. WebSocket URL 错误
2. 网络连接问题
3. 服务端需要订阅消息

**解决方案**：
1. 检查 `WS_URL` 格式：`wss://host/path`
2. 测试网络连接：`ping host`
3. 查看WebSocket服务端文档，可能需要在 `ws_consume` 中发送订阅消息：
   ```python
   # 在 ws_consume 函数中，连接后发送订阅
   await ws.send(json.dumps({"subscribe": "depth", "symbol": "BTCUSDT"}))
   ```

### 常见错误4: 60秒无数据自动重连

**现象**：
```
[WARN] No data for 60s, triggering reconnect (heartbeat timeout)
```

**说明**：这是**正常的心跳机制**，60秒无数据会触发重连，避免僵死连接。

### 常见错误5: 背压警告频繁

**现象**：
```
[WARN] Backpressure: skipped 10 stale frames (queue depth was 11)
```

**说明**：
- 消费速度 < 生产速度
- 系统自动丢弃陈旧数据，保留最新帧
- **这是保护机制，非错误**

**优化建议**：
1. 降低数据源频率
2. 简化OFI计算逻辑
3. 增加队列大小（不推荐，可能导致延迟）

---

## 🔧 高级配置

### 修改消息解析格式

如果WebSocket返回的消息格式不同，修改 `parse_message()` 函数：

```python
def parse_message(msg: str) -> Optional[Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]]:
    """
    自定义解析逻辑
    返回: (bids, asks)
    bids: [(price, qty), ...] 降序
    asks: [(price, qty), ...] 升序
    """
    try:
        data = json.loads(msg)
        # 修改这里以适配不同的消息格式
        bids = data.get("bids", [])  # 或 data["data"]["bids"]
        asks = data.get("asks", [])  # 或 data["data"]["asks"]
        
        # 标准化处理
        bids = topk_pad(bids, K_LEVELS, reverse=True)
        asks = topk_pad(asks, K_LEVELS, reverse=False)
        return bids, asks
    except Exception:
        return None
```

### 发送订阅消息

在 `ws_consume()` 函数中添加：

```python
async with websockets.connect(url, ping_interval=20, close_timeout=5) as ws:
    backoff = 1
    # 发送订阅消息
    subscribe_msg = json.dumps({
        "method": "SUBSCRIBE",
        "params": [f"{SYMBOL.lower()}@depth@100ms"],
        "id": 1
    })
    await ws.send(subscribe_msg)
    print(f"[INFO] Sent subscription: {subscribe_msg}")
    
    while not stop.is_set():
        # ...
```

### 调整重连参数

修改 `ws_consume()` 中的参数：

```python
# 初始退避时间（秒）
backoff = 1  # 改为 2 或 5

# 最大退避时间（秒）
backoff = min(backoff*2, 30)  # 改为 60 或更大

# 心跳超时时间（秒）
msg = await asyncio.wait_for(ws.recv(), timeout=60)  # 改为 120 或更大
```

---

## 📊 性能基准

**测试环境**：
- CPU: Intel i5-8250U @ 1.60GHz
- RAM: 8GB
- Python: 3.10
- OS: Windows 10

**DEMO模式（50 Hz）**：
- 平均延迟: 0.1-0.2 ms
- p95 延迟: 0.3-0.5 ms
- 内存占用: ~50 MB（稳态）
- CPU占用: ~5%

**真实WebSocket（100 msgs/s）**：
- 平均延迟: < 1 ms
- p95 延迟: < 2 ms
- 内存占用: ~70 MB（稳态）
- CPU占用: ~10%

---

## 🔍 日志级别说明

| 级别 | 用途 | 示例 |
|------|------|------|
| **[INFO]** | 正常操作流程 | 连接、重连成功、配置初始化 |
| **[WARN]** | 可恢复的异常 | 跳帧、心跳超时、断连重连 |
| **[ERROR]** | 需要关注的错误 | 解析错误、异常捕获 |
| **[STAT]** | 性能统计 | p50/p95/队列深度/丢帧数 |

---

## 🛡️ 生产环境建议

### 1. 日志持久化

将日志输出到文件：

```bash
python run_realtime_ofi.py --demo > ofi_$(date +%Y%m%d_%H%M%S).log 2>&1
```

### 2. 监控告警

关注以下指标：
- `parse_errors > 0`: 消息格式不兼容
- `dropped > 100`: 背压严重
- `p95_ms > 10`: 性能下降
- `reconnects > 5/min`: 网络不稳定

### 3. 自动重启

使用 `systemd`、`supervisor` 或 `pm2` 管理进程：

```bash
# 使用 supervisor
[program:realtime_ofi]
command=python /path/to/run_realtime_ofi.py
directory=/path/to/v13_ofi_ai_system/examples
autostart=true
autorestart=true
stderr_logfile=/var/log/ofi.err.log
stdout_logfile=/var/log/ofi.out.log
```

### 4. 资源限制

设置内存和CPU限制，避免资源耗尽：

```bash
# Linux: 使用 ulimit
ulimit -v 500000  # 限制虚拟内存 500MB
python run_realtime_ofi.py
```

---

## 📝 许可证

MIT License

---

## 🙋 常见问题

**Q: 如何停止程序？**
A: 按 `Ctrl+C`，程序会优雅退出，关闭所有连接。

**Q: warmup期多长？**
A: 默认 `max(5, z_window//5)` 条消息。z_window=300时，warmup=60条。

**Q: 如何验证OFI计算正确？**
A: 运行 `--demo` 模式，观察：
   - 买入增加/卖出减少时，OFI > 0
   - 买入减少/卖出增加时，OFI < 0
   - warmup结束后，Z-score开始计算

**Q: 支持多个交易对吗？**
A: 当前版本仅支持单交易对。多交易对需要创建多个实例。

---

**文档版本**: 1.0  
**最后更新**: 2025-10-17  
**维护者**: V13 OFI+CVD+AI System

