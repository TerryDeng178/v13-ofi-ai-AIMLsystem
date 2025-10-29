# 自动重连修复方案

## 问题
当前采集器在WebSocket连接中断后会退出，缺少自动重连机制。

## 原因
在 `_handle_unified_trade_stream` 和 `_handle_unified_orderbook_stream` 方法中：
- 使用 `return_exceptions=True` 捕获异常但不重连
- 连接异常后函数直接退出
- 主循环接收到异常后进入 `finally` 并退出

## 解决方案

### 方案A: 简单外层重试（推荐，改动最小）

在外层守护进程中添加自动重启逻辑，不修改主采集器代码。

优点：
- 不破坏现有代码结构
- 实现简单
- 日志清晰

实现：
使用已有的守护进程机制（harvestd），设置 `restart: always`

### 方案B: 代码内部重连（需要较多修改）

在每个WebSocket处理方法中添加 `while` 循环：

```python
async def _handle_unified_trade_stream(self, url: str):
    max_reconnect = float(os.getenv('WSS_MAX_RECONNECT_ATTEMPTS', '0'))
    reconnect_delay = float(os.getenv('WSS_RECONNECT_DELAY', '1.0'))
    reconnect_count = 0
    
    while self.running and (max_reconnect == 0 or reconnect_count < max_reconnect):
        try:
            async with websockets.connect(...) as websocket:
                reconnect_count = 0  # 成功连接后重置
                async for message in websocket:
                    # 处理消息
                    ...
        except asyncio.CancelledError:
            break
        except Exception as e:
            reconnect_count += 1
            logger.error(f"连接错误: {e}, 等待{reconnect_delay}秒后重连...")
            await asyncio.sleep(reconnect_delay)
```

## 当前文件问题

由于之前的修改尝试，文件 `run_success_harvest.py` 出现了语法错误：
- Line 1511: Try statement must have at least one except or finally clause
- Line 1555: Expected expression
- Line 1556: Unexpected indentation

需要还原到修改前状态。

## 建议

使用方案A，利用现有的守护进程机制实现自动恢复。
当前的代码修改尝试导致了语法错误，建议先还原文件再采用方案A。

