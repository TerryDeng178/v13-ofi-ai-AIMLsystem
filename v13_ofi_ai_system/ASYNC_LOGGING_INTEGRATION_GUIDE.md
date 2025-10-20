# async_logging.py 模块集成指南

## 📋 模块概述

`async_logging.py` 是一个轻量级的异步日志模块，专为高性能场景设计。

### 🎯 核心功能

| 功能 | 说明 | 优势 |
|------|------|------|
| **异步队列** | 使用 `QueueHandler` + `QueueListener` | 非阻塞，不影响主线程性能 |
| **丢弃策略** | 队列满时自动丢弃新日志 | 保护系统，防止内存溢出 |
| **日志轮转** | 支持按时间或大小轮转 | 自动管理日志文件大小 |
| **性能监控** | 跟踪队列深度、最大深度、丢弃数 | 监控日志系统健康状况 |

---

## 🏗️ 模块架构

```python
# src/utils/async_logging.py

class DropQueueHandler(logging.handlers.QueueHandler):
    """队列处理器（支持丢弃策略）"""
    - drops: 丢弃的日志数
    - max_depth: 队列最大深度
    - emit(): 尝试将日志放入队列，满时丢弃

def setup_async_logging():
    """设置异步日志系统"""
    1. 创建队列 (Queue)
    2. 创建队列处理器 (DropQueueHandler)
    3. 创建文件处理器 (RotatingFileHandler / TimedRotatingFileHandler)
    4. 创建队列监听器 (QueueListener)
    5. 启动监听器
    
def sample_queue_metrics():
    """采样队列指标"""
    - depth: 当前队列深度
    - max_depth: 历史最大深度
    - drops: 累计丢弃数
```

---

## 🔌 在 WebSocket 中的应用

### 当前使用方式

```python
# src/binance_websocket_client.py

from utils.async_logging import setup_async_logging, sample_queue_metrics

class BinanceOrderBookStream:
    def __init__(self, ...):
        # 设置异步日志
        self.logger, self.listener, self.queue_handler = setup_async_logging(
            name="binance_ws",
            log_path="logs/binance_ws.log",
            rotate='interval',      # 按时间轮转
            rotate_sec=60,          # 每60秒轮转
            max_bytes=5_000_000,    # 文件最大5MB
            backups=7,              # 保留7个备份
            level=logging.INFO,     # 日志级别
            queue_max=10000,        # 队列最大容量
            to_console=True         # 同时输出到控制台
        )
    
    def get_metrics(self):
        # 获取日志队列指标
        queue_metrics = sample_queue_metrics(self.queue_handler)
        return {
            'log_queue_depth': queue_metrics['depth'],
            'log_drops': queue_metrics['drops'],
            ...
        }
```

### 使用场景

**高频数据流场景**：
- 订单簿更新：每秒数百条
- 交易数据流：每秒数千条
- 需要记录但不能阻塞主线程

**性能优势**：
```
同步日志：每条日志 ~1-5ms (阻塞)
异步日志：每条日志 ~0.01ms (非阻塞)
提升：100-500倍
```

---

## 🔧 集成到统一配置系统

### 1. 配置定义 (config/system.yaml)

```yaml
websocket:
  logging:
    log_level: "INFO"
    queue_max: 10000
    rotate: "interval"
    rotate_sec: 60
    max_bytes: 5000000
    backups: 7
    to_console: true
```

### 2. 配置加载器

```python
# src/async_logging_config_loader.py

from dataclasses import dataclass
from src.utils.config_loader import ConfigLoader

@dataclass
class AsyncLoggingConfig:
    level: str = "INFO"
    queue_max: int = 10000
    rotate: str = "interval"
    rotate_sec: int = 60
    max_bytes: int = 5_000_000
    backups: int = 7
    to_console: bool = True

class AsyncLoggingConfigLoader:
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
    
    def load_config(self, component: str = "websocket") -> AsyncLoggingConfig:
        component_config = self.config_loader.get(f'{component}.logging', {})
        return AsyncLoggingConfig(
            level=component_config.get('log_level', 'INFO'),
            queue_max=component_config.get('queue_max', 10000),
            # ... 其他参数
        )
```

### 3. 使用示例

```python
from src.utils.config_loader import ConfigLoader
from src.async_logging_config_loader import AsyncLoggingConfigLoader
from src.utils.async_logging import setup_async_logging

# 加载配置
config_loader = ConfigLoader()
logging_config_loader = AsyncLoggingConfigLoader(config_loader)
logging_config = logging_config_loader.load_config(component='websocket')

# 设置异步日志（使用配置）
logger, listener, queue_handler = setup_async_logging(
    name="binance_ws",
    log_path="logs/binance_ws.log",
    rotate=logging_config.rotate,
    rotate_sec=logging_config.rotate_sec,
    max_bytes=logging_config.max_bytes,
    backups=logging_config.backups,
    level=logging_config.get_log_level(),
    queue_max=logging_config.queue_max,
    to_console=logging_config.to_console
)
```

### 4. 环境变量覆盖

```bash
# 覆盖日志级别
export V13__WEBSOCKET__LOGGING__LOG_LEVEL=DEBUG

# 覆盖队列大小
export V13__WEBSOCKET__LOGGING__QUEUE_MAX=20000

# 覆盖轮转策略
export V13__WEBSOCKET__LOGGING__ROTATE=size
export V13__WEBSOCKET__LOGGING__MAX_BYTES=10000000
```

---

## 📊 性能监控

### 队列指标

```python
queue_metrics = sample_queue_metrics(queue_handler)

print(f"当前队列深度: {queue_metrics['depth']}")
print(f"历史最大深度: {queue_metrics['max_depth']}")
print(f"累计丢弃数: {queue_metrics['drops']}")
```

### Prometheus 集成

```python
from prometheus_client import Gauge, Counter

# 定义指标
log_queue_depth = Gauge('log_queue_depth', 'Async log queue depth')
log_queue_max_depth = Gauge('log_queue_max_depth', 'Async log queue max depth')
log_drops_total = Counter('log_drops_total', 'Total dropped logs')

# 更新指标
def update_log_metrics(queue_handler):
    metrics = sample_queue_metrics(queue_handler)
    log_queue_depth.set(metrics['depth'])
    log_queue_max_depth.set(metrics['max_depth'])
    log_drops_total.inc(metrics['drops'])
```

---

## 🔍 故障排查

### 问题1: 日志丢失过多

**症状**: `drops` 计数持续增长

**原因**: 队列容量不足或写入速度过快

**解决方案**:
```yaml
websocket:
  logging:
    queue_max: 20000  # 增加队列容量
    rotate_sec: 30    # 更频繁的轮转
```

### 问题2: 日志延迟

**症状**: 日志输出延迟明显

**原因**: 队列积压严重

**解决方案**:
```python
# 监控队列深度
if queue_metrics['depth'] > 8000:  # 80% 容量
    logger.warning(f"Log queue high: {queue_metrics['depth']}")
```

### 问题3: 磁盘空间不足

**症状**: 日志文件占用过多空间

**解决方案**:
```yaml
websocket:
  logging:
    backups: 3         # 减少备份数
    max_bytes: 2000000 # 减小文件大小
```

---

## 🎯 最佳实践

### 1. 队列容量设置

```python
# 估算队列容量
messages_per_sec = 1000      # 每秒消息数
log_rate = 0.1               # 10% 消息需要记录
processing_time = 0.001      # 每条日志处理时间（秒）

min_queue_size = messages_per_sec * log_rate * processing_time * 10
# = 1000 * 0.1 * 0.001 * 10 = 1

# 建议：留有余量
recommended_queue_size = min_queue_size * 100  # 100倍余量
```

### 2. 日志级别策略

| 环境 | 级别 | 说明 |
|------|------|------|
| **开发** | DEBUG | 详细调试信息 |
| **测试** | INFO | 标准运行信息 |
| **生产** | WARNING | 仅警告和错误 |

### 3. 轮转策略选择

| 策略 | 适用场景 | 配置示例 |
|------|---------|---------|
| **按时间** | 消息量稳定 | `rotate='interval', rotate_sec=3600` |
| **按大小** | 消息量波动 | `rotate='size', max_bytes=10_000_000` |

---

## 📚 相关文档

- [WebSocket配置集成](STAGE4_TRADE_STREAM_CONFIG_SUMMARY.md)
- [统一配置系统](UNIFIED_CONFIG_INTEGRATION_COMPLETE.md)
- [配置测试结果](UNIFIED_CONFIG_TEST_RESULTS.md)

---

**创建时间**: 2025-10-20  
**维护者**: V13 OFI+CVD AI System  
**状态**: ✅ 完成
