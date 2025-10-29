# OFI+CVD 数据采集器使用指南

## 概述

`run_success_harvest.py` 是一个高性能的实时OFI+CVD数据采集系统，从Binance Futures WebSocket实时采集交易数据和订单簿数据，并计算多个技术指标。

### 核心功能

1. **实时数据采集**：WebSocket连接Binance Futures，采集交易流和订单簿流
2. **多指标计算**：OFI（订单流不平衡）、CVD（累积成交量差额）、融合信号、背离检测
3. **2×2场景标签**：基于活跃度和波动率的市场状态分类
4. **数据分仓存储**：权威库（raw）和预览库（preview）分离
5. **自动轮转**：定时保存数据到Parquet文件
6. **性能监控**：延迟统计、健康检查、去重处理

---

## 安装与依赖

### 系统要求

- Python 3.11+
- Windows 10+ 或 Linux

### 依赖包

```bash
pip install asyncio websockets pandas numpy pyarrow pathlib
```

### 核心组件

确保以下核心组件已部署：

- `real_ofi_calculator.py` - OFI计算器
- `real_cvd_calculator.py` - CVD计算器
- `ofi_cvd_fusion.py` - 融合计算器
- `ofi_cvd_divergence.py` - 背离检测器

---

## 快速开始

### 1. 基本运行

```bash
# 默认配置（6个主流币种，运行1小时）
python deploy/run_success_harvest.py
```

### 2. 自定义配置

```bash
# 设置环境变量
export SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT"
export RUN_HOURS=24
export OUTPUT_DIR="deploy/data/ofi_cvd"

# 运行
python deploy/run_success_harvest.py
```

### 3. Windows批处理（推荐）

```cmd
# 使用配套的启动脚本
deploy\start_harvestd.bat
```

---

## 配置参数详解

### 环境变量

#### 基本配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `SYMBOLS` | `BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT` | 交易对列表（逗号分隔） |
| `RUN_HOURS` | `1` | 运行时长（小时） |
| `OUTPUT_DIR` | `deploy/data/ofi_cvd` | 输出目录 |

#### 数据采集配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PARQUET_ROTATE_SEC` | `60` | 文件轮转间隔（秒） |
| `WSS_PING_INTERVAL` | `20` | WebSocket心跳间隔（秒） |
| `WSS_HEARTBEAT_TIMEOUT` | `30` | WebSocket心跳超时（秒） |
| `WSS_RECONNECT_DELAY` | `1.0` | 重连延迟（秒） |
| `DEDUP_LRU` | `8192` | 去重缓存大小 |

#### 订单簿配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `ENABLE_ORDERBOOK` | `1` | 启用订单簿收集（0/1） |
| `ORDERBOOK_ROTATE_SEC` | `60` | 订单簿轮转间隔（秒） |

#### CVD配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `Z_MODE` | `delta` | Z值模式 |
| `SCALE_MODE` | `hybrid` | 缩放模式 |
| `MAD_MULTIPLIER` | `1.8` | MAD乘数 |
| `SCALE_FAST_WEIGHT` | `0.20` | 快速权重 |
| `HALF_LIFE_SEC` | `600` | 半衰期（秒） |
| `WINSOR_LIMIT` | `8.0` | 截断限值 |

#### 场景标签配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `WIN_SECS` | `300` | 时间窗口（秒） |
| `ACTIVE_TPS` | `2.0` | 活跃模式阈值（笔/秒） |
| `VOL_SPLIT` | `0.5` | 波动率分割点 |
| `SCENARIO_SCHEME` | `regime2x2` | 场景方案 |
| `FEE_TIER` | `TM` | 手续费等级 |

#### 预览目录配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PREVIEW_DIR` | `deploy/preview/ofi_cvd` | 预览库目录 |

#### 高级功能

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PAPER_ENABLE` | `0` | 启用Paper交易（0/1） |

---

## 数据输出

### 目录结构

```
deploy/
├── data/ofi_cvd/                    # 权威原始库（raw）
│   └── date=2024-01-01/
│       ├── symbol=BTCUSDT/
│       │   ├── kind=prices/         # 交易价格数据
│       │   └── kind=orderbook/      # 订单簿数据
│       └── symbol=ETHUSDT/
│           └── ...
└── preview/ofi_cvd/                 # 预览分析库
    └── date=2024-01-01/
        ├── symbol=BTCUSDT/
        │   ├── kind=ofi/            # OFI指标
        │   ├── kind=cvd/            # CVD指标
        │   ├── kind=fusion/         # 融合信号
        │   ├── kind=events/         # 事件检测
        │   └── kind=features/       # 特征宽表
        └── ...
```

### 数据格式

#### 权威库（raw）- prices

| 字段 | 类型 | 说明 |
|------|------|------|
| `ts_ms` | int64 | 事件时间戳（毫秒） |
| `recv_ts_ms` | int64 | 接收时间戳（毫秒） |
| `symbol` | string | 交易对 |
| `price` | float | 价格 |
| `qty` | float | 数量 |
| `agg_trade_id` | int64 | 聚合交易ID |
| `latency_ms` | int32 | 延迟（毫秒） |
| `best_buy_fill` | float | 买入可得价 |
| `best_sell_fill` | float | 卖出可得价 |
| `row_id` | string | 行ID（MD5） |

#### 权威库（raw）- orderbook

| 字段 | 类型 | 说明 |
|------|------|------|
| `ts_ms` | int64 | 事件时间戳（毫秒） |
| `symbol` | string | 交易对 |
| `best_bid` | float | 最优买价 |
| `best_ask` | float | 最优卖价 |
| `mid` | float | 中间价 |
| `spread_bps` | float | 价差（基点） |
| `first_id` | int64 | 首次更新ID |
| `last_id` | int64 | 最后更新ID |
| `prev_last_id` | int64 | 前一次最后ID |
| `d_bid_qty_agg` | float | 买单量变化（聚合） |
| `d_ask_qty_agg` | float | 卖单量变化（聚合） |
| `d_b0` ~ `d_b4` | float | 买单逐档变化 |
| `d_a0` ~ `d_a4` | float | 卖单逐档变化 |

#### 预览库（preview）- ofi

| 字段 | 类型 | 说明 |
|------|------|------|
| `ts_ms` | int64 | 时间戳 |
| `ofi_value` | float | OFI值 |
| `ofi_z` | float | OFI Z-score |
| `scale` | float | 缩放因子 |
| `lag_ms_to_trade` | int32 | 到交易的滞后（毫秒） |
| `scenario_2x2` | string | 场景标签 |

#### 预览库（preview）- cvd

| 字段 | 类型 | 说明 |
|------|------|------|
| `ts_ms` | int64 | 时间戳 |
| `cvd` | float | CVD值 |
| `delta` | float | Delta值 |
| `z_raw` | float | 原始Z-score |
| `z_cvd` | float | 截断后Z-score |
| `scenario_2x2` | string | 场景标签 |

#### 预览库（preview）- fusion

| 字段 | 类型 | 说明 |
|------|------|------|
| `ts_ms` | int64 | 时间戳 |
| `score` | float | 融合分数 |
| `proba` | float | 概率 |
| `consistency` | float | 一致性 |
| `dispersion` | float | 离散度 |
| `signal` | string | 信号（buy/sell/neutral） |
| `scenario_2x2` | string | 场景标签 |

---

## 使用场景

### 场景1: 日度数据采集

```bash
# 采集24小时数据
export RUN_HOURS=24
export SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT"
python deploy/run_success_harvest.py
```

### 场景2: 长期监控

```bash
# 使用守护进程模式
deploy\start_harvestd.bat

# 或使用systemd（Linux）
sudo systemctl start harvestd
```

### 场景3: 特定交易对

```bash
# 只采集BTC和ETH
export SYMBOLS="BTCUSDT,ETHUSDT"
export RUN_HOURS=12
python deploy/run_success_harvest.py
```

### 场景4: 自定义输出目录

```bash
# 输出到自定义目录
export OUTPUT_DIR="/path/to/custom/data"
python deploy/run_success_harvest.py
```

---

## 性能与监控

### 性能指标

- **延迟**：P50 < 100ms, P99 < 500ms
- **吞吐量**：单symbol约1000-5000条/秒
- **内存占用**：约200-500MB（取决于symbol数量）
- **磁盘占用**：约10-50MB/小时（取决于活跃度）

### 健康检查

采集器内置健康检查机制：

```python
# 在代码中
self.health_check_interval = 60  # 每60秒检查一次
self.data_timeout = 300  # 5分钟数据超时
self.max_connection_errors = 10  # 最大连接错误次数
```

### 监控端点

如果通过守护进程运行，可访问：

- `http://localhost:8088/health` - 健康检查
- `http://localhost:8088/orderbook` - 订单簿状态
- `http://localhost:8088/metrics` - Prometheus指标

---

## 故障排查

### 问题1: 连接失败

**症状**：无法连接到Binance WebSocket

**解决方案**：
1. 检查网络连接
2. 验证防火墙设置
3. 尝试使用代理（如果需要）

### 问题2: 数据质量异常

**症状**：延迟过高、缺失数据

**解决方案**：
1. 检查网络带宽
2. 减少并发symbol数量
3. 调整WebSocket参数（重连延迟、超时）

### 问题3: 内存占用过高

**症状**：系统内存不足

**解决方案**：
1. 减小`DEDUP_LRU`缓存大小
2. 增加`PARQUET_ROTATE_SEC`轮转频率
3. 减少symbol数量

### 问题4: 文件写入失败

**症状**：Permission denied

**解决方案**：
1. 检查输出目录权限
2. 确保磁盘空间充足
3. 手动创建目录结构

---

## 最佳实践

### 1. 生产环境配置

```bash
# 推荐配置
export SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT"
export RUN_HOURS=72  # 3天
export PARQUET_ROTATE_SEC=300  # 5分钟轮转
export DEDUP_LRU=16384  # 增大缓存
export ENABLE_ORDERBOOK=1  # 启用订单簿
```

### 2. 监控建议

- 定期检查日志文件
- 监控磁盘使用情况
- 设置数据质量告警
- 使用验证脚本检查数据完整性

### 3. 性能优化

- 减少不必要的symbol
- 调整时间窗口参数
- 优化去重缓存大小
- 使用SSD存储

---

## 开发与扩展

### 自定义指标

```python
# 在采集器中添加自定义指标
def _calculate_custom_indicator(self, symbol, data):
    # 实现你的指标逻辑
    return indicator_value
```

### 集成自定义分析

```python
# 在_process_trade_data中调用
if os.getenv("CUSTOM_ANALYSIS", "0") == "1":
    result = self._calculate_custom_indicator(symbol, trade_data)
```

### 数据导出

```python
# 使用pandas读取Parquet
import pandas as pd

df = pd.read_parquet("deploy/data/ofi_cvd/date=2024-01-01/symbol=BTCUSDT/kind=prices/part-*.parquet")
```

---

## 技术支持

### 日志位置

- 运行日志：`deploy/artifacts/run_logs/`
- 数据质量报告：`deploy/artifacts/dq_reports/`

### 常见问题

参考 `deploy/README.md` 中的故障排查章节。

### 联系方式

如有问题，请查看项目文档或提交issue。

---

## 附录

### A. 场景标签说明

| 标签 | 含义 |
|------|------|
| `A_H` | Active High - 高活跃度高波动 |
| `A_L` | Active Low - 高活跃度低波动 |
| `Q_H` | Quiet High - 低活跃度高波动 |
| `Q_L` | Quiet Low - 低活跃度低波动 |

### B. 2×2场景矩阵

```
        高活跃度(A)    低活跃度(Q)
高波动(H)   A_H         Q_H
低波动(L)   A_L         Q_L
```

### C. 时间对齐说明

- **事件时间**：交易所生成数据的时间
- **接收时间**：采集器收到数据的时间
- **延迟**：receive_time - event_time

时间对齐确保OFI计算使用交易发生时刻的订单簿快照。
