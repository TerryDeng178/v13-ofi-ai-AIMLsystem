# Task 1.3.1 v2 数据采集系统使用指南

## 📋 概述

Task 1.3.1 v2 是一个完整的OFI+CVD数据采集系统，支持48-72小时连续数据采集，产出5类分区化数据集，并提供完整的监控和验证功能。

## 🚀 快速开始

### 1. 环境准备

确保已安装必要的依赖包：

```bash
pip install pandas pyarrow prometheus_client websockets numpy
```

### 2. 一键启动（推荐）

**Windows批处理脚本：**
```cmd
scripts\start_harvest.bat
```

**PowerShell脚本：**
```powershell
scripts\start_harvest.ps1
```

### 3. 手动启动

```bash
# 设置环境变量
set SYMBOLS=BTCUSDT,ETHUSDT
set RUN_HOURS=72
set Z_MODE=delta
set SCALE_MODE=hybrid

# 运行采集脚本
python examples/run_realtime_harvest.py
```

## 📊 输出数据结构

### 目录结构
```
data/ofi_cvd/
  date=2025-01-20/
    symbol=BTCUSDT/
      kind=prices/
        part-*.parquet
      kind=ofi/
        part-*.parquet
      kind=cvd/
        part-*.parquet
      kind=fusion/
        part-*.parquet
      kind=events/
        part-*.parquet
    symbol=ETHUSDT/
      ...
artifacts/
  run_logs/harvest_20250120_1430.log
  dq_reports/dq_20250120_1430.json
  state/checkpoint.json
```

### 数据表结构

#### prices表
- `ts_ms`: 时间戳（毫秒）
- `event_ts_ms`: 事件时间戳
- `symbol`: 交易对
- `price`: 价格
- `qty`: 数量
- `agg_trade_id`: 聚合交易ID
- `latency_ms`: 延迟（毫秒）
- `recv_rate_tps`: 接收速率（TPS）

#### ofi表
- `ts_ms`: 时间戳
- `symbol`: 交易对
- `ofi_value`: OFI值
- `ofi_z`: OFI Z-score
- `scale`: 尺度
- `regime`: 市场状态

#### cvd表
- `ts_ms`: 时间戳
- `symbol`: 交易对
- `cvd`: CVD值
- `delta`: 增量
- `z_raw`: 原始Z-score
- `z_cvd`: CVD Z-score
- `scale`: 尺度
- `sigma_floor`: 地板值
- `floor_used`: 地板使用标志
- `regime`: 市场状态

#### fusion表
- `ts_ms`: 时间戳
- `symbol`: 交易对
- `score`: 融合分数
- `score_z`: 融合Z-score
- `regime`: 市场状态

#### events表
- `ts_ms`: 时间戳
- `symbol`: 交易对
- `event_type`: 事件类型
- `meta_json`: 元数据JSON

## 🔧 配置参数

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `SYMBOLS` | BTCUSDT,ETHUSDT | 交易对列表 |
| `RUN_HOURS` | 72 | 运行时长（小时） |
| `PARQUET_ROTATE_SEC` | 60 | Parquet文件滚动间隔（秒） |
| `WSS_PING_INTERVAL` | 20 | WebSocket心跳间隔（秒） |
| `DEDUP_LRU` | 8192 | 去重缓存大小 |
| `Z_MODE` | delta | Z-score计算模式 |
| `SCALE_MODE` | hybrid | 尺度计算模式 |
| `MAD_MULTIPLIER` | 1.8 | MAD乘数 |
| `SCALE_FAST_WEIGHT` | 0.20 | 快速尺度权重 |
| `HALF_LIFE_SEC` | 600 | 半衰期（秒） |
| `WINSOR_LIMIT` | 8 | Winsor限制 |
| `PROMETHEUS_PORT` | 8009 | Prometheus端口 |
| `LOG_LEVEL` | INFO | 日志级别 |

### 配置文件

主要配置在 `config/system.yaml` 的 `data_harvest` 部分：

```yaml
data_harvest:
  symbols: ["BTCUSDT", "ETHUSDT"]
  run_hours: 72
  parquet_rotate_sec: 60
  websocket:
    ping_interval: 20
    heartbeat_timeout: 30
  data_quality:
    precheck_minutes: 10
    acceptance_criteria:
      max_empty_bucket_rate: 0.001
      max_duplicate_rate: 0.005
      max_latency_p99_ms: 120
      max_latency_p50_ms: 60
      min_events_per_72h: 1000
      min_winsor_effect: 0.1
```

## 📈 监控系统

### 1. 设置监控环境

```cmd
scripts\setup_monitoring.bat
```

### 2. 访问监控界面

- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **指标端点**: http://localhost:8009/metrics

### 3. 监控指标

#### 核心指标
- `recv_rate_tps`: 接收速率（TPS）
- `ws_reconnects_total`: WebSocket重连次数
- `dedup_hits_total`: 去重命中次数
- `latency_ms`: 延迟分布
- `cvd_scale_median`: CVD尺度中位数
- `cvd_floor_hit_rate`: CVD地板命中率
- `data_rows_total`: 数据行数
- `write_errors_total`: 写入错误次数

#### 告警规则
- 延迟过高（P99 > 120ms）
- WebSocket重连频繁（>10次/小时）
- CVD Floor命中率过高（>60%）
- 数据写入错误
- 接收速率过低
- 服务不可用

## 🧪 数据质量验证

### 1. 自动验证

采集完成后会自动运行验证脚本。

### 2. 手动验证

```bash
python scripts/validate_ofi_cvd_harvest.py --base-dir data/ofi_cvd --output-dir artifacts/dq_reports
```

### 3. 验收标准（DoD）

| 指标 | 标准 | 说明 |
|------|------|------|
| 空桶率 | < 0.1% | 按1分钟桶聚合 |
| 重复率 | < 0.5% | 按agg_trade_id去重 |
| 延迟P99 | < 120ms | 网络延迟 |
| 延迟P50 | < 60ms | 网络延迟 |
| 事件总数 | ≥ 1000 | 72小时内 |
| Winsor效果 | ≥ 10% | CVD一致性 |

## 🔄 稳定性与恢复

### 1. 检查点机制

- 自动保存到 `artifacts/state/checkpoint.json`
- 包含最后offset和时间戳
- 支持进程重启后恢复

### 2. 自动恢复

- WebSocket断线自动重连
- 写入失败自动重试（最多3次）
- 错误时跳过而不中断流

### 3. 错误处理

- 最大错误率限制（100次/小时）
- 错误冷却期（300秒）
- 详细错误日志记录

## 📝 日志和报告

### 1. 日志文件

- 位置: `artifacts/run_logs/`
- 格式: `harvest_YYYYMMDD_HHMM.log`
- 级别: INFO（可配置）

### 2. 数据质量报告

- 位置: `artifacts/dq_reports/`
- 格式: `dq_YYYYMMDD_HHMM.json`
- 内容: 完整性、去重、延迟、信号量、一致性统计

### 3. 检查点文件

- 位置: `artifacts/state/checkpoint.json`
- 内容: 最后处理位置和时间戳

## ⚠️ 注意事项

### 1. 系统要求

- **磁盘空间**: 预估2-5GB（BTC/ETH 72小时）
- **内存**: 建议4GB以上
- **网络**: 稳定的互联网连接
- **写入速度**: ≥ 10MB/s

### 2. 时间同步

- 以`event_ts_ms`为主时钟
- 系统时间漂移>200ms会记录并上报
- 建议使用NTP同步

### 3. 权限要求

- 输出目录可写权限
- 异常退出不会遗留半写文件
- 使用临时文件名+原子重命名

## 🚨 故障排除

### 1. 常见问题

**Q: WebSocket连接失败**
A: 检查网络连接，查看重连日志

**Q: 数据写入错误**
A: 检查磁盘空间和权限

**Q: 延迟过高**
A: 检查网络状况，考虑调整参数

**Q: 重复率过高**
A: 检查去重缓存大小设置

### 2. 调试模式

```bash
set LOG_LEVEL=DEBUG
python examples/run_realtime_harvest.py
```

### 3. 预检模式

```bash
python examples/run_realtime_harvest.py --precheck-only
```

## 📚 相关文档

- [Task 1.3.1 任务文档](../TASKS/Stage1_真实OFI+CVD核心/Task_1.3.1_收集历史OFI+CVD数据.md)
- [系统配置文档](../config/system.yaml)
- [Grafana仪表板](../grafana/dashboards/ofi_cvd_harvest.json)
- [Prometheus配置](../grafana/prometheus.yml)

## 🔗 相关链接

- 上一个任务: [Task_1.2.13_CVD_Z-score微调优化](../TASKS/Stage1_真实OFI+CVD核心/Task_1.2.13_CVD_Z-score微调优化.md)
- 下一个任务: [Task_1.3.2_创建OFI+CVD信号分析工具](../TASKS/Stage1_真实OFI+CVD核心/Task_1.3.2_创建OFI+CVD信号分析工具.md)
- 阶段总览: [📋V13_TASK_CARD.md](../📋V13_TASK_CARD.md)
