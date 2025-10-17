# CVD实时计算测试指南

## 📋 文档信息

- **模块名称**: CVD实时计算与测试
- **版本**: v1.0.0
- **创建时间**: 2025-10-17
- **任务来源**: Task 1.2.9 - 集成Trade流和CVD计算

---

## 🎯 功能概述

本目录包含用于Binance Trade流CVD实时计算和测试的工具：

1. **`binance_trade_stream.py`**: 核心WebSocket客户端（位于`../src/`）
   - 连接Binance aggTrade流
   - 实时计算CVD指标
   - 监控指标记录（reconnect_count, queue_dropped, latency_ms）

2. **`run_realtime_cvd.py`**: 测试与数据落盘脚本
   - 运行指定时长的实时测试
   - 自动导出Parquet数据文件
   - 生成验收报告JSON

---

## 🚀 快速开始

### 1. 基础测试（10分钟）

```bash
cd v13_ofi_ai_system/examples
python run_realtime_cvd.py
```

**默认配置**:
- 交易对: ETHUSDT
- 时长: 600秒（10分钟）
- 输出: `../data/CVDTEST/`

---

### 2. 自定义测试

```bash
# 测试BTCUSDT，运行15分钟
python run_realtime_cvd.py --symbol BTCUSDT --duration 900

# 指定输出目录
python run_realtime_cvd.py --output-dir ./my_test_data

# 快速验证（3分钟）
python run_realtime_cvd.py --duration 180
```

---

### 3. 环境变量配置

```bash
# 设置环境变量
export SYMBOL=ETHUSDT
export DURATION=600
export DATA_OUTPUT_DIR=./data/CVDTEST
export PRINT_EVERY=100
export LOG_LEVEL=INFO

# 运行
python run_realtime_cvd.py
```

---

## 📊 输出文件说明

### 1. Parquet数据文件

**文件名格式**: `cvd_{symbol}_{timestamp}.parquet`

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | float | 接收时间戳（Unix秒） |
| `event_time_ms` | int | 交易所事件时间（毫秒） |
| `price` | float | 成交价格 |
| `qty` | float | 成交数量 |
| `is_buy` | bool | 买卖方向（True=买入） |
| `cvd` | float | CVD值 |
| `z_cvd` | float/None | Z-score标准化CVD |
| `ema_cvd` | float/None | EMA平滑CVD |
| `warmup` | bool | warmup状态 |
| `std_zero` | bool | 标准差为0标记 |
| `bad_points` | int | 坏数据点计数（累积） |
| `queue_dropped` | int | 队列丢弃计数（累积） |
| `reconnect_count` | int | 重连次数（累积） |
| `latency_ms` | float | 延迟（毫秒） |

---

### 2. 验收报告JSON

**文件名格式**: `report_{symbol}_{timestamp}.json`

**结构说明**:

```json
{
  "test_info": {
    "symbol": "ETHUSDT",
    "duration_planned": 600,
    "duration_actual": 601.2,
    "start_time": "2025-10-17T10:00:00",
    "end_time": "2025-10-17T10:10:01"
  },
  "data_stats": {
    "total_records": 15432,
    "avg_rate_per_sec": 25.7,
    "cvd_range": [-1234.56, 5678.90],
    "z_cvd_stats": {
      "p50": 0.12,
      "p95": 2.34,
      "p99": 3.45
    },
    "latency_stats": {
      "p50": 120.5,
      "p95": 350.2,
      "p99": 850.7
    }
  },
  "metrics": {
    "reconnect_count": 0,
    "queue_dropped": 5,
    "total_messages": 15437,
    "parse_errors": 0,
    "queue_dropped_rate": 0.0003
  },
  "validation": {
    "duration_ok": true,
    "parse_errors_ok": true,
    "queue_dropped_rate_ok": true,
    "latency_p95_ok": true,
    "reconnect_ok": true
  }
}
```

---

## ✅ 验收标准（Task 1.2.9）

### 功能验收
- ✅ **连接成功**: 持续接收数据 ≥10分钟 → `duration_ok = true`
- ✅ **解析正确**: 解析错误率 = 0 → `parse_errors_ok = true`
- ✅ **CVD连续性**: 抽样验证（在数据分析中）
- ✅ **方向判定**: m字段正确映射为is_buy

### 性能验收
- ✅ **处理延迟**: p95 < 5s (宽松阈值) → `latency_p95_ok = true`
- ✅ **稳定性**: 重连次数 ≤3次 → `reconnect_ok = true`
- ✅ **队列丢弃率**: ≤0.5% → `queue_dropped_rate_ok = true`
- ✅ **内存增长**: <30MB（手动观察）

### 输出验收
- ✅ **实时打印**: 每100条成交打印一次
- ✅ **数据落盘**: Parquet文件包含完整字段

---

## 🔧 故障排查

### 问题1: 连接失败

**症状**: 
```
WARNING Reconnect due to error: ...
```

**解决方案**:
1. 检查网络连接
2. 确认Binance Futures API可访问
3. 尝试更换交易对（BTCUSDT, ETHUSDT）

---

### 问题2: 数据量为0

**症状**: 
```
⚠️ No records collected!
```

**解决方案**:
1. 检查交易对是否正确
2. 确认WebSocket URL正确
3. 检查日志中的解析错误

---

### 问题3: 高丢弃率

**症状**: 
```
queue_dropped_rate > 0.5%
```

**解决方案**:
1. 增大队列大小: `export QUEUE_SIZE=2048`
2. 降低打印频率: `export PRINT_EVERY=200`
3. 检查系统性能

---

### 问题4: 高延迟

**症状**: 
```
latency_ms > 5000ms
```

**解决方案**:
1. 检查网络状况
2. 使用更近的Binance服务器
3. 确认系统时间同步

---

## 📈 数据分析示例

### 1. 读取Parquet文件

```python
import pandas as pd

# 读取数据
df = pd.read_parquet("../data/CVDTEST/cvd_ethusdt_20251017_100000.parquet")

# 基础统计
print(f"Total records: {len(df)}")
print(f"CVD range: {df['cvd'].min():.2f} to {df['cvd'].max():.2f}")
print(f"Avg latency: {df['latency_ms'].mean():.1f}ms")
```

---

### 2. CVD连续性验证

```python
# 验证CVD连续性（抽样1%）
sample = df.sample(frac=0.01).sort_values('timestamp')
sample['cvd_diff'] = sample['cvd'].diff()
sample['qty_signed'] = sample['qty'] * sample['is_buy'].map({True: 1, False: -1})

# 检查误差
errors = abs(sample['cvd_diff'] - sample['qty_signed']) > 1e-9
error_rate = errors.sum() / len(sample)
print(f"CVD continuity error rate: {error_rate:.4%}")
```

---

### 3. 可视化

```python
import matplotlib.pyplot as plt

# CVD时间序列
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'] - df['timestamp'].iloc[0], df['cvd'])
plt.xlabel('Time (seconds)')
plt.ylabel('CVD')
plt.title('CVD Over Time')
plt.grid(True)
plt.savefig('cvd_timeseries.png')

# Z-score分布
plt.figure(figsize=(10, 6))
df[df['z_cvd'].notna()]['z_cvd'].hist(bins=50, alpha=0.7)
plt.xlabel('Z-score')
plt.ylabel('Frequency')
plt.title('Z-score Distribution')
plt.grid(True)
plt.savefig('z_cvd_distribution.png')
```

---

## ⚠️ 注意事项

### 1. 测试时长建议
- **快速验证**: 3-5分钟（检查连接和基本功能）
- **标准测试**: 10-15分钟（Task 1.2.9验收标准）
- **稳定性测试**: 30-60分钟（Task 1.2.10长期测试）

### 2. 资源使用
- **内存**: 约10-30MB（取决于时长）
- **磁盘**: 每小时约50-100MB Parquet文件
- **网络**: 持续WebSocket连接，流量约1-5KB/s

### 3. 数据保留
- 测试数据保存在 `../data/CVDTEST/`
- 建议定期清理旧测试数据
- Parquet文件可压缩存档

---

## 🔗 相关文件

- **核心模块**: `v13_ofi_ai_system/src/binance_trade_stream.py`
- **CVD计算器**: `v13_ofi_ai_system/src/real_cvd_calculator.py`
- **CVD文档**: `v13_ofi_ai_system/src/README_CVD_CALCULATOR.md`
- **任务卡**: `v13_ofi_ai_system/TASKS/Stage1_真实OFI+CVD核心/Task_1.2.9_集成Trade流和CVD计算.md`

---

## 📞 支持与反馈

- **项目**: V13 OFI+CVD+AI System
- **任务来源**: Task 1.2.9
- **问题反馈**: 通过项目任务卡系统提交

---

**最后更新**: 2025-10-17  
**文档版本**: v1.0.0  
**状态**: ✅ 已验证（3分钟快速测试通过）

