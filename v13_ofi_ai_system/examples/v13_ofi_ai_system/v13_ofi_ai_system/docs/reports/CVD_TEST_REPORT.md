# Task 1.2.10 CVD计算测试报告

**测试执行时间**: 2025-10-18 12:12:04

**测试级别**: Gold（≥120分钟）

**数据源**: `C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\v13_ofi_ai_system\data\cvd_p0b_60min_ethusdt`

---

## 测试摘要

- **采集时长**: 59.9 分钟 (1.00 小时)
- **数据点数**: 4,519 笔
- **平均速率**: 1.26 笔/秒
- **解析错误**: 0
- **重连次数**: 0
- **队列丢弃率**: 0.0000%

---

## 验收标准对照结果

### 1. 时长与连续性
- [ ] 运行时长: 59.9分钟 (≥120分钟)
- [ ] max_gap_ms: 8156.04ms (≤2000ms)

### 2. 数据质量
- [x] parse_errors: 0 (==0)
- [x] queue_dropped_rate: 0.0000% (≤0.5%)

### 3. 性能指标
- [ ] p95_latency: 4893.410ms (<300ms)

### 4. Z-score稳健性
- [ ] median(|z_cvd|): 1.6910 (≤0.5)
- [ ] IQR(z_cvd): 3.2399 (∈[1.0, 2.0])
- [ ] P(|Z|>2): 35.77% (∈[1%, 8%])
- [ ] P(|Z|>3): 12.51% (<1%)
- [x] std_zero: 0 (==0)

### 5. 一致性验证（全量检查）
- [ ] 逐笔守恒: 0 错误 (容差≤1e-9)
- [ ] 首尾守恒误差: 6.00e-03 (≤1e-6)
- 检查样本: 4519 笔 (全量)

### 6. 稳定性
- [x] 重连频率: 0.00次/小时 (≤3/小时)

---

## 图表

### 1. Z-score分布直方图
![Z-score直方图](../../../../../figs_cvd_p0b_60min/hist_z.png)

### 2. CVD时间序列
![CVD时间序列](../../../../../figs_cvd_p0b_60min/cvd_timeseries.png)

### 3. Z-score时间序列
![Z-score时间序列](../../../../../figs_cvd_p0b_60min/z_timeseries.png)

### 4. 延迟箱线图
![延迟箱线图](../../../../../figs_cvd_p0b_60min/latency_box.png)

### 5. 消息到达间隔分布
![Interarrival分布](../../../../../figs_cvd_p0b_60min/interarrival_hist.png)

**Interarrival统计**:
- P50: 0.0ms
- P95: 3801.1ms
- P99: 4688.5ms
- Max: 8156.0ms

### 6. Event ID差值分布
![Event ID差值](../../../../../figs_cvd_p0b_60min/event_id_diff.png)

**aggTradeId检查**:
- 重复ID: 0 (0.000%)
- 倒序ID: 0 (0.000%)
- 大跳跃(>10k): 0
- event_time_ms同毫秒: 706 (15.6%, 信息项)

---

## 结论

**验收标准通过率**: 3/8 (37.5%)

**⚠️ 部分验收标准未通过**

需要关注的指标:
- ⚠️ 运行时长未达标
- ⚠️ 数据连续性未达标
- ⚠️ Z-score分布未达标
- ⚠️ CVD连续性验证未通过
