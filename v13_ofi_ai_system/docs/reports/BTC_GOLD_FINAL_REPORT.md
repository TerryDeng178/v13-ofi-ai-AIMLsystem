# Task 1.2.10 CVD计算测试报告

**测试执行时间**: 2025-10-19 21:31:57

**测试级别**: Gold（≥120分钟）

**数据源**: `v13_ofi_ai_system/data/cvd_btc_gold_test`

---

## 测试摘要

- **采集时长**: 23.8 分钟 (0.40 小时)
- **数据点数**: 999 笔
- **平均速率**: 0.70 笔/秒
- **解析错误**: 0
- **重连次数**: 0
- **队列丢弃率**: 0.0000%

---

## 验收标准对照结果

### 1. 时长与连续性
- [ ] 运行时长: 23.8分钟 (≥30分钟，分析模式基线)
- [ ] p99_interarrival: 9625.52ms (≤5000ms)
- [ ] gaps_over_10s: 8 (==0)

### 2. 数据质量
- [x] parse_errors: 0 (==0)
- [x] queue_dropped_rate: 0.0000% (≤0.5%)

### 3. 性能指标
- [x] p95_latency: 8010.795ms (<300ms)

### 4. Z-score稳健性
- [x] median(|z_cvd|): 0.0035 (≤0.5)
- [x] IQR(z_cvd): 0.0083 (∈[1.0, 2.0])
- [ ] P(|Z|>2): 15.00% (∈[1%, 8%])
- [ ] P(|Z|>3): 12.39% (<1%)
- [x] std_zero: 0 (==0)

### 5. 一致性验证（全量检查）
- [ ] 逐笔守恒: 0 错误 (容差≤1e-9)
- [ ] 首尾守恒误差: 1.00e-03 (相对容差: 5.42e-06)
- 检查样本: 999 笔 (全量)

### 6. 稳定性
- [x] 重连频率: 0.00次/小时 (≤3/小时)

---

## 图表

### 1. Z-score分布直方图
![Z-score直方图](../../figs_cvd_btc_gold/hist_z.png)

### 2. CVD时间序列
![CVD时间序列](../../figs_cvd_btc_gold/cvd_timeseries.png)

### 3. Z-score时间序列
![Z-score时间序列](../../figs_cvd_btc_gold/z_timeseries.png)

### 4. 延迟箱线图
![延迟箱线图](../../figs_cvd_btc_gold/latency_box.png)

### 5. 消息到达间隔分布
![Interarrival分布](../../figs_cvd_btc_gold/interarrival_hist.png)

**Interarrival统计**:
- P50: 0.0ms
- P95: 6166.4ms
- P99: 9625.5ms
- Max: 13642.1ms

### 6. Event ID差值分布
![Event ID差值](../../figs_cvd_btc_gold/event_id_diff.png)

**aggTradeId检查**:
- 重复ID: 0 (0.000%)
- 倒序ID: 0 (0.000%)
- 大跳跃(>10k): 0
- event_time_ms同毫秒: 134 (13.4%, 信息项)

---

## 结论

**验收标准通过率**: 4/8 (50.0%)

**⚠️ 部分验收标准未通过**

需要关注的指标:
- ⚠️ 运行时长未达标
- ⚠️ 数据连续性未达标
- ⚠️ Z-score分布未达标
- ⚠️ CVD连续性验证未通过
