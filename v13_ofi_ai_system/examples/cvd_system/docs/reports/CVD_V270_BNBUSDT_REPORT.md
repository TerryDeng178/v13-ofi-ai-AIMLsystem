# Task 1.2.10 CVD计算测试报告 (v2.1, CVD-only)

**测试执行时间**: 2025-10-27 19:57:20

**测试级别**: Gold（≥120分钟）

**数据源**: `deploy/preview/ofi_cvd/date=2025-10-27/symbol=BNBUSDT/kind=cvd`

---

## 测试摘要

- **采集时长**: 348.2 分钟 (5.80 小时)
- **数据点数**: 77,660 笔
- **平均速率**: 3.72 笔/秒
- **解析错误**: 0
- **队列丢弃率**: 560.5473%

---

## 验收标准对照结果（CVD）

### 1. 时长与连续性
- [x] 运行时长: 348.2分钟 (≥30分钟)
- [ ] p99_interarrival: 1362.00ms (≤5000ms)
- [ ] gaps_over_10s: 2 (==0)

### 2. 数据质量
- [x] parse_errors: 0 (==0)
- [ ] queue_dropped_rate: 560.5473% (≤0.5%)

### 3. 性能（信息项）
- [x] p95_latency: 10508.100ms （信息项，不阻断）

### 4. CVD Z-score稳健性
- [ ] median(|z_cvd|): 1.2468 (≤1.0)
- [x] IQR(z_cvd): 2.4967 （参考值，不阻断）
- [ ] P(|Z|>2): 19.72% (≤8%)
- [ ] P(|Z|>3): 4.38% (≤2%)
- [x] std_zero: 0 (==0)
- [x] warmup占比: 0.00% (≤10%)

### 5. 一致性验证（跳过一次性验证（无qty/is_buy））
- [x] 逐笔守恒错误: 0
- [ ] 首尾守恒误差: 0.00e+00 (容差: 0.00e+00)

### 6. 稳定性
- [x] 重连频率: 0.52次/小时 (≤3/小时)

---

## 图表

### 1. Z-score分布直方图
![Z-score直方图](../../figs_v270/BNBUSDT/cvd_hist_z.png)

### 2. CVD时间序列
![CVD时间序列](../../figs_v270/BNBUSDT/cvd_timeseries.png)

### 3. Z-score时间序列
![Z-score时间序列](../../figs_v270/BNBUSDT/cvd_z_timeseries.png)

### 4. 延迟箱线图
![延迟箱线图](../../figs_v270/BNBUSDT/cvd_latency_box.png)

### 5. 消息到达间隔分布
![Interarrival分布](../../figs_v270/BNBUSDT/cvd_interarrival_hist.png)

---

## 结论

**验收标准通过率**: 5/8 (62.5%)

**⚠️ 部分验收标准未通过**
