# Task 1.2.10: CVD计算测试

## 📋 任务信息

- **任务编号**: Task_1.2.10
- **任务名称**: CVD计算测试
- **所属阶段**: 阶段1.2 - 真实OFI+CVD计算
- **优先级**: 高
- **预计时间**: 2小时
- **实际时间**: ___（完成后填写）___
- **任务状态**: ⏳ 待开始

---

## 🎯 任务目标

运行CVD实时计算长期测试（≥2小时），验证稳定性、准确性和一致性。

---

## ⏱️ 测试时长分级

| 等级 | 时长 | 停止条件 | 说明 |
|------|------|----------|------|
| **Bronze（冒烟）** | ≥30分钟 | ≥1805s | 快速验证基本功能 |
| **Silver（稳定）** | ≥60分钟 | ≥3605s | 中期稳定性检验 |
| **Gold（验收目标）** | ≥120分钟 | ≥7205s | 长期稳定性验证（推荐默认） |

**本次目标**: Gold级别（≥120分钟，防止1.99小时边界值）

---

## 📝 任务清单

- [ ] 使用 `binance_trade_stream.py` 作为实时采集入口
- [ ] 运行CVD实时计算 ≥120分钟（Gold级别）
- [ ] 收集CVD数据到Parquet格式
- [ ] 使用 `analysis.py` 分析CVD分布和统计特性
- [ ] 生成4张标准图表（`figs_cvd/`目录）
- [ ] 验证CVD一致性（抽样1%）
- [ ] 生成完整测试报告 `CVD_TEST_REPORT.md`

---

## ✅ 验证标准（量化DoD）

### 1. 时长与连续性
- [ ] **运行时长**: ≥120分钟（Gold级别）
- [ ] **连续性**: `max_gap_ms` ≤ 2000ms

### 2. 数据质量
- [ ] **解析错误**: `parse_errors` == 0
- [ ] **队列丢弃率**: `queue_dropped_rate` ≤ 0.5%（若超出需在报告中解释）

### 3. 性能指标
- [ ] **处理延迟**: `p95_proc_ms` < 5ms

### 4. Z-score稳健性（与OFI对齐）
- [ ] **中位数偏离**: `median(|z_cvd|)` ≈ 0（给出实际数值）
- [ ] **四分位距**: `IQR(z_cvd)` ∈ [1.0, 2.0]（可按实测微调）
- [ ] **2-sigma比例**: `P(|Z|>2)` ∈ [1%, 8%]
- [ ] **3-sigma比例**: `P(|Z|>3)` < 1%
- [ ] **std_zero标记**: `std_zero` == 0（或接近0）

### 5. 一致性验证（抽样1%）
- [ ] **CVD连续性**: `cvd_t == cvd_{t-1} + Σ(±qty)`（容差 ≤ 1e-9）

### 6. 产出完整性
- [ ] **测试报告**: `CVD_TEST_REPORT.md`
- [ ] **图表**: `figs_cvd/*.png`（4张标准图）
- [ ] **分析结果**: `analysis_results.json`
- [ ] **监控指标**: `cvd_run_metrics.json`

---

## 📊 测试结果

### 运行配置
- **测试时长**: ___ 分钟（目标≥120分钟）
- **交易对**: ___
- **采集入口**: `binance_trade_stream.py`
- **数据格式**: Parquet

### 数据统计
- **总接收消息数**: ___
- **CVD值范围**: ___
- **Z-score统计**: 
  - P50 = ___
  - P95 = ___
  - P99 = ___
  - `median(|z_cvd|)` = ___
  - `IQR(z_cvd)` = ___

### Z-score分布验证
- **P(|Z|>2)**: ___% （目标: 1%~8%）
- **P(|Z|>3)**: ___% （目标: <1%）
- **std_zero标记**: ___

### 性能指标
- **处理延迟P95**: ___ ms （目标: <5ms）
- **队列丢弃率**: ___% （目标: ≤0.5%）
- **解析错误**: ___ （目标: 0）
- **重连次数**: ___

### 连续性验证（抽样1%）
- **max_gap_ms**: ___ ms （目标: ≤2000ms）
- **CVD一致性**: ___ （容差: ≤1e-9）

### 图表产出
- `figs_cvd/hist_z.png` - Z-score分布直方图
- `figs_cvd/cvd_timeseries.png` - CVD时序图
- `figs_cvd/z_timeseries.png` - Z-score时序图
- `figs_cvd/latency_box.png` - 延迟箱线图

---

## 🔗 相关文件

### Allowed files（统一项目结构）
- `v13_ofi_ai_system/src/binance_trade_stream.py` （实时采集入口）
- `v13_ofi_ai_system/src/real_cvd_calculator.py` （CVD计算器）
- `v13_ofi_ai_system/examples/analysis.py` （数据分析脚本）
- `v13_ofi_ai_system/docs/reports/CVD_TEST_REPORT.md` （测试报告输出）
- `v13_ofi_ai_system/figs_cvd/*.png` （图表输出）
- `v13_ofi_ai_system/data/cvd/%Y%m%d/%symbol%/run_%H%M%S_*.parquet` （数据输出）

### 数据产物与路径

**Parquet字段**（最少含）:
```python
- ts: 接收时间（时间戳）
- event_time_ms: 交易所事件时间（毫秒）
- cvd: CVD原始值
- z_cvd: Z-score标准化CVD
- ema_cvd: EMA平滑CVD
- meta.bad_points: 累计坏点数
- meta.warmup: warmup标记
- meta.std_zero: std_zero标记
- reconnect_count: 累计重连次数
- queue_dropped: 累计队列丢弃数
- total_messages: 总消息数
- parse_errors: 累计解析错误数
```

**监控指标JSON** (`cvd_run_metrics.json`):
```json
{
  "run_info": {
    "start_time": "...",
    "end_time": "...",
    "duration_seconds": ...,
    "total_records": ...
  },
  "performance": {
    "p95_proc_ms": ...,
    "queue_dropped_rate": ...
  },
  "z_statistics": {
    "median_abs_z": ...,
    "iqr_z": ...,
    "p_z_gt_2": ...,
    "p_z_gt_3": ...
  }
}
```

### 依赖
- `websockets>=10,<13` （WebSocket客户端）
- Python标准库: `asyncio`, `json`, `logging`, `pandas`

---

## ⚠️ 注意事项

### 字段口径统一
1. **时间字段**:
   - `ts`: 接收时间（本地时间戳）
   - `event_time_ms`: 交易所事件时间（毫秒）
   - 避免 `timestamp/ts` 混用

2. **端到端延迟定义**:
   ```python
   latency_ms = ts - event_time_ms
   ```
   若 `event_time_ms` 缺失则不统计

3. **分析链路与命令**:
   - 使用现有 `analysis.py` 进行数据分析
   - 生成4张标准图表（存放至 `figs_cvd/` 目录）:
     - `hist_z.png` - Z-score分布直方图
     - `cvd_timeseries.png` - CVD时序图（替换原 `ofi_timeseries.png`）
     - `z_timeseries.png` - Z-score时序图
     - `latency_box.png` - 延迟箱线图

4. **测试等级选择**:
   - 推荐Gold级别（≥120分钟）
   - 停止条件使用 `>=7205s` 防止边界值（1.99小时）

---

## 📋 DoD检查清单（量化）

### 时长与连续性
- [ ] **运行时长**: ≥7205秒（120分钟，Gold级别）
- [ ] **max_gap_ms**: ≤2000ms

### 数据质量
- [ ] **parse_errors**: == 0
- [ ] **queue_dropped_rate**: ≤0.5%

### 性能指标
- [ ] **p95_proc_ms**: <5ms

### Z-score稳健性
- [ ] **median(|z_cvd|)**: 接近0（给出数值）
- [ ] **IQR(z_cvd)**: ∈[1.0, 2.0]
- [ ] **P(|Z|>2)**: ∈[1%, 8%]
- [ ] **P(|Z|>3)**: <1%
- [ ] **std_zero**: ==0或接近0

### 一致性验证
- [ ] **CVD连续性**（抽样1%）: `cvd_t == cvd_{t-1} + Σ(±qty)` (容差≤1e-9)

### 产出完整性
- [ ] **CVD_TEST_REPORT.md**: 完整测试报告
- [ ] **figs_cvd/*.png**: 4张标准图表
- [ ] **analysis_results.json**: 分析结果
- [ ] **cvd_run_metrics.json**: 监控指标汇总

### 文档与提交
- [ ] **更新相关文档**
- [ ] **提交Git**

---

## 📝 执行记录

### 1. 启动命令与环境
```bash
# 启动命令
cd v13_ofi_ai_system/src
python binance_trade_stream.py --symbol ETHUSDT --duration 7205

# 环境变量
export LOG_LEVEL=INFO
export QUEUE_SIZE=1024
export HEARTBEAT_TIMEOUT=60

# 版本信息
Python版本: ___
websockets版本: ___
Git commit: ___
```

### 2. 运行记录
- **起始时间**: ___
- **结束时间**: ___
- **实际时长**: ___ 秒（目标≥7205秒）
- **总条数**: ___

### 3. 失败/异常与重连
- **重连次数**: ___
- **异常情况**: ___
- **处理方式**: ___

### 4. 分析命令与关键指标
```bash
# 数据分析命令
cd v13_ofi_ai_system/examples
python analysis.py --data-dir ../data/cvd/%Y%m%d/%symbol% --output-dir ../figs_cvd
```

**analysis_results.json摘要**:
```json
{
  "duration": ___ ,
  "total_records": ___,
  "z_statistics": {
    "median_abs_z": ___,
    "iqr_z": ___,
    "p_z_gt_2": ___,
    "p_z_gt_3": ___
  },
  "performance": {
    "p95_proc_ms": ___,
    "queue_dropped_rate": ___
  }
}
```

### 遇到的问题
___（记录遇到的问题）___

### 解决方案
___（记录解决方案）___

---

## 📈 质量评分

- **稳定性**: ___/10
- **数据质量**: ___/10
- **总体评分**: ___/10

---

## 🔄 任务状态更新

- **开始时间**: ___
- **完成时间**: ___
- **是否可以继续**: ⬜ 是 / ⬜ 否

---

**创建时间**: 2025-10-17  
**最后更新**: 2025-10-17

