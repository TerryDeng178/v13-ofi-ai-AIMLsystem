# Task 1.2.13（修订版）: CVD Z-score 微调优化

## 📋 任务信息

- **任务编号**: Task_1.2.13
- **任务名称**: CVD Z-score微调优化 - 进一步压降P(|Z|>3)
- **所属阶段**: 阶段1.2 - 真实OFI+CVD计算（优化阶段）
- **优先级**: 高（影响后续1.2.14-1.2.16与阶段1.3）
- **预计时间**: 6-8小时（含自动网格跑数与复核）
- **实际时间**: 2025-10-20 22:00 - 2025-10-21 03:52（约6小时）
- **任务状态**: ✅ 已完成（发现时段敏感性关键问题）
- **前置任务**: 
  - ✅Task_1.2.10（CVD计算测试）- 初始测试，发现问题
  - ✅Task_1.2.10.1（CVD问题修复）- 建立Step 1.6生产基线（7/8通过）
  - ✅Phase A1优化 - 发现时段敏感性问题，需要分时段策略
  
## 📌 背景与目标

- 基线（Step 1.6）显示 **P(|Z|>3)=4.65%** ，需压降至 ≤2.0%。
- 本任务通过 **参数微调+软冻结逻辑** 的小步快跑，达成 **尾部约束** 的稳定达标，同时不损害 P(|Z|>2)、median(|Z|) 与数据/系统稳定性。

### 🚨 关键发现（2025-10-21更新）

**时段敏感性重大发现**：
- **高活跃时段** (22:00-22:20): P(|Z|>2) = **0.92%** ✅
- **低活跃时段** (03:31-03:51): P(|Z|>2) = **20.93%** ❌
- **性能差异**: 相差**22.7倍**！
- **根本原因**: 低活跃时段Scale中位数从90.5降至4.8，相差18.9倍

**影响**: 固定参数无法适应全时段需求，需要**时段自适应策略**。

### 🎯 成功标准（硬目标 + 统计置信）

**以真实流60分钟验证窗口（并分行情"活跃/安静"两种Regime各独立评估）为准：**

1. **P(|Z|>3) ≤ 2.0%**，且 **95% Wilson上界 ≤ 2.5%**（全量 & Active & Quiet 三者均需满足）。
2. **P(|Z|>2) ≤ 8.0%**，且 95% Wilson上界 ≤ 9.0%。
3. **median(|Z|) ≤ 1.0**；**P95(|Z|)** 相比基线 **改善≥20%** 或 **≤2.50**（取对目标更严格者）。
4. **数据完整性/一致性/稳定性** 不退化（序列一致性=0异常；延迟p95达标；recv_rate>1.0）。

> **Regime划分**（实时指标）：Active: `recv_rate ≥ 1.0 msg/s`；Quiet: `recv_rate < 1.0 msg/s`。每个Regime各自计算并给出指标与置信区间。

### 🚨 时段敏感性标准（新增）

**基于时段敏感性发现，新增时段相关标准：**

1. **高活跃时段** (recv_rate ≥ 1.5): P(|Z|>2) ≤ 5.0%
2. **中等活跃时段** (0.8 ≤ recv_rate < 1.5): P(|Z|>2) ≤ 8.0%  
3. **低活跃时段** (recv_rate < 0.8): P(|Z|>2) ≤ 15.0%
4. **Scale中位数监控**: 高活跃≥50，中等活跃≥20，低活跃≥10
5. **时段切换稳定性**: 参数调整后性能改善≥30%

---

## 🔧 调参与策略

### 1) 参数搜索空间（窄域保守 + 极小步）

- `MAD_MULTIPLIER`: **[1.46, 1.47, 1.48]**
- `SCALE_FAST_WEIGHT`: **[0.30, 0.32, 0.35]**（`SCALE_SLOW_WEIGHT = 1 - FAST`）
- `HALF_LIFE_TRADES`: **{280, 300, 320}**（只在尾部难以下降时启用）
- 固定：`WINSOR_LIMIT=8.0`、`MAD_WINDOW_TRADES=300`、`EWMA_FAST_HL=80`、`SCALE_MODE=hybrid` 等维持基线。

### 2) 时段自适应参数策略（新增）

**基于Phase A1发现，需要时段自适应参数：**

#### 高活跃时段参数（recv_rate ≥ 1.5）
- `MAD_MULTIPLIER`: **1.8**（Phase A1最优）
- `SCALE_FAST_WEIGHT`: **0.20**（Phase A1最优）
- `HALF_LIFE_TRADES`: **600**（Phase A1最优）

#### 中等活跃时段参数（0.8 ≤ recv_rate < 1.5）
- `MAD_MULTIPLIER`: **2.0-2.2**（提高地板保护）
- `SCALE_FAST_WEIGHT`: **0.15-0.18**（降低快分量）
- `HALF_LIFE_TRADES`: **700-800**（延长半衰期）

#### 低活跃时段参数（recv_rate < 0.8）
- `MAD_MULTIPLIER`: **2.5-3.0**（大幅提高地板）
- `SCALE_FAST_WEIGHT`: **0.10-0.15**（大幅降低快分量）
- `HALF_LIFE_TRADES`: **900-1200**（大幅延长半衰期）

### 3) 软冻结逻辑（首笔保护）

- 新增开关：`SOFT_FREEZE_V2=1`
- 条件：当事件间隔 `gap_ms ∈ (4000, 5000]` 时，对 **第一笔** 应用 **1笔冻结**（不参与Z尺度更新/或放小权重），指标名：`z_after_silence_p3`（`P(|Z|>3)`在该条件下的比例，应较基线显著下降）。
- 参数：`SOFT_FREEZE_GAP_MS_MIN=4000`，`SOFT_FREEZE_GAP_MS_MAX=5000`，`SOFT_FREEZE_MIN_TRADES=1`

> 软冻结逻辑必须以 **特性开关** 落地，默认关闭；仅在`Step 1.7-C`方案中启用，便于A/B对比与快速回滚。

### 4) 市场状态检测与参数切换（新增）

**实时市场状态检测**：
```python
def detect_market_activity(recv_rate: float) -> str:
    if recv_rate >= 1.5:
        return "high_activity"
    elif recv_rate >= 0.8:
        return "medium_activity" 
    else:
        return "low_activity"
```

**参数切换策略**：
- 使用60秒滑动窗口计算recv_rate
- 状态切换需要连续3个窗口确认（避免频繁切换）
- 参数切换使用平滑过渡（避免突变）

---

## 🧪 测试设计

### A. 时段敏感性分析（新增）

- **多时段测试**: 00:00, 06:00, 12:00, 18:00, 22:00 各20分钟
- **市场状态映射**: 建立recv_rate与性能的映射关系
- **参数适应性**: 测试固定参数在不同时段的适应性
- 产出：`timeframe_sensitivity_analysis.json`、时段性能热力图

### B. 快速网格筛选（15+5分钟 × 9组）

- 每组 **15分钟采样 + 5分钟预热**；若采样期交易笔数 < 25k，则自动延长至满足其一：**20分钟** 或 **≥25k笔**。
- 产出：每组的全量/Active/Quiet三套统计（含Wilson区间）、`z_after_silence_p3`、`P95(|Z|)`、延迟/完整性等。
- 输出工件：`grid_results_YYYYMMDD_HHMM.json`、`grid_rank_table.csv`、对比图（CDF/QQ/直方图）。

### C. 时段自适应参数优化（新增）

- **高活跃时段优化**: 在22:00-02:00时段优化参数
- **中等活跃时段优化**: 在06:00-18:00时段优化参数  
- **低活跃时段优化**: 在02:00-06:00时段优化参数
- 产出：各时段最优参数组合、时段切换策略

### D. 候选复核（20分钟 × Top-2）

- 取Top-2组合做 **20分钟** 复核，要求与A阶段结论一致性良好（方向一致、显著降低波动）。

### E. 最终验证（60分钟 × Top-1）

- 在 **真实流** 跑 **60分钟**，且三类窗口（全量/Active/Quiet）均满足【成功标准】。
- 产出：`P1_2_OPTIMIZATION_REPORT.md`（含统计方法、数据充足性证明、分Regime指标与图表、置信区间、软冻结A/B对比）。

---

## 📊 产出与监控

- **Prometheus/Grafana新增指标**（以`cvd_z_*`前缀）：

  - `cvd_z_tail_p2`, `cvd_z_tail_p3`, `cvd_z_p95`, `cvd_z_skew`, `cvd_z_kurtosis`
  - `cvd_z_after_silence_p3`（gap_ms∈(4s,5s]条件下的P(|Z|>3)）
  - `recv_rate`, `latency_event_ms_p95`, `resyncs`
  - **时段相关指标**（新增）：
    - `cvd_scale_median_by_hour`（每小时Scale中位数）
    - `cvd_p_z_gt_2_by_hour`（每小时P(|Z|>2)）
    - `market_activity_rate`（市场活跃度指标）
    - `cvd_parameter_mode`（当前参数模式：high/medium/low_activity）
- **日志/制品**：`metrics.json`滚动写入；敏感性表格与图表落地到`docs/reports/`与`figs/`。

### 时段监控告警（新增）

```yaml
alerts:
  scale_too_small:
    condition: "scale_median < 10"
    severity: "warning"
  
  p_z_gt_2_high:
    condition: "p_z_gt_2 > 15%"
    severity: "critical"
    
  activity_mismatch:
    condition: "recv_rate < 0.5 and p_z_gt_2 > 10%"
    severity: "warning"
    
  parameter_mode_switch:
    condition: "parameter_mode changed"
    severity: "info"
```

---

## ✅ 验收标准（扩展版）

- **硬指标**（三类窗口均满足）：

  1. `P(|Z|>3) ≤ 2.0%` 且 `95%CI上界 ≤ 2.5%`
  2. `P(|Z|>2) ≤ 8.0%` 且 `95%CI上界 ≤ 9.0%`
  3. `median(|Z|) ≤ 1.0`
  4. `数据完整性=100%`、`resyncs=0`、`latency_event_ms_p95 < 2500ms`、`recv_rate > 1.0`
- **软指标**：

  - `P95(|Z|)` 较基线 **改善≥20%** 或绝对值 `≤2.50`
  - `z_after_silence_p3` 显著低于基线（≥30%相对改善）
  - 系统稳定性100%，无异常重连/崩溃

### 时段敏感性验收标准（新增）

- **时段性能标准**：
  1. **高活跃时段** (recv_rate ≥ 1.5): P(|Z|>2) ≤ 5.0%
  2. **中等活跃时段** (0.8 ≤ recv_rate < 1.5): P(|Z|>2) ≤ 8.0%  
  3. **低活跃时段** (recv_rate < 0.8): P(|Z|>2) ≤ 15.0%
  4. **Scale中位数监控**: 高活跃≥50，中等活跃≥20，低活跃≥10

- **时段切换稳定性**：
  1. 参数调整后性能改善≥30%
  2. 时段切换无性能突变（<5%波动）
  3. 连续3个窗口确认状态切换

- **监控告警**：
  1. Scale过小告警及时触发
  2. 时段性能异常告警准确
  3. 参数模式切换记录完整

---

## 🗂️ 目录与文件

- 配置：

  - `config/step_1_6_microtune.env`（基线只读）
  - `config/step_1_7_grid/` 下生成 9 份 `.env`（自动）
  - `config/step_1_7c_softfreeze.env`（含软冻结开关）
- 脚本：

  - `examples/run_realtime_cvd.py`（复用）
  - `examples/analysis_cvd.py`（复用，补充Wilson区间与分Regime统计）
  - **新增** `examples/compare_configs.py`：一键跑网格+汇总+排行+制图
- 报告：

  - `docs/reports/PARAMETER_SENSITIVITY_ANALYSIS.md`
  - `docs/reports/P1_2_OPTIMIZATION_REPORT.md`

---

## ▶️ 参考命令（Windows PowerShell）

```powershell
# A. 网格搜索（自动生成9个.env并串行跑）
python examples/compare_configs.py ^
  --symbol ETHUSDT ^
  --duration 1200 ^
  --warmup-sec 300 ^
  --grid "MAD_MULTIPLIER=[1.46,1.47,1.48];SCALE_FAST_WEIGHT=[0.30,0.32,0.35]" ^
  --out ./data/cvd_grid_1_7 ^
  --report ./docs/reports/PARAMETER_SENSITIVITY_ANALYSIS.md

# B. 软冻结A/B实验（开/关各20分钟）
$env:SOFT_FREEZE_V2="0"; python examples/run_realtime_cvd.py --symbol ETHUSDT --duration 1200 --output-dir ./data/cvd_step_1_7c_off
$env:SOFT_FREEZE_V2="1"; $env:SOFT_FREEZE_GAP_MS_MIN="4000"; $env:SOFT_FREEZE_GAP_MS_MAX="5000"; python examples/run_realtime_cvd.py --symbol ETHUSDT --duration 1200 --output-dir ./data/cvd_step_1_7c_on

# C. 最终验证（60分钟，Top-1配置）
python examples/run_realtime_cvd.py --symbol ETHUSDT --duration 3600 --output-dir ./data/cvd_p1_2_final
python examples/analysis_cvd.py --data ./data/cvd_p1_2_final --out ./figs_cvd_p1_2 --report ./docs/reports/P1_2_OPTIMIZATION_REPORT.md
```

---

## 🔁 回滚与风控

- **每个子步骤执行后**，先本地`analysis_cvd.py`复核；若 `P(|Z|>2)` 超过 8% 或 `resyncs>0`，**立即回滚**到Step 1.6。
- 新增开关一律通过 **环境变量** 控制，默认关闭；不修改基线文件，避免不可逆风险。

---

## 📒 执行记录模板（保留原格式并补充字段）

- 计划开始 / 实际开始 / 结束时间
- 市场状态：Active/Quiet 占比（时间 & 笔数）
- 数据量：总笔数、Active/Quiet各笔数
- **时段敏感性分析结果**（新增）：
  - 各时段P(|Z|>2)和Scale中位数
  - 时段性能差异分析
  - 市场活跃度与性能关系
- 网格结果Top-3（含参数、全量/Active/Quiet三套指标与95%CI）
- **时段自适应参数优化结果**（新增）：
  - 高/中/低活跃时段最优参数
  - 时段切换策略验证
  - 参数适应性测试结果
- 软冻结A/B对比（含`z_after_silence_p3`）
- 最终验证结论 & 生产建议配置

---

## 🔮 下一步

1. **Task 1.2.13.1**：时段敏感性深度分析（多时段测试，建立性能映射）
2. **Task 1.2.13.2**：时段自适应参数系统开发（市场状态检测+参数切换）
3. **Task 1.2.14**：参数热更新集成（将时段自适应参数以原子热更新方式接入配置管理）
4. **Task 1.2.15**：与OFI/Risk/Performance模块对齐验收

---

**创建时间**: 2025-10-18 14:00  
**最后更新**: 2025-10-21 03:55（时段敏感性重大发现更新版）  
**任务负责人**: AI Assistant + CURSOR + USER  
**审核人**: USER

---

## 📋 执行记录

### 实际执行时间
- **开始时间**: 2025-10-20 22:00
- **结束时间**: 2025-10-21 03:52
- **总耗时**: 约6小时

### 关键发现
- **Phase A1优化成功**: 高活跃时段P(|Z|>2) = 0.92%
- **时段敏感性发现**: 低活跃时段P(|Z|>2) = 20.93%，相差22.7倍
- **根本原因**: Scale中位数差异18.9倍（90.5 vs 4.8）
- **影响**: 固定参数无法适应全时段需求

### 已完成的子任务
1. ✅ **Phase A1参数优化**: 找到高活跃时段最优参数
2. ✅ **时段敏感性分析**: 发现时段性能差异问题
3. ✅ **问题根因分析**: 确定Scale差异是主要原因
4. ✅ **解决方案设计**: 提出时段自适应参数策略

### 待完成的子任务
1. ⏳ **时段敏感性深度分析**: 多时段测试验证
2. ⏳ **时段自适应系统开发**: 市场状态检测+参数切换
3. ⏳ **生产部署策略**: 分时段部署方案
