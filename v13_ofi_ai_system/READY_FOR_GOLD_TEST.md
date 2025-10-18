# ✅ Step 1.6 金测准备就绪报告

**日期**: 2025-10-19  
**状态**: 🟢 所有修复已完成，可以开始金测

---

## 📦 已完成的工作

### 1. 核心代码修复（5项）✅

#### `analysis_cvd.py` - 4个口径问题已修复
1. **连续性判定顺序** ✅
   - 问题：`continuity_pass` 在 `gap_p99_ms` 计算之前就被判定，导致误判
   - 修复：将判定移到第107行，确保在 `gap_p99_ms` 计算之后
   
2. **时长口径统一** ✅
   - 问题：代码用≥30分钟，报告写≥120分钟
   - 修复：第553行统一为"≥30分钟，分析模式基线"
   
3. **守恒容差使用相对容差** ✅
   - 问题：报告写固定1e-6，代码用相对容差
   - 修复：第576行报告也使用相对容差说明
   
4. **CLI参数别名** ✅
   - 问题：文档用 `--input/--output-dir`，代码用 `--data/--out`
   - 修复：第23-24行添加别名支持，两种都可用

#### `run_realtime_cvd.py` - 数据管道问题已修复 ✅
- 队列策略：`DROP_OLD=false`，`maxsize=50000`（分析模式阻塞不丢）
- Watermark flush：每200ms周期flush
- 时间戳修复：`force_flush_timeout` 使用正确的时间戳
- 默认参数：所有Step 1.6参数已设为默认值
- 指标分离：`queue_dropped` 和 `late_event_dropped` 已分离

#### `real_cvd_calculator.py` - 诊断优化 ✅
- 日志频率：从每60笔改为每1000笔
- 权重归一化：确保 `w_fast + w_slow = 1.0`
- 配置打印：启动时打印完整的Step 1.6配置

### 2. 配置文件完整 ✅

所有配置文件已包含关键参数：
- `DROP_OLD=false` （分析模式阻塞队列）
- `POST_STALE_FREEZE=2` （>5s空窗后首2笔不产Z）
- `HARD_FREEZE_MS=5000` / `SOFT_FREEZE_MS=4000` （分段冻结阈值）
- `PRINT_EVERY=1000` （诊断日志频率）
- `WATERMARK_FLUSH_INTERVAL_MS=200` （分析档）/ `=100` （实时档）

### 3. 文档完善 ✅

新增/更新文档：
- `docs/CVD_SYSTEM_FILES_GUIDE.md` - 系统文件全面指南
- `docs/CONFIG_PARAMETERS_GUIDE.md` - 配置参数详细对比
- `docs/FILE_ORGANIZATION_GUIDE.md` - 文件组织结构
- `docs/CLEANUP_SUMMARY.md` - 清理归档总结
- `docs/reports/HEALTH_CHECK_FIXES.md` - 代码审计修复总结

### 4. 项目清理 ✅

- 历史测试数据归档到 `archive/test_data/`
- 重复文档删除
- 目录结构优化
- Git版本已提交（commit: `完成代码审计修复和项目文件清理`）

---

## 🎯 下一步行动（按顺序执行）

### 第1步：快速验证（5分钟）⚡

**目的**: 确认所有修复生效、配置正确加载

```powershell
# 进入工作目录
cd C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\v13_ofi_ai_system\examples

# 运行5分钟快速测试
python run_realtime_cvd.py --symbol ETHUSDT --duration 300 --output-dir ../../data/cvd_quick_verify

# 检查启动日志是否显示：
# - Z_MODE=delta
# - SCALE_MODE=hybrid  
# - HALF_LIFE_TRADES=300
# - WINSOR_LIMIT=8
# - FREEZE_MIN=80
# - STALE_THRESHOLD_MS=5000
# - SCALE_FAST_WEIGHT=0.35, SCALE_SLOW_WEIGHT=0.65 (归一化后)
# - MAD_MULTIPLIER=1.45
# - WATERMARK_MS=2000

# 检查运行中日志：
# - 每1000笔打印一次分母自检
# - queue_dropped_rate = 0%
```

**成功标准**:
- ✅ 配置正确打印
- ✅ queue_dropped_rate = 0%
- ✅ 无异常错误

---

### 第2步：完整金测（40分钟）🏆

**目的**: 完成Step 1.6基线的完整验证

```powershell
# 终端1：启动测试
cd C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\v13_ofi_ai_system\examples

$timestamp = Get-Date -Format 'yyyyMMdd_HHmm'
echo "开始金测: $timestamp"
echo "预计完成时间: $(Get-Date).AddMinutes(45)"

python run_realtime_cvd.py `
  --symbol ETHUSDT `
  --duration 2400 `
  --output-dir "../../data/cvd_final_gold_$timestamp"

# 终端2：监控进度（可选）
cd C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\v13_ofi_ai_system\scripts
python monitor_final_gold.py
```

**测试期间监控**:
- 系统资源：CPU、内存使用率
- 关键指标：queue_dropped_rate、采集速率
- 日志输出：分母自检、冻结统计

---

### 第3步：分析结果（10分钟）📊

```powershell
cd C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\v13_ofi_ai_system\examples

# 使用你测试时的实际时间戳
$timestamp = "20251019_0230"  # 替换为实际值

python analysis_cvd.py `
  --data "../../data/cvd_final_gold_$timestamp/*.parquet" `
  --out "../../docs/reports/cvd_final_gold_$timestamp" `
  --report "../../docs/reports/cvd_final_gold_$timestamp/REPORT.md"

# 查看报告
cat "../../docs/reports/cvd_final_gold_$timestamp/REPORT.md"
```

**验收标准 (8/8)**:

| # | 指标 | 目标 | 说明 |
|---|------|------|------|
| 1 | parse_errors | = 0 | 数据完整性 |
| 2 | queue_dropped_rate | = 0% | 无数据丢失 |
| 3 | p99_interarrival | ≤ 5000ms | 连续性 |
| 4 | gaps_over_10s | = 0 | 无长空窗 |
| 5 | 逐笔守恒 | = 0 错误 | CVD计算正确 |
| 6 | 首尾守恒 | < 相对容差 | 端到端一致 |
| 7 | median\|Z\| | ≤ 1.0 | Z-score质量 |
| 8a | P(\|Z\|>2) | ≤ 8% | 2σ尾部 |
| 8b | P(\|Z\|>3) | ≤ 2% | 3σ尾部（优化目标） |

**判定逻辑**:
- **如果 8/8 全部通过** → 跳到第5步（打标签）
- **如果 7/8 通过，仅 P(|Z|>3) 略高（2-3%）** → 进入第4步（S7微调）
- **如果其他指标未通过** → 需要回溯检查问题

---

### 第4步：S7微调（可选，仅在需要时）🔧

**触发条件**: 8/8中除了 `P(|Z|>3)` 外全部通过，但 `P(|Z|>3)` 在 2-3% 区间

#### S7-A方案：提高MAD地板

```bash
# 修改一个参数
MAD_MULTIPLIER: 1.45 → 1.47

# 测试两个交易对各20分钟
python run_realtime_cvd.py --symbol ETHUSDT --duration 1200 --output-dir ../../data/cvd_s7a_eth
python run_realtime_cvd.py --symbol BTCUSDT --duration 1200 --output-dir ../../data/cvd_s7a_btc

# 分析对比
python analysis_cvd.py --data "../../data/cvd_s7a_eth/*.parquet" --out ../../docs/reports/cvd_s7a_eth
python analysis_cvd.py --data "../../data/cvd_s7a_btc/*.parquet" --out ../../docs/reports/cvd_s7a_btc
```

#### S7-B方案：增加快速分量权重

```bash
# 修改两个参数（确保和为1）
SCALE_FAST_WEIGHT: 0.35 → 0.38
SCALE_SLOW_WEIGHT: 0.65 → 0.62

# 测试
python run_realtime_cvd.py --symbol ETHUSDT --duration 1200 --output-dir ../../data/cvd_s7b_eth
python run_realtime_cvd.py --symbol BTCUSDT --duration 1200 --output-dir ../../data/cvd_s7b_btc

# 分析对比
python analysis_cvd.py --data "../../data/cvd_s7b_eth/*.parquet" --out ../../docs/reports/cvd_s7b_eth
python analysis_cvd.py --data "../../data/cvd_s7b_btc/*.parquet" --out ../../docs/reports/cvd_s7b_btc
```

**择优标准**:
1. P(|Z|>3) 降低最多
2. P(|Z|>2) 保持 ≤ 8%
3. median|Z| 保持 ≤ 1.0

**确认步骤**:
择优配置后运行60分钟验证：
```bash
python run_realtime_cvd.py --symbol ETHUSDT --duration 3600 --output-dir ../../data/cvd_s7_final
```

---

### 第5步：打标签并固化配置🏷️

**前提**: 8/8验收全部通过

```bash
cd C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\v13_ofi_ai_system

# 1. 确认配置文件
# 如果用的是Step 1.6基线
git add config/step_1_6_analysis.env
git add config/profiles/analysis.env
git add config/profiles/realtime.env

# 2. 归档测试报告
git add docs/reports/cvd_final_gold_*/

# 3. 提交
git commit -m "Step 1.6 基线金测通过 8/8 验收标准

- 测试时长: 40分钟
- 交易对: ETHUSDT
- queue_dropped_rate: 0%
- P(|Z|>2): X.XX%
- P(|Z|>3): X.XX%
- median|Z|: X.XXX

所有数据完整性、连续性、守恒性、Z-score质量指标全部达标。"

# 4. 打标签
git tag -a v13_cvd_step1.6_baseline -m "Step 1.6 Delta-Z + Hybrid Scale 基线配置

核心参数:
- Z_MODE=delta
- SCALE_MODE=hybrid
- HALF_LIFE_TRADES=300
- WINSOR_LIMIT=8
- FREEZE_MIN=80
- STALE_THRESHOLD_MS=5000
- SCALE_FAST_WEIGHT=0.35
- SCALE_SLOW_WEIGHT=0.65
- MAD_MULTIPLIER=1.45
- WATERMARK_MS=2000 (分析档)

通过8/8验收标准，可用于灰度上线。"

# 5. 查看标签
git tag -l "v13_cvd*"
git show v13_cvd_step1.6_baseline
```

---

## 🚀 灰度上线准备

### 实时模式参数调整

从 `config/profiles/analysis.env` 调整到 `config/profiles/realtime.env`:

| 参数 | 分析模式 | 实时模式 | 说明 |
|------|----------|----------|------|
| `WATERMARK_MS` | 2000 | 500-1000 | 降低延迟 |
| `WATERMARK_FLUSH_INTERVAL_MS` | 200 | 100 | 更频繁flush |
| `DROP_OLD` | false | false (初期) | 初期保持阻塞，稳定后可改true |
| 其他参数 | 相同 | 相同 | 核心算法参数不变 |

### 触发护栏策略

**双条件确认**:
```python
# 条件1：Z-score异常
if abs(z_cvd) > 3:
    # 条件2：OFI/价量确认
    if ofi_signal_confirmed or price_volume_confirmed:
        trigger_alert()
```

**保留冻结逻辑**:
- 事件时间间隔 > 5s → 首2笔不产Z
- 事件时间间隔 4-5s → 首1笔不产Z
- 交易笔数 < FREEZE_MIN(80) → 不产Z

### 监控看板

**关键指标**:
1. **Z-score质量**:
   - median|Z|, P(|Z|>2), P(|Z|>3)
   - Z分布直方图
   
2. **分母健康**:
   - scale的p5/p50/p95
   - ewma_fast/slow/mix趋势
   - sigma_floor趋势

3. **数据质量**:
   - queue_dropped_rate
   - late_event_dropped
   - p99_interarrival
   - gaps_over_10s

4. **系统性能**:
   - P95延迟
   - CPU/内存使用
   - 处理吞吐量

---

## 📝 测试记录模板

```
========================================
Step 1.6 金测结果记录
========================================

测试信息：
- 日期：2025-10-19
- 时段：02:30-03:10
- 交易对：ETHUSDT
- 时长：40分钟

配置（启动日志确认）：
- Z_MODE=delta
- SCALE_MODE=hybrid
- HALF_LIFE_TRADES=300
- WINSOR_LIMIT=8
- FREEZE_MIN=80
- STALE_THRESHOLD_MS=5000
- SCALE_FAST_WEIGHT=0.35/0.65 (归一化)
- MAD_MULTIPLIER=1.45
- WATERMARK_MS=2000

验收结果 (8/8)：
✅ parse_errors: 0
✅ queue_dropped_rate: 0.0000%
✅ p99_interarrival: 4523.45 ms (≤5000ms)
✅ gaps_over_10s: 0
✅ 逐笔守恒: 0 错误
✅ 首尾守恒: 1.23e-10 < 2.45e-09 (相对容差)
✅ median|Z|: 0.8523 (≤1.0)
✅ P(|Z|>2): 5.67% (≤8%)
✅ P(|Z|>3): 1.23% (≤2%)

分母健康度：
- scale_p5: 1234.56
- scale_p50: 2345.67
- scale_p95: 3456.78

冻结统计：
- z_freeze_count: 23
- post-stale-3trades |z|分布：[略]

环境：
- CPU: 平均35%
- 内存: 平均2.5GB
- 网络: 稳定
```

---

## ✅ 最终检查清单

在开始金测前，请确认：

- [ ] 所有代码修复已完成并已提交Git
- [ ] 配置文件中包含所有必需参数
- [ ] 文档已更新到最新版本
- [ ] 测试环境网络稳定
- [ ] 系统资源充足（CPU < 80%, 内存 < 80%）
- [ ] 已创建测试输出目录权限
- [ ] 时间充裕（预留60分钟）

---

## 🎯 预期结果

基于前期测试和所有修复，预期：

**高置信度** (≥90%)：
- ✅ queue_dropped_rate = 0%
- ✅ 数据完整性、连续性、守恒性全部通过
- ✅ median|Z| ≤ 1.0
- ✅ P(|Z|>2) ≤ 8%

**中等置信度** (60-70%)：
- ⚠️ P(|Z|>3) ≤ 2% （可能需要S7微调）

如果 P(|Z|>3) 在 2-3% 区间，属于"接近达标"，可通过S7-A或S7-B小步调参即可压到2%以内。

---

## 📞 问题升级

如遇到以下情况，需要暂停并诊断：

1. **queue_dropped_rate > 0%** → 检查网络、系统资源、队列配置
2. **配置未正确加载** → 检查环境变量、启动日志
3. **中途崩溃** → 检查错误日志、内存溢出
4. **其他验收指标未通过** → 逐项诊断根因

---

**祝测试顺利！🎉**

所有准备工作已完成，你现在可以信心满满地开始Step 1.6的最终金测了！

