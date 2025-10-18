# Step 1.6 最终金测执行计划

## 📋 测试概况

**目标**: 完成Step 1.6基线的35-40分钟干净金测，验证所有工程修复和口径修正是否生效

**时间**: 2025-10-19
**状态**: 准备中

## ✅ 前置检查清单

### 1. 代码修复已完成
- [x] `analysis_cvd.py` 连续性判定顺序修正
- [x] `analysis_cvd.py` 时长口径统一（≥30分钟）
- [x] `analysis_cvd.py` 守恒容差使用相对容差
- [x] `analysis_cvd.py` CLI参数别名支持
- [x] `run_realtime_cvd.py` 队列策略修复（DROP_OLD=false, maxsize=50000）
- [x] `run_realtime_cvd.py` watermark flush周期修复
- [x] `run_realtime_cvd.py` 默认参数对齐Step 1.6
- [x] `real_cvd_calculator.py` 日志频率优化（每1000笔）
- [x] `real_cvd_calculator.py` 权重归一化验证

### 2. 配置文件就绪
- [x] `config/profiles/analysis.env` - 完整配置
- [x] `config/profiles/realtime.env` - 完整配置  
- [x] `config/step_1_6_analysis.env` - 基线配置
- [x] `config/step_1_6_fixed_gold.env` - 修复版配置

### 3. 文档更新
- [x] `docs/CVD_SYSTEM_FILES_GUIDE.md` - 系统文件指南
- [x] `docs/CONFIG_PARAMETERS_GUIDE.md` - 参数对比指南
- [x] `docs/FILE_ORGANIZATION_GUIDE.md` - 文件组织指南
- [x] `docs/CLEANUP_SUMMARY.md` - 清理总结
- [x] `docs/reports/HEALTH_CHECK_FIXES.md` - 修复总结

## 🎯 测试执行步骤

### Step 1: 快速验证（5分钟）
```powershell
cd v13_ofi_ai_system/examples
python run_realtime_cvd.py --symbol ETHUSDT --duration 300 --output-dir ../../data/cvd_quick_verify_$(Get-Date -Format 'yyyyMMdd_HHmm')
```

**验证点**:
- [ ] 启动日志显示正确的Step 1.6配置
- [ ] Z_MODE=delta, SCALE_MODE=hybrid
- [ ] FREEZE_MIN=80, STALE_THRESHOLD_MS=5000
- [ ] weights=0.35/0.65, MAD_MULTIPLIER=1.45
- [ ] 分母自检日志每1000笔输出一次
- [ ] queue_dropped_rate = 0%

### Step 2: 完整金测（40分钟）
```powershell
cd v13_ofi_ai_system/examples
$timestamp = Get-Date -Format 'yyyyMMdd_HHmm'
python run_realtime_cvd.py --symbol ETHUSDT --duration 2400 --output-dir ../../data/cvd_final_gold_$timestamp

# 测试完成后立即分析
python analysis_cvd.py `
  --data "../../data/cvd_final_gold_$timestamp/*.parquet" `
  --out "../../docs/reports/cvd_final_gold_$timestamp" `
  --report "../../docs/reports/cvd_final_gold_$timestamp/REPORT.md"
```

### Step 3: 分析结果（8/8验收）

#### 3.1 数据完整性
- [ ] parse_errors = 0
- [ ] queue_dropped_rate = 0%
- [ ] 无backwards / duplicates

#### 3.2 连续性
- [ ] p99_interarrival ≤ 5000ms
- [ ] gaps_over_10s = 0
- [ ] 时长 ≥ 30分钟

#### 3.3 守恒性
- [ ] 逐笔守恒错误 = 0
- [ ] 首尾守恒误差 < 相对容差

#### 3.4 Z-score质量
- [ ] median|Z| ≤ 1.0
- [ ] P(|Z|>2) ≤ 8%
- [ ] P(|Z|>3) ≤ 2% （优化目标）

#### 3.5 工程指标
- [ ] 配置正确加载并打印
- [ ] 分母自检日志正常（ewma_fast/slow/mix, sigma_floor, scale）
- [ ] 事件时间冻结生效（>5s首2笔、4-5s首1笔）
- [ ] z_freeze_count > 0（如有空窗）

## 🔄 后续步骤

### 如果 8/8 全部通过
1. **打Git标签**: `v13_cvd_step1.6_baseline` 或 `v13_cvd_step1.6_fixed`
2. **固化配置**: 确认 `config/profiles/*.env` 参数
3. **准备灰度**: 
   - 实时档使用 `WATERMARK_MS=500-1000`
   - 初期保持 `DROP_OLD=false`
   - 触发条件：双条件确认（|Z|>3 + OFI/价量）
   - 保留空窗冻结逻辑

### 如果 P(|Z|>3) 仍偏高（但其他7/8通过）
按顺序尝试S7微调：

#### S7-A: 提高MAD地板
```bash
# 修改配置
MAD_MULTIPLIER: 1.45 → 1.47

# 测试
python run_realtime_cvd.py --symbol ETHUSDT --duration 1200
python run_realtime_cvd.py --symbol BTCUSDT --duration 1200

# 对比 P(|Z|>2) 和 P(|Z|>3)
```

#### S7-B: 增加快速分量权重
```bash
# 修改配置
SCALE_FAST_WEIGHT: 0.35 → 0.38
SCALE_SLOW_WEIGHT: 0.65 → 0.62  # 确保和为1

# 测试
python run_realtime_cvd.py --symbol ETHUSDT --duration 1200
python run_realtime_cvd.py --symbol BTCUSDT --duration 1200
```

**择优标准**:
- P(|Z|>3) 降低最多
- P(|Z|>2) 保持在8%以内
- median|Z| 保持≤1.0

**确认步骤**: 择优配置后跑60分钟验证，然后打标签

## 📊 监控要点

### 实时监控
```powershell
# 终端1: 运行测试
cd v13_ofi_ai_system/examples
python run_realtime_cvd.py ...

# 终端2: 监控进度
cd v13_ofi_ai_system/scripts
python monitor_final_gold.py
```

### 关键指标
1. **系统资源**: CPU < 50%, 内存 < 70%
2. **队列丢弃**: queue_dropped_rate = 0%
3. **数据速率**: ~500-2000 trades/min (ETHUSDT)
4. **Z-score**: 实时查看 median|Z|、P(|Z|>2/3)

## 📝 记录要求

测试完成后需要记录：
1. **配置快照**: 启动日志中的 "Effective config"
2. **8/8验收结果**: 每项指标的具体数值和通过状态
3. **分母健康度**: scale 的 p5/p50/p95
4. **冻结统计**: z_freeze_count、post-stale-3trades |z| 分布
5. **运行环境**: 日期、时段、网络状况、系统资源

## 🎯 成功标准

**最低要求** (7/8):
- 数据完整性、连续性、守恒性、工程指标全部通过
- P(|Z|>2) ≤ 8%, median|Z| ≤ 1.0

**优化目标** (8/8):
- 在最低要求基础上
- P(|Z|>3) ≤ 2%

**灰度上线**:
- 8/8 全部达标
- 配置固化并打标签
- 实时档参数调整（WATERMARK_MS=500-1000）
- 双条件触发护栏
- 监控看板就绪

---

## ⏰ 时间线

| 阶段 | 预计时间 | 状态 |
|------|----------|------|
| 快速验证（5分钟） | ~10分钟 | ⏳ 待开始 |
| 完整金测（40分钟） | ~45分钟 | ⏳ 待开始 |
| 结果分析 | ~10分钟 | ⏳ 待开始 |
| S7微调（如需要） | ~40-60分钟 | 🔄 条件执行 |
| 打标签和固化 | ~10分钟 | ⏳ 待开始 |

**总计**: 1-2小时（取决于是否需要S7微调）

