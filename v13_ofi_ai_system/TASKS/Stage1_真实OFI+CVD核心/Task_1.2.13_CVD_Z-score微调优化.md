# Task 1.2.13: CVD Z-score微调优化

## 📋 任务信息

- **任务编号**: Task_1.2.13
- **任务名称**: CVD Z-score微调优化 - 进一步压降P(|Z|>3)
- **所属阶段**: 阶段1.2 - 真实OFI+CVD计算（优化阶段）
- **优先级**: 中
- **预计时间**: 4-6小时
- **实际时间**: ___（完成后填写）___
- **任务状态**: ⏳ 待开始
- **前置任务**: 
  - ✅Task_1.2.10（CVD计算测试）- 初始测试，发现问题
  - ✅Task_1.2.10.1（CVD问题修复）- 建立Step 1.6生产基线（7/8通过）
  
## 📌 任务背景

本任务是CVD系统优化的第三阶段，基于已完成的修复任务：
- **Task 1.2.10** 发现Z-score分布问题（P(|Z|>3) = 6.32%严重超标）
- **Task 1.2.10.1** 通过Delta-Z算法和数据流修复，建立了Step 1.6生产基线（P(|Z|>3) = 4.65%）
- **本任务目标** 进一步优化参数，将P(|Z|>3)从4.65%压降到≤2%，达成完美8/8通过

---

## 🎯 任务目标

基于Step 1.6生产基线配置，通过精细参数微调进一步将P(|Z|>3)从4.65%压降到≤2%，同时保持P(|Z|>2)≤8%的达标状态。

---

## 📊 当前基线状态（Step 1.6）

### 已达成指标 ✅
- **P(|Z|>2)**: 5.73% ✅（目标≤8%）
- **median(|Z|)**: 0.0013 ✅（目标≤1.0）
- **数据完整性**: 100% ✅
- **数据一致性**: 100% ✅
- **系统稳定性**: 100% ✅

### 待优化指标 🎯
- **P(|Z|>3)**: 4.65% ⚠️（目标≤2%，差距2.65%）

### 当前配置（Step 1.6基线）
```bash
CVD_Z_MODE=delta
HALF_LIFE_TRADES=300
WINSOR_LIMIT=8.0
STALE_THRESHOLD_MS=5000
FREEZE_MIN=80
SCALE_MODE=hybrid
EWMA_FAST_HL=80
SCALE_FAST_WEIGHT=0.35
SCALE_SLOW_WEIGHT=0.65
MAD_WINDOW_TRADES=300
MAD_SCALE_FACTOR=1.4826
MAD_MULTIPLIER=1.45
```

---

## 📝 任务清单

### 阶段1: 参数敏感性分析（1-2小时）
- [ ] 分析当前参数空间，识别P(|Z|>3)的主要影响因素
- [ ] 测试MAD_MULTIPLIER微调（1.45 → 1.47, 1.48, 1.49）
- [ ] 测试SCALE_FAST_WEIGHT微调（0.35 → 0.32, 0.38）
- [ ] 测试HALF_LIFE_TRADES微调（300 → 295, 305）
- [ ] 生成参数敏感性报告

### 阶段2: 保守微调测试（2-3小时）
- [ ] 实施Step 1.7-A: MAD_MULTIPLIER=1.47
- [ ] 实施Step 1.7-B: MAD_MULTIPLIER=1.48 + SCALE_FAST_WEIGHT=0.32
- [ ] 实施Step 1.7-C: 软冻结扩展（4.0s < E间隔 ≤ 5.0s → 1笔冻结）
- [ ] 每个配置运行20分钟测试
- [ ] 对比分析结果

### 阶段3: 最优配置验证（1-2小时）
- [ ] 选择最佳配置组合
- [ ] 运行60分钟验证测试
- [ ] 生成最终优化报告
- [ ] 更新生产配置

---

## 🎯 微调策略

### 策略1: 地板微调（保守）
```bash
# Step 1.7-A: 轻微抬高地板
MAD_MULTIPLIER=1.47  # 从1.45提升到1.47

# 预期效果: P(|Z|>3) 降低0.5-1.0%
```

### 策略2: 权重平衡（中等）
```bash
# Step 1.7-B: 调整快慢权重
MAD_MULTIPLIER=1.48
SCALE_FAST_WEIGHT=0.32  # 从0.35降低到0.32
SCALE_SLOW_WEIGHT=0.68  # 从0.65提升到0.68

# 预期效果: 更稳定的尺度计算
```

### 策略3: 软冻结扩展（激进）
```bash
# Step 1.7-C: 扩展软冻结覆盖
# 当前: 3.5s < E间隔 ≤ 5.0s → 1笔冻结
# 优化: 4.0s < E间隔 ≤ 5.0s → 1笔冻结

# 预期效果: 减少"静默后首笔"的异常Z-score
```

---

## ✅ 验收标准

### 硬指标（必须达成）
- [ ] **P(|Z|>3)**: ≤2% ✅
- [ ] **P(|Z|>2)**: ≤8%（不能退化）✅
- [ ] **median(|Z|)**: ≤1.0（不能退化）✅
- [ ] **数据完整性**: 100%（不能退化）✅

### 软指标（期望达成）
- [ ] **P(|Z|>3)**: 相比基线改善≥30%
- [ ] **P95(|Z|)**: 相比基线改善≥20%
- [ ] **系统稳定性**: 保持100%

---

## 📊 测试计划

### 快速验证测试（20分钟×3）
```bash
# Step 1.7-A测试
cd v13_ofi_ai_system/examples
$env:MAD_MULTIPLIER="1.47"; python run_realtime_cvd.py --symbol ETHUSDT --duration 1200 --output-dir ../data/cvd_step_1_7a

# Step 1.7-B测试  
$env:MAD_MULTIPLIER="1.48"; $env:SCALE_FAST_WEIGHT="0.32"; python run_realtime_cvd.py --symbol ETHUSDT --duration 1200 --output-dir ../data/cvd_step_1_7b

# Step 1.7-C测试（需先修改软冻结逻辑）
python run_realtime_cvd.py --symbol ETHUSDT --duration 1200 --output-dir ../data/cvd_step_1_7c
```

### 最终验证测试（60分钟）
```bash
# 最优配置验证
python run_realtime_cvd.py --symbol ETHUSDT --duration 3600 --output-dir ../data/cvd_p1_2_final
python analysis_cvd.py --data ../data/cvd_p1_2_final --out ../figs_cvd_p1_2 --report ../docs/reports/P1_2_OPTIMIZATION_REPORT.md
```

---

## 📈 预期成果

### 目标指标
- **P(|Z|>3)**: 4.65% → ≤2%（改善≥57%）
- **P(|Z|>2)**: 5.73% → ≤6%（保持达标）
- **P95(|Z|)**: 2.71 → ≤2.5（改善≥8%）

### 配置优化
- 确定最优MAD_MULTIPLIER值
- 确定最优SCALE_FAST_WEIGHT值
- 优化软冻结逻辑

### 文档交付
- P1.2微调优化报告
- 参数敏感性分析报告
- 更新生产配置文档

---

## ⚠️ 风险控制

### 回滚策略
- 每个微调步骤后立即验证，如有退化立即回滚
- 保留Step 1.6基线配置作为应急回滚点
- 设置P(|Z|>2)退化阈值：>8%立即停止

### 监控指标
- 实时监控P(|Z|>2)和P(|Z|>3)变化
- 监控median(|Z|)和P95(|Z|)趋势
- 确保数据完整性指标不退化

---

## 🔗 相关文件

### 配置文件
- `v13_ofi_ai_system/config/step_1_6_microtune.env`（基线配置）
- `v13_ofi_ai_system/config/step_1_7a_microtune.env`（微调A）
- `v13_ofi_ai_system/config/step_1_7b_microtune.env`（微调B）
- `v13_ofi_ai_system/config/step_1_7c_microtune.env`（微调C）

### 分析脚本
- `v13_ofi_ai_system/examples/analysis_cvd.py`
- `v13_ofi_ai_system/examples/compare_configs.py`（需新建）

### 报告文档
- `v13_ofi_ai_system/docs/reports/P1_2_OPTIMIZATION_REPORT.md`
- `v13_ofi_ai_system/docs/reports/PARAMETER_SENSITIVITY_ANALYSIS.md`

---

## 📝 执行记录

### 开始时间
- **计划开始**: 2025-10-18 15:00
- **实际开始**: ___

### 阶段完成情况
- **阶段1（参数分析）**: ⏳ 待开始
- **阶段2（微调测试）**: ⏳ 待开始  
- **阶段3（验证优化）**: ⏳ 待开始

### 测试结果
- **Step 1.7-A**: ___/8 通过
- **Step 1.7-B**: ___/8 通过
- **Step 1.7-C**: ___/8 通过
- **最终验证**: ___/8 通过

### 遇到的问题
___（记录遇到的问题）___

### 解决方案
___（记录解决方案）___

---

## 🎯 下一步任务

完成P1.2微调优化后，建议的后续任务：

1. **Task 1.2.14**: 集成CVD模块参数热更新
2. **Task 1.2.15**: 集成OFI/Risk/Performance模块
3. **Task 1.2.16**: 真实环境24小时测试
4. **Task 1.3.1**: 收集历史OFI+CVD数据

---

**创建时间**: 2025-10-18 14:00  
**最后更新**: 2025-10-18 14:00  
**任务负责人**: AI Assistant + CURSOR + USER  
**审核人**: USER
