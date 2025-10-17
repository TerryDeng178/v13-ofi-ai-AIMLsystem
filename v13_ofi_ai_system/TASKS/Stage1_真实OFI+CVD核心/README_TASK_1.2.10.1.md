# Task 1.2.10.1 快速参考

## 🎯 任务目标
将CVD测试通过率从 **5/8（62.5%）** 提升到 **8/8（100%）**

---

## 🚀 快速启动指南

### 第一步：P0-A 最小验证（1小时）
```bash
# 1. 立即开始修改代码
cd v13_ofi_ai_system

# 2. 修改文件：
#    - src/binance_trade_stream.py（添加agg_trade_id字段）
#    - examples/analysis_cvd.py（修复排序逻辑）

# 3. 运行30分钟测试
cd examples
python run_realtime_cvd.py --symbol ETHUSDT --duration 1800 --output-dir ../data/cvd_p0a_test

# 4. 分析结果
python analysis_cvd.py --data ../data/cvd_p0a_test --out ../figs_cvd_p0a --report ../docs/reports/P0A_VERIFICATION_REPORT.md
```

**预期结果**: 通过率 6-7/8（75%-87.5%）

---

### 第二步：P0-B 完整修复（2小时）
```bash
# 1. 添加2s水位线重排
#    - 修改 src/binance_trade_stream.py（实现缓冲队列）

# 2. 运行60分钟测试
cd examples
python run_realtime_cvd.py --symbol ETHUSDT --duration 3600 --output-dir ../data/cvd_p0b_test

# 3. 分析结果
python analysis_cvd.py --data ../data/cvd_p0b_test --out ../figs_cvd_p0b --report ../docs/reports/P0B_FINAL_REPORT.md
```

**预期结果**: 通过率 7-8/8（87.5%-100%）

---

### 第三步：P1 Z-score优化（3小时）
```bash
# 1. 重构Z-score到增量域
#    - 修改 src/real_cvd_calculator.py（实现delta-Z）

# 2. 对比测试（旧版vs新版）
cd examples
CVD_Z_MODE=level python run_realtime_cvd.py --symbol ETHUSDT --duration 1800 --output-dir ../data/cvd_p1_old
CVD_Z_MODE=delta python run_realtime_cvd.py --symbol ETHUSDT --duration 1800 --output-dir ../data/cvd_p1_new

# 3. 生成对比报告
python compare_z_scores.py --old ../data/cvd_p1_old --new ../data/cvd_p1_new --report ../docs/reports/P1_Z_OPTIMIZATION_REPORT.md
```

**预期结果**: 通过率 8/8（100%），Z-score改进≥50%

---

## 📊 当前问题总结

| 指标 | 当前值 | 目标值 | 偏差 |
|------|--------|--------|------|
| `max_gap_ms` | 8387ms | ≤2000ms | **超标4倍** ❌ |
| `median(\|Z\|)` | 1.49 | ≈0 | **严重偏离** ❌ |
| `P(\|Z\|>2)` | 25.59% | 1%-8% | **超标3倍** ❌ |
| `P(\|Z\|>3)` | 6.32% | <1% | **超标6倍** ❌ |
| `continuity_mismatch` | 144/145 | 0 | **99.3%失败** ❌ |
| `duplicate_event_time_ms` | 3354 (23.1%) | 0 | **大量重复** ❌ |

---

## 🔑 核心修复要点

### P0修复（数据流）
1. ✅ 使用`agg_trade_id (a字段)`作为唯一键
2. ✅ 实现2s水位线乱序重排
3. ✅ 按`(event_time_ms, agg_trade_id)`双键排序
4. ✅ 改进CVD一致性检查（增量守恒）

### P1修复（Z-score）
1. ✅ 改为基于ΔCVD增量域标准化
2. ✅ 使用EWMA(|Δ|)稳健尺度
3. ✅ 实现winsorize截断（±8）
4. ✅ 实现stale冻结（>5s不产出z）

---

## 📁 关键文件位置

- **任务卡**: `Task_1.2.10.1_CVD问题修复（特别任务）.md`
- **修复指南**: `../../CVD_FIX_GUIDE_FOR_CURSOR.md`
- **问题分析**: `../../docs/reports/CVD_TEST_ISSUE_ANALYSIS.md`
- **测试报告**: `../../docs/reports/CVD_TEST_REPORT.md`

---

## 🎯 阶段验收标准

| 阶段 | 通过率目标 | 关键指标 |
|------|-----------|---------|
| **P0-A** | 6-7/8 | `agg_dup_rate` ≤ 1%, `continuity_mismatch` ≤ 5% |
| **P0-B** | 7-8/8 | `agg_dup_rate` = 0, `continuity_mismatch` = 0 |
| **P1** | 8/8 | `median(\|Z\|)` ≤ 1.0, `P(\|Z\|>2)` ≤ 10% |

---

## 🚨 注意事项

1. **每个阶段都要运行测试验证**，不要跳过
2. **P0-A通过后再做P0-B**，避免一次改动过多
3. **保留配置兼容性**，添加`z_mode`开关支持回滚
4. **所有修改前先备份**，确保可以随时回退

---

**立即开始**: 建议从P0-A开始，预计1小时可见效果！

