# CVD系统文件组织指南

## 📋 文档概述

**创建时间**: 2025-10-19  
**Git提交**: 6a6b958  
**状态**: 已完成配置优化和文档完善

---

## 🎯 当前文件组织结构

### ✅ 核心文件（生产使用）

#### 配置文件 (`config/`)
```
config/
├── profiles/                    # 生产配置（★ 主要使用）
│   ├── analysis.env            # 分析模式配置
│   └── realtime.env            # 实时模式配置
│
├── step_1_6_analysis.env       # Step 1.6基线配置
├── step_1_6_clean_gold.env     # Step 1.6干净金测配置
├── step_1_6_fixed_gold.env     # Step 1.6修复版配置
└── binance_config.yaml         # Binance配置
```

**使用建议**:
- ✅ **生产环境**: 使用 `profiles/analysis.env` 或 `profiles/realtime.env`
- ✅ **测试验证**: 使用 `step_1_6_*.env` 系列
- ✅ **参数基线**: 参考 `step_1_6_analysis.env`

---

#### 源代码 (`src/`)
```
src/
├── real_cvd_calculator.py      # CVD核心算法 ★
└── __init__.py
```

---

#### 示例脚本 (`examples/`)
```
examples/
├── run_realtime_cvd.py         # 实时采集主程序 ★
├── analysis_cvd.py              # 数据分析脚本 ★
└── test_delta_z.py             # Delta-Z测试
```

---

#### 文档 (`docs/`)
```
docs/
├── CVD_SYSTEM_FILES_GUIDE.md           # 文件关联完全说明 ★★★
├── CONFIG_PARAMETERS_GUIDE.md          # 配置参数完全指南 ★★★
├── CVDSYSTEM_ARCHITECTURE.md           # 系统架构
├── CVD_SYSTEM_README.md                # 系统README
├── CVD_QUICK_REFERENCE.md              # 快速参考
│
├── monitoring/
│   └── dashboard_config.md             # 监控配置
│
├── reports/                            # 测试报告
│   ├── STEP_1_6_CLEAN_GOLD_TEST_REPORT.md      # Step 1.6金测报告 ★
│   ├── ENGINEERING_FIXES_REPORT.md             # 工程修复报告 ★
│   ├── CODE_AUDIT_REPORT.md                    # 代码审查报告 ★
│   ├── ANALYSIS_MODE_VERIFICATION_FINAL.md     # 分析模式验证 ★
│   ├── STEP_1_FINAL_SUMMARY.md                 # Step 1总结
│   └── ... (其他历史报告)
│
└── roadmap/
    └── P1.2_optimization_plan.md       # P1.2优化计划
```

**核心文档**（★★★ 强烈推荐阅读）:
1. `CVD_SYSTEM_FILES_GUIDE.md` - 文件关联完全说明
2. `CONFIG_PARAMETERS_GUIDE.md` - 配置参数完全指南

**关键报告**（★ 推荐参考）:
1. `STEP_1_6_CLEAN_GOLD_TEST_REPORT.md` - 最新测试结果
2. `ENGINEERING_FIXES_REPORT.md` - 工程修复详情
3. `CODE_AUDIT_REPORT.md` - 代码审查结果

---

### 📦 数据文件（已归档到Git）

#### 测试数据 (`data/`)
```
data/
├── cvd_clean_gold_test/                # 5分钟干净金测
├── cvd_clean_gold_test_35min/          # 35分钟干净金测
├── cvd_fixed_gold_test/                # 修复版金测
├── cvd_quick_check/                    # 快速检查
├── cvd_analysis_verify/                # 分析模式验证v1
├── cvd_analysis_verify_v2/             # 分析模式验证v2
├── cvd_analysis_verify_v3/             # 分析模式验证v3
└── cvd_analysis_verify_v4/             # 分析模式验证v4
```

**说明**: 这些测试数据已经保存到Git，用于验证CVD算法和配置。

---

#### 图表文件 (`figs_cvd_*`)
```
figs_cvd_analysis_verify/               # 分析模式验证图表v1
figs_cvd_analysis_verify_v2/            # 分析模式验证图表v2
figs_cvd_analysis_verify_v3/            # 分析模式验证图表v3
```

**说明**: Z-score分布、时间序列、延迟等可视化图表。

---

### 🔧 辅助脚本 (`scripts/`)
```
scripts/
└── monitor_clean_gold_test.py          # 金测监控脚本
```

---

## 📁 建议的清理/归档策略

### 方案1: 保持当前结构（推荐）✅

**理由**:
- 所有文件已经提交到Git（commit 6a6b958）
- 文件结构清晰，易于查找
- 测试数据和报告有历史价值

**优点**:
- ✅ 无需额外操作
- ✅ 保留完整历史记录
- ✅ 便于追溯问题

---

### 方案2: 清理冗余文档（可选）

如果确实需要简化目录，可以归档以下文件：

#### 可归档的重复文档
```bash
# 移动到 _archive/docs_redundant/
docs/CVD_QUICK_REFERENCE.md             # 与 CVD_SYSTEM_FILES_GUIDE.md 部分重复
src/CVD_QUICK_REFERENCE.md              # 与 docs/ 重复
src/CVD_SYSTEM_README.md                # 与 docs/ 重复
```

#### 可归档的中间验证报告
```bash
# 移动到 _archive/reports_history/
docs/reports/ANALYSIS_MODE_VERIFICATION.md       # v1（已有Final版）
docs/reports/ANALYSIS_MODE_VERIFICATION_V2.md    # v2（已有Final版）
docs/reports/ANALYSIS_MODE_VERIFICATION_V3.md    # v3（已有Final版）
docs/reports/P1.2_MICROTUNE_FAILURE_SUMMARY.md  # 失败记录（已完成修复）
```

#### 可归档的早期测试数据
```bash
# 移动到 _archive/test_data_history/
data/cvd_analysis_verify/               # v1测试（已有v4）
data/cvd_analysis_verify_v2/            # v2测试（已有v4）
data/cvd_analysis_verify_v3/            # v3测试（已有v4）
data/cvd_quick_check/                   # 快速检查（已完成正式测试）

figs_cvd_analysis_verify/               # v1图表
figs_cvd_analysis_verify_v2/            # v2图表
figs_cvd_analysis_verify_v3/            # v3图表
```

---

## 🗂️ 核心文件快速索引

### 我要...

#### 🎯 开始使用CVD系统
**阅读**: `docs/CVD_SYSTEM_FILES_GUIDE.md`  
**使用**: `config/profiles/analysis.env` + `examples/run_realtime_cvd.py`

---

#### 🔧 调整配置参数
**阅读**: `docs/CONFIG_PARAMETERS_GUIDE.md`  
**修改**: `config/profiles/analysis.env` 或 `config/profiles/realtime.env`

---

#### 📊 分析测试数据
**运行**: `examples/analysis_cvd.py`  
**参考**: `docs/reports/STEP_1_6_CLEAN_GOLD_TEST_REPORT.md`

---

#### 🐛 排查问题
**阅读**:
1. `docs/CVD_SYSTEM_FILES_GUIDE.md` - 常见问题排查部分
2. `docs/reports/CODE_AUDIT_REPORT.md` - 代码审查结果
3. `docs/reports/ENGINEERING_FIXES_REPORT.md` - 已知问题和修复

---

#### 📈 了解性能指标
**阅读**:
- `docs/reports/STEP_1_6_CLEAN_GOLD_TEST_REPORT.md` - 最新测试结果
- `docs/CONFIG_PARAMETERS_GUIDE.md` - 参数影响矩阵

---

#### 🚀 生产部署
**配置**:
1. 复制 `config/profiles/analysis.env` 为 `config/profiles/production.env`
2. 根据实际需求调整参数
3. 使用 `examples/run_realtime_cvd.py` 启动

**监控**: `docs/monitoring/dashboard_config.md`

---

## 📝 文件命名规范

### 配置文件
- `*.env` - 环境变量配置
- `profiles/` - 生产配置
- `step_*` - 测试/微调配置

### 文档文件
- `*_GUIDE.md` - 指南类文档
- `*_REPORT.md` - 报告类文档
- `*_SUMMARY.md` - 总结类文档

### 数据文件
- `cvd_*.parquet` - CVD数据
- `figs_*` - 图表目录
- `data/cvd_*` - 测试数据目录

---

## 🔍 快速查找技巧

### 查找配置文件
```bash
# 生产配置
ls config/profiles/

# 测试配置
ls config/step_*
```

### 查找文档
```bash
# 核心指南
ls docs/*_GUIDE.md

# 测试报告
ls docs/reports/*_REPORT.md
```

### 查找测试数据
```bash
# 最新测试
ls -lt data/cvd_* | head -5

# 特定测试
ls data/cvd_clean_gold_test*
```

---

## ⚠️ 重要提醒

1. **不要删除以下文件**（生产依赖）:
   - `config/profiles/` 目录
   - `src/real_cvd_calculator.py`
   - `examples/run_realtime_cvd.py`
   - `examples/analysis_cvd.py`
   - `docs/CVD_SYSTEM_FILES_GUIDE.md`
   - `docs/CONFIG_PARAMETERS_GUIDE.md`

2. **归档前务必确认**:
   - 文件已提交到Git
   - 无其他文件引用
   - 有完整的备份

3. **保留Git历史**:
   - 所有归档操作都应通过Git进行
   - 保留提交历史，便于恢复

---

## 📊 文件统计

### 当前状态（Commit 6a6b958）

| 类别 | 数量 | 说明 |
|------|------|------|
| 配置文件 | 6个 | profiles/ + step_1_6_* 系列 |
| 源代码 | 2个 | real_cvd_calculator.py + __init__.py |
| 示例脚本 | 3个 | run_realtime_cvd.py + analysis_cvd.py + test_delta_z.py |
| 核心文档 | 5个 | *_GUIDE.md + *_ARCHITECTURE.md |
| 测试报告 | 20+个 | reports/ 目录 |
| 测试数据 | 8个目录 | data/cvd_* 系列 |
| 图表目录 | 3个 | figs_cvd_* 系列 |

---

## 🎯 下一步行动

### 如果需要清理

1. **创建清理分支**:
   ```bash
   git checkout -b cleanup/file-organization
   ```

2. **移动文件到归档**:
   ```bash
   # 参照"方案2"中的建议
   mkdir -p _archive/{docs_redundant,reports_history,test_data_history}
   # 然后逐步移动
   ```

3. **提交归档**:
   ```bash
   git add -A
   git commit -m "chore: 归档冗余文档和早期测试数据"
   ```

4. **合并回主分支**:
   ```bash
   git checkout master
   git merge cleanup/file-organization
   ```

### 如果保持现状

✅ **不需要任何操作**  
当前文件结构已经很清晰，所有关键文档都已完善。

---

*文档版本: v1.0*  
*最后更新: 2025-10-19*  
*Git Commit: 6a6b958*

