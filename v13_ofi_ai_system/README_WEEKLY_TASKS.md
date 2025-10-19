# 本周三件事 - 背离检测参数调优

## 🎯 概述

本文档提供了"本周三件事"的完整可执行清单、脚手架和验收标准，用于背离检测模块的参数调优、单调性验证、指标对齐和配置热更新。

## 📋 任务清单

### 1️⃣ 参数调优（3×3×3粗网格扫描）
- **目标**: 找到最优参数组合，按分桶给出全局最佳与分场景最佳
- **脚本**: `scripts/tune_divergence.py`
- **参数**: swing_L ∈ {8,13,21}, z_hi ∈ {1.2,1.6,2.0}, z_mid ∈ {0.4,0.6,0.8}
- **分桶**: session×liquidity×source
- **验收**: 至少1个OFI桶和1个CVD桶达到acc≥55%且p<0.05

### 2️⃣ Score→收益单调性验证
- **目标**: 证明分数越高，未来收益越好，生成校准映射
- **脚本**: `scripts/score_monotonicity.py`
- **方法**: 10分位分箱、Spearman相关性、等势回归
- **验收**: Spearman ρ>0且p<0.05，分位曲线单调

### 3️⃣ 指标侧对齐（Prometheus/Grafana）
- **目标**: 离线评估指标与在线监控口径一致
- **脚本**: `scripts/metrics_alignment.py`
- **指标**: 事件计数、检测延迟、分数分布、前瞻收益等
- **验收**: 事件速率±10%对齐，P95延迟<3ms，计数闭合

### 4️⃣ 参数固化与热更新
- **目标**: 支持配置文件热更新和条件参数选择
- **脚本**: `scripts/config_hot_update.py`
- **配置**: `config/system.yaml`
- **验收**: 配置加载成功，热更新工作，校准配置可用

## 🚀 快速开始

### 一键执行（推荐）
```bash
# Linux/Mac
./scripts/quick_start.sh data/replay/btcusdt_2025-10-01_2025-10-19.parquet runs/weekly_tasks

# Windows
scripts\quick_start.bat data\replay\btcusdt_2025-10-01_2025-10-19.parquet runs\weekly_tasks
```

### 分步执行
```bash
# 1. 参数调优
python scripts/tune_divergence.py \
  --data data/replay/btcusdt_2025-10-01_2025-10-19.parquet \
  --out runs/tune_20251020

# 2. 单调性验证
python scripts/score_monotonicity.py \
  --data data/replay/btcusdt_2025-10-01_2025-10-19.parquet \
  --out runs/monotonicity

# 3. 指标对齐
python scripts/metrics_alignment.py \
  --out runs/metrics

# 4. 配置热更新
python scripts/config_hot_update.py --test
```

### 执行所有任务
```bash
python scripts/run_weekly_tasks.py \
  --data data/replay/btcusdt_2025-10-01_2025-10-19.parquet \
  --out runs/weekly_tasks_20251020
```

## 📊 验收标准

### 质量门槛（本周结束前必须达成）
1. ✅ **至少1个OFI_ONLY桶 & 1个CVD_ONLY桶满足acc≥55% & p<0.05**
2. ✅ **Spearman ρ(score, fwd_ret@H)>0 & p<0.05（H取10或20任一）**
3. ✅ **Prometheus/Grafana三类核心图卡能跑通且数值闭合（±10%）**
4. ✅ **config/system.yaml与calibration.json生效，支持热更新且指标可见**

### 详细验收标准
- 参见 `docs/weekly_tasks_acceptance.md`

## 📁 文件结构

```
v13_ofi_ai_system/
├── scripts/                          # 执行脚本
│   ├── tune_divergence.py           # 参数调优
│   ├── score_monotonicity.py        # 单调性验证
│   ├── metrics_alignment.py         # 指标对齐
│   ├── config_hot_update.py         # 配置热更新
│   ├── run_weekly_tasks.py          # 一键执行
│   ├── quick_start.sh               # Linux/Mac快速启动
│   └── quick_start.bat              # Windows快速启动
├── config/                           # 配置文件
│   ├── system.yaml                  # 主配置文件（包含背离检测配置）
│   └── calibration/                 # 校准配置
│       └── divergence_score_calibration.json
├── docs/                            # 文档
│   ├── divergence_tuning.md         # 调优指南
│   └── weekly_tasks_acceptance.md   # 验收标准
└── runs/                            # 结果目录
    ├── tune_*/                      # 参数调优结果
    ├── monotonicity*/               # 单调性验证结果
    ├── metrics*/                    # 指标对齐结果
    └── weekly_tasks_*/              # 综合结果
```

## 🔧 技术实现

### 参数调优
- 3×3×3网格搜索（27种组合）
- 分桶评估（session×liquidity×source）
- 统计显著性检验
- 全局最佳和分桶最佳参数

### 单调性验证
- 10分位分箱分析
- Spearman相关性检验
- 等势回归拟合
- Bootstrap置信区间

### 指标对齐
- 统一Prometheus指标定义
- Grafana仪表盘配置
- 离线在线数据对比
- 告警规则配置

### 配置热更新
- YAML配置文件
- 条件覆盖规则
- 文件监控热更新
- 上下文参数选择

## 📈 预期效果

### 参数调优
- 找到最优参数组合
- 不同场景差异化配置
- 准确率提升至55%以上

### 单调性验证
- 建立分数-收益映射关系
- 支持策略决策
- 提供期望收益预测

### 指标对齐
- 实时监控背离检测状态
- 快速定位问题
- 支持生产部署

### 配置热更新
- 支持参数动态调整
- 无需重启服务
- 条件化参数选择

## ⚠️ 注意事项

### 数据要求
- 需要足够的历史数据（建议≥1000条记录）
- 数据应包含price、z_ofi、z_cvd等字段
- 支持Parquet格式

### 环境要求
- Python 3.8+
- 必要包：pandas, numpy, scipy, scikit-learn, pyyaml, matplotlib
- 可选：prometheus_client, watchdog

### 性能要求
- 参数调优：<1小时
- 单调性验证：<30分钟
- 指标对齐：<15分钟
- 配置热更新：<5分钟

## 🚨 故障排除

### 常见问题
1. **没有检测到背离事件**: 检查数据质量，调整参数范围
2. **准确率始终为0%**: 检查前瞻收益计算，调整评分机制
3. **Spearman相关性为负**: 检查分数计算逻辑，调整评分权重
4. **Prometheus指标为空**: 检查指标导出器，确认数据源

### 调试方法
1. 查看脚本的`--help`参数
2. 检查生成的报告文件
3. 查看日志文件获取详细错误信息
4. 参考验收标准文档

## 📞 支持

- 调优指南: `docs/divergence_tuning.md`
- 验收标准: `docs/weekly_tasks_acceptance.md`
- 脚本帮助: `python scripts/<script_name>.py --help`

## 🎯 下一步计划

1. **本周**: 完成参数调优和单调性验证
2. **下周**: 完善指标对齐和配置热更新
3. **下月**: 集成到生产环境，支持实时交易

---

**创建时间**: 2025-01-20  
**版本**: v1.0  
**状态**: 可执行
