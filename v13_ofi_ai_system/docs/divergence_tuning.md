# 背离检测参数调优指南

## 📋 概述

本文档详细说明了背离检测模块的参数调优流程，包括3×3×3粗网格扫描、单调性验证、指标对齐和配置热更新等核心功能。

## 🎯 本周三件事

### 1. 参数调优（3×3×3粗网格扫描）

#### 目标
用最小代价跑完 `swing_L × z_hi × z_mid` 粗搜，按"分桶"给出全局最佳与分场景最佳。

#### 推荐取值
- **swing_L**: {8, 13, 21} - 短中长枢轴，与常见周期对齐
- **z_hi**: {1.2, 1.6, 2.0} - 高阈值，触发更少但更稳
- **z_mid**: {0.4, 0.6, 0.8} - 中阈值，用于去噪/配对过滤

#### 数据分桶
- **交易时段**: active（成交/波动上分位）vs quiet
- **来源**: OFI_ONLY、CVD_ONLY、FUSION
- **时间**: day（本地09:00–17:00）/ night

#### 执行命令
```bash
# 离线网格调参
python scripts/tune_divergence.py \
  --data data/replay/btcusdt_2025-10-01_2025-10-19.parquet \
  --out runs/tune_20251020 \
  --horizons 10,20 \
  --buckets "session=day,night;liquidity=active,quiet;source=OFI,CVD,FUSION"
```

#### 产出物
- `best_global.yaml` - 全局最佳参数
- `best_by_bucket.yaml` - 各桶最佳参数
- `summary.csv` - 详细结果
- `tuning_report.json` - 调优报告

#### 验收标准
- 每个桶都有结果
- 至少1个OFI_ONLY桶和1个CVD_ONLY桶达到acc@10或acc@20 ≥ 55%且p<0.05
- summary.csv、best_*.yaml可复现、可追溯

### 2. Score→收益单调性验证

#### 目标
证明分数越高，未来收益越好（方向正确、斜率为正），并生成可部署的分数→期望收益/命中率映射。

#### 方法
- **分箱**: 按score做10分位（或等频20分位）
- **相关性**: Spearman ρ(score, fwd_ret@H)，要求ρ>0且p<0.05
- **单调拟合**: 等势回归（Isotonic Regression）

#### 执行命令
```bash
# 单调性验证
python scripts/score_monotonicity.py \
  --data data/replay/btcusdt_2025-10-01_2025-10-19.parquet \
  --out runs/monotonicity \
  --bins 10
```

#### 产出物
- `score_monotonicity_10.png` - 10期分位曲线图
- `score_monotonicity_20.png` - 20期分位曲线图
- `divergence_score_calibration.json` - 校准映射文件
- `monotonicity_report.json` - 验证报告

#### 验收标准
- 至少一个前瞻窗口（10或20 bars）满足Spearman ρ > 0且p < 0.05
- 分位曲线总体上行（经等势回归后单调）

### 3. 指标侧对齐（Prometheus/Grafana）

#### 目标
离线评估指标与在线监控口径一致，一眼能看懂"触发-延迟-效果"。

#### 指标规范
- **事件计数**: `divergence_events_total{source,side,kind}`
- **检测延迟**: `divergence_detection_latency_seconds{source}`
- **分数分布**: `divergence_score_bucket{source}`
- **配对间隔**: `divergence_pairing_gap_bars{source}`
- **前瞻收益**: `divergence_forward_return{horizon,source}`
- **生效配置**: `divergence_active_config_info{swing_L,z_hi,z_mid,version}`

#### 执行命令
```bash
# 指标对齐
python scripts/metrics_alignment.py \
  --out runs/metrics
```

#### 产出物
- `prometheus_divergence.yml` - Prometheus配置
- `alerting_rules/divergence_alerts.yaml` - 告警规则
- `dashboards/divergence_overview.json` - Grafana仪表盘
- `divergence_metrics_exporter.py` - 指标导出器
- `metrics_alignment_check.py` - 对齐检查脚本

#### 验收标准
- 事件速率与离线数据量级一致（±10%）
- 在线P95延迟 < 3ms
- 事件计数闭合：bull+bear ≈ all；regular+hidden ≈ all

### 4. 参数固化与热更新

#### 目标
回答"参数存在哪里，AI怎么用？"

#### 配置文件
```yaml
# config/system.yaml
divergence_detection:
  version: "v1"
  default:
    swing_L: 13
    z_hi: 1.6
    z_mid: 0.6
  overrides:
    - when: {source: "OFI", liquidity: "quiet"}
      set: {z_hi: 1.2, z_mid: 0.4}
  calibration:
    file: "config/calibration/divergence_score_calibration.json"
```

#### 执行命令
```bash
# 配置热更新测试
python scripts/config_hot_update.py --test

# 启动配置监控
python scripts/config_hot_update.py --watch
```

#### 产出物
- `config/system.yaml` - 主配置文件（包含背离检测配置）
- `config/calibration/divergence_score_calibration.json` - 校准映射
- 热更新机制和监控

#### 验收标准
- 配置文件加载成功
- 热更新机制工作正常
- 校准配置可用

## 🚀 一键执行

### 执行所有任务
```bash
# 运行本周三件事
python scripts/run_weekly_tasks.py \
  --data data/replay/btcusdt_2025-10-01_2025-10-19.parquet \
  --out runs/weekly_tasks_20251020
```

### 执行单个任务
```bash
# 只运行参数调优
python scripts/run_weekly_tasks.py \
  --data data/replay/btcusdt_2025-10-01_2025-10-19.parquet \
  --out runs/tune_only \
  --task tune_params
```

## 📊 质量门槛

### 本周结束前必须达成
1. **至少1个OFI_ONLY桶 & 1个CVD_ONLY桶满足acc≥55% & p<0.05**
2. **Spearman ρ(score, fwd_ret@H)>0 & p<0.05（H取10或20任一）**
3. **Prometheus/Grafana三类核心图卡能跑通且数值闭合（±10%）**
4. **configs/divergence.yaml与calibration.json生效，支持热更新且指标可见**

## ⚠️ 风险与预案

### 数据稀疏
- **问题**: 静态回放不足
- **预案**: 追加更长区间或更多交易对；分桶降维（10→5分位）保持稳健性

### 单调性断层
- **问题**: 分数与收益关系不单调
- **预案**: 用等势回归"拉直"并在低分位设置最小阈值（低分位直接不触发）

### 指标爆卡
- **问题**: 标签集合过大
- **预案**: 限制标签集合（source/kind/side/horizon），杜绝symbol级高基数；图卡聚合在上层

## 📁 文件结构

```
v13_ofi_ai_system/
├── scripts/
│   ├── tune_divergence.py          # 参数调优脚本
│   ├── score_monotonicity.py       # 单调性验证脚本
│   ├── metrics_alignment.py        # 指标对齐脚本
│   ├── config_hot_update.py        # 配置热更新脚本
│   └── run_weekly_tasks.py         # 一键执行脚本
├── config/
│   ├── system.yaml                 # 主配置文件（包含背离检测配置）
│   └── calibration/
│       └── divergence_score_calibration.json
├── runs/
│   ├── tune_20251020/              # 参数调优结果
│   ├── monotonicity/               # 单调性验证结果
│   ├── metrics/                    # 指标对齐结果
│   └── weekly_tasks_20251020/      # 综合结果
└── docs/
    └── divergence_tuning.md        # 本文档
```

## 🔧 技术实现要点

### 参数调优
- 使用3×3×3网格搜索，共27种参数组合
- 按数据源、时段、流动性分桶评估
- 统计显著性检验（p<0.05）
- 生成全局最佳和分桶最佳参数

### 单调性验证
- 10分位分箱分析
- Spearman相关性检验
- 等势回归拟合
- Bootstrap置信区间

### 指标对齐
- 统一Prometheus指标定义
- Grafana仪表盘配置
- 离线在线数据对比验证
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

## 🎯 下一步计划

1. **本周**: 完成参数调优和单调性验证
2. **下周**: 完善指标对齐和配置热更新
3. **下月**: 集成到生产环境，支持实时交易

## 📞 支持

如有问题，请参考：
- 各脚本的`--help`参数
- 生成的报告文件
- 本文档的故障排除部分
