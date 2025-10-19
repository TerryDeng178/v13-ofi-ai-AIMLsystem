# 本周三件事验收标准

## 📋 总体要求

本周结束前必须达成以下4项质量门槛：

1. ✅ **至少1个OFI_ONLY桶 & 1个CVD_ONLY桶满足acc≥55% & p<0.05**
2. ✅ **Spearman ρ(score, fwd_ret@H)>0 & p<0.05（H取10或20任一）**
3. ✅ **Prometheus/Grafana三类核心图卡能跑通且数值闭合（±10%）**
4. ✅ **configs/divergence.yaml与calibration.json生效，支持热更新且指标可见**

---

## 1️⃣ 参数调优验收标准

### 基本要求
- [ ] 完成3×3×3粗网格扫描（27种参数组合）
- [ ] 按分桶统计结果（session×liquidity×source）
- [ ] 生成全局最佳和分桶最佳参数文件

### 数据要求
- [ ] 至少2个数据源类型（OFI_ONLY、CVD_ONLY、FUSION）
- [ ] 至少2个时段类型（day、night）
- [ ] 至少2个流动性类型（active、quiet）

### 准确率要求
- [ ] **至少1个OFI_ONLY桶达到acc@10 ≥ 55%且p < 0.05**
- [ ] **至少1个CVD_ONLY桶达到acc@10 ≥ 55%且p < 0.05**
- [ ] 或acc@20 ≥ 55%且p < 0.05（二选一）

### 产出物要求
- [ ] `best_global.yaml` - 全局最佳参数
- [ ] `best_by_bucket.yaml` - 分桶最佳参数
- [ ] `summary.csv` - 详细结果表格
- [ ] `tuning_report.json` - 调优报告

### 可追溯性要求
- [ ] 所有结果文件包含数据范围和时间戳
- [ ] 参数组合可复现
- [ ] 结果文件包含哈希值

---

## 2️⃣ 单调性验证验收标准

### 基本要求
- [ ] 完成10分位分箱分析
- [ ] 计算Spearman相关性
- [ ] 进行等势回归拟合

### 统计显著性要求
- [ ] **至少一个前瞻窗口（10或20 bars）满足Spearman ρ > 0且p < 0.05**
- [ ] 分位曲线总体上行趋势
- [ ] 等势回归后保持单调性

### 产出物要求
- [ ] `score_monotonicity_10.png` - 10期分位曲线图
- [ ] `score_monotonicity_20.png` - 20期分位曲线图
- [ ] `divergence_score_calibration.json` - 校准映射文件
- [ ] `monotonicity_report.json` - 验证报告

### 图表要求
- [ ] 分位曲线图包含置信区间
- [ ] 散点图包含等势回归线
- [ ] 图表包含统计信息标注

---

## 3️⃣ 指标对齐验收标准

### Prometheus指标要求
- [ ] 事件计数指标：`divergence_events_total{source,side,kind}`
- [ ] 检测延迟指标：`divergence_detection_latency_seconds{source}`
- [ ] 分数分布指标：`divergence_score_bucket{source}`
- [ ] 配对间隔指标：`divergence_pairing_gap_bars{source}`
- [ ] 前瞻收益指标：`divergence_forward_return{horizon,source}`
- [ ] 配置信息指标：`divergence_active_config_info{swing_L,z_hi,z_mid,version}`

### Grafana面板要求
- [ ] 事件速率面板：`rate(divergence_events_total[5m])`
- [ ] 检测延迟P95面板：`histogram_quantile(0.95, ...)`
- [ ] 分数分布热力图面板
- [ ] 前瞻收益时间序列面板
- [ ] 配置状态表格面板

### 对齐验证要求
- [ ] **事件速率与离线数据量级一致（±10%）**
- [ ] **在线P95延迟 < 3ms**
- [ ] **事件计数闭合：bull+bear ≈ all；regular+hidden ≈ all**

### 产出物要求
- [ ] `prometheus_divergence.yml` - Prometheus配置
- [ ] `alerting_rules/divergence_alerts.yaml` - 告警规则
- [ ] `dashboards/divergence_overview.json` - Grafana仪表盘
- [ ] `divergence_metrics_exporter.py` - 指标导出器
- [ ] `metrics_alignment_check.py` - 对齐检查脚本

---

## 4️⃣ 配置热更新验收标准

### 配置文件要求
- [ ] `config/system.yaml` - 主配置文件（包含背离检测配置）
- [ ] `config/calibration/divergence_score_calibration.json` - 校准映射
- [ ] 支持条件覆盖规则
- [ ] 支持环境变量配置

### 热更新功能要求
- [ ] **配置文件加载成功**
- [ ] **热更新机制工作正常**
- [ ] **校准配置可用**
- [ ] 支持文件监控
- [ ] 支持参数条件选择

### 配置内容要求
- [ ] 默认参数配置
- [ ] 条件覆盖规则（至少3条）
- [ ] 监控配置
- [ ] 热更新配置
- [ ] 实验配置（可选）

### 产出物要求
- [ ] `config/system.yaml` - 主配置文件（包含背离检测配置）
- [ ] `config/calibration/divergence_score_calibration.json` - 校准映射
- [ ] `scripts/config_hot_update.py` - 热更新脚本
- [ ] 配置加载和选择功能

---

## 🎯 综合验收标准

### 执行要求
- [ ] 所有脚本可独立运行
- [ ] 所有脚本支持`--help`参数
- [ ] 所有脚本有错误处理
- [ ] 所有脚本有进度显示

### 文档要求
- [ ] `docs/divergence_tuning.md` - 完整调优指南
- [ ] `docs/weekly_tasks_acceptance.md` - 验收标准文档
- [ ] 所有脚本有详细注释
- [ ] 所有配置文件有说明

### 测试要求
- [ ] 所有脚本通过语法检查
- [ ] 所有脚本通过基本功能测试
- [ ] 所有配置文件格式正确
- [ ] 所有产出物格式正确

### 性能要求
- [ ] 参数调优在1小时内完成
- [ ] 单调性验证在30分钟内完成
- [ ] 指标对齐在15分钟内完成
- [ ] 配置热更新在5分钟内完成

---

## 📊 验收检查清单

### 快速检查
```bash
# 1. 检查所有脚本是否存在
ls -la scripts/tune_divergence.py
ls -la scripts/score_monotonicity.py
ls -la scripts/metrics_alignment.py
ls -la scripts/config_hot_update.py
ls -la scripts/run_weekly_tasks.py

# 2. 检查配置文件是否存在
ls -la config/system.yaml
ls -la config/calibration/divergence_score_calibration.json

# 3. 检查文档是否存在
ls -la docs/divergence_tuning.md
ls -la docs/weekly_tasks_acceptance.md
```

### 功能检查
```bash
# 1. 测试参数调优
python scripts/tune_divergence.py --help

# 2. 测试单调性验证
python scripts/score_monotonicity.py --help

# 3. 测试指标对齐
python scripts/metrics_alignment.py --help

# 4. 测试配置热更新
python scripts/config_hot_update.py --test

# 5. 测试一键执行
python scripts/run_weekly_tasks.py --help
```

### 结果检查
```bash
# 1. 检查参数调优结果
ls -la runs/tune_*/best_global.yaml
ls -la runs/tune_*/summary.csv

# 2. 检查单调性验证结果
ls -la runs/monotonicity*/score_monotonicity_*.png
ls -la runs/monotonicity*/divergence_score_calibration.json

# 3. 检查指标对齐结果
ls -la runs/metrics*/prometheus_divergence.yml
ls -la runs/metrics*/dashboards/divergence_overview.json

# 4. 检查配置热更新结果
ls -la config/system.yaml
ls -la config/calibration/divergence_score_calibration.json
```

---

## 🚨 常见问题

### 参数调优问题
- **Q**: 没有检测到背离事件
- **A**: 检查数据质量，调整参数范围，增加数据量

- **Q**: 准确率始终为0%
- **A**: 检查前瞻收益计算，调整评分机制

### 单调性验证问题
- **Q**: Spearman相关性为负
- **A**: 检查分数计算逻辑，调整评分权重

- **Q**: 分位曲线不单调
- **A**: 使用等势回归，调整分箱数量

### 指标对齐问题
- **Q**: Prometheus指标为空
- **A**: 检查指标导出器，确认数据源

- **Q**: Grafana面板无数据
- **A**: 检查Prometheus配置，确认数据源连接

### 配置热更新问题
- **Q**: 配置文件加载失败
- **A**: 检查YAML格式，确认文件路径

- **Q**: 热更新不工作
- **A**: 检查文件监控，确认权限设置

---

## 📞 支持联系

如有问题，请：
1. 查看脚本的`--help`参数
2. 检查生成的报告文件
3. 参考本文档的故障排除部分
4. 查看日志文件获取详细错误信息
