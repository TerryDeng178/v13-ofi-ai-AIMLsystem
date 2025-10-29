# Task 1.8：影子与回测一致性比对

## 📋 任务信息
- **阶段**：阶段1 - 真实OFI+CVD核心
- **状态**：⏳ **待开始**
- **优先级**：P0（核心验证）
- **观测窗口**：一致性比对周期
- **范围**：对比影子运行与回测结果的一致性，验证系统逻辑正确性
- **创建时间**：2025-10-22 09:00:00 UTC+09:00

## 🎯 目标（量化可验）
1) **交易触发点一致性**：影子运行与回测在相同时间点的交易触发逻辑一致
2) **交易次数一致性**：影子运行与回测的交易次数差异在可接受范围内
3) **净收益一致性**：影子运行与回测的净收益差异在可接受范围内
4) **Sharpe误差率**：影子运行与回测的Sharpe比率差异在可接受范围内
5) **系统逻辑验证**：确保影子运行与回测使用相同的算法逻辑

## 🚫 非目标
- 不进行真实交易，仅一致性比对
- 不修改算法逻辑，仅验证一致性
- 不接入新的数据源

## 🔗 前置条件
- Task 1.6 6小时影子运行验证完成
- Task 1.7 回测与网格搜索验证完成
- 影子运行结果文件可用
- 回测结果文件可用
- 最佳参数配置确定

## 📦 范围与组件
### 8.1 影子运行结果分析（Shadow Results Analysis）
- **交易触发点**：分析影子运行中的交易触发时间点
- **交易动作**：记录每笔交易的入场/出场动作
- **信号强度**：记录触发时的信号强度（score、z_ofi、z_cvd）
- **时间戳**：精确到毫秒的交易时间记录

### 8.2 回测结果分析（Backtest Results Analysis）
- **回测触发点**：分析回测中的交易触发时间点
- **回测动作**：记录每笔回测交易的入场/出场动作
- **回测信号**：记录回测中的信号强度
- **回测时间戳**：回测中的时间戳记录

### 8.3 一致性比对算法（Consistency Comparison Algorithm）
- **时间对齐**：将影子运行与回测结果按时间对齐
- **触发点比对**：比较相同时间点的交易触发情况
- **次数统计**：统计交易次数的差异
- **收益计算**：计算净收益的差异
- **Sharpe计算**：计算Sharpe比率的差异

### 8.4 误差率分析（Error Rate Analysis）
- **触发点误差率**：交易触发点的时间误差率
- **次数误差率**：交易次数的相对误差率
- **收益误差率**：净收益的相对误差率
- **Sharpe误差率**：Sharpe比率的相对误差率

## 📈 监控指标与SLO（门槛）
### 一致性指标
- **触发点一致性**：时间误差 ≤ 5秒
- **次数一致性**：相对误差 ≤ 10%
- **收益一致性**：相对误差 ≤ 15%
- **Sharpe一致性**：相对误差 ≤ 20%

### 误差率门槛
- **触发点误差率**：≤ 5%
- **次数误差率**：≤ 10%
- **收益误差率**：≤ 15%
- **Sharpe误差率**：≤ 20%

### 准入门槛
- **整体一致性**：所有指标都在可接受范围内
- **逻辑正确性**：影子运行与回测使用相同算法
- **数据完整性**：比对数据完整无缺失
- **结果可信度**：一致性结果可信

## ⚡ 执行步骤（Checklists）
### 预检
- [ ] 确认影子运行结果文件可用
- [ ] 确认回测结果文件可用
- [ ] 验证最佳参数配置一致
- [ ] 检查比对工具可用性

### 执行中
- [ ] 加载影子运行结果数据
- [ ] 加载回测结果数据
- [ ] 执行时间对齐
- [ ] 计算触发点一致性
- [ ] 计算交易次数一致性
- [ ] 计算净收益一致性
- [ ] 计算Sharpe一致性
- [ ] 生成一致性报告

### 收尾
- [ ] 分析一致性结果
- [ ] 识别不一致的原因
- [ ] 输出改进建议
- [ ] 保存比对结果

## 📁 交付物
### 核心文件
- **影子结果分析**：`analysis/shadow_results_analyzer.py`
- **回测结果分析**：`analysis/backtest_results_analyzer.py`
- **一致性比对**：`analysis/consistency_comparator.py`
- **误差率分析**：`analysis/error_rate_analyzer.py`

### 配置文件
- **比对配置**：`configs/consistency_config.yaml`
- **误差率配置**：`configs/error_rate_config.yaml`
- **输出配置**：`configs/output_config.yaml`

### 输出文件
- **一致性报告**：`artifacts/consistency/consistency_report.json`
- **误差率分析**：`artifacts/consistency/error_rate_analysis.csv`
- **比对结果**：`artifacts/consistency/comparison_results.csv`
- **改进建议**：`artifacts/consistency/improvement_suggestions.md`

## 👥 角色与分工
- **算法/量化**：@Quant（一致性算法设计、误差率分析）
- **数据/平台**：@DataEng（数据加载、比对工具）
- **SRE/监控**：@SRE（一致性监控、告警设置）
- **PM/评审**：@PM（结果评审、改进建议）

## 🚨 风险与缓解
- **数据不一致** → 检查数据源，确保使用相同数据
- **时间对齐错误** → 使用精确的时间戳对齐
- **算法差异** → 确保影子运行与回测使用相同算法
- **计算误差** → 使用高精度计算，避免舍入误差

## 🚪 准入门槛（进入下一阶段）
- 所有一致性指标达标
- 误差率在可接受范围内
- 系统逻辑正确性验证通过
- 改进建议明确

## 📋 附录
### A. 一致性比对指标
```yaml
consistency_metrics:
  trigger_point_tolerance: 5s
  trade_count_tolerance: 10%
  net_return_tolerance: 15%
  sharpe_tolerance: 20%
```

### B. 误差率计算公式
```
Error Rate Formulas:
  trigger_point_error = |shadow_time - backtest_time| / shadow_time
  trade_count_error = |shadow_count - backtest_count| / shadow_count
  net_return_error = |shadow_return - backtest_return| / |shadow_return|
  sharpe_error = |shadow_sharpe - backtest_sharpe| / |shadow_sharpe|
```

### C. 一致性比对命令
```bash
# 加载影子运行结果
python analysis/shadow_results_analyzer.py --input artifacts/shadow_run/trade_details.csv --output artifacts/consistency/shadow_analysis.json

# 加载回测结果
python analysis/backtest_results_analyzer.py --input artifacts/backtest/metrics_summary.csv --output artifacts/consistency/backtest_analysis.json

# 执行一致性比对
python analysis/consistency_comparator.py --shadow artifacts/consistency/shadow_analysis.json --backtest artifacts/consistency/backtest_analysis.json --output artifacts/consistency/consistency_report.json

# 误差率分析
python analysis/error_rate_analyzer.py --input artifacts/consistency/consistency_report.json --output artifacts/consistency/error_rate_analysis.csv
```

### D. 预期输出格式
```json
{
  "consistency_summary": {
    "trigger_point_consistency": 0.95,
    "trade_count_consistency": 0.92,
    "net_return_consistency": 0.88,
    "sharpe_consistency": 0.85
  },
  "error_rates": {
    "trigger_point_error": 0.03,
    "trade_count_error": 0.08,
    "net_return_error": 0.12,
    "sharpe_error": 0.15
  },
  "overall_consistency": 0.90,
  "status": "PASS"
}
```

---

## 📊 执行状态

### 当前进展
- **计划执行时间**: 2025-10-22 10:00-12:00 JST (2小时比对周期)
- **比对范围**: 影子运行 vs 回测结果
- **验证重点**: 交易触发点、次数、净收益、Sharpe误差率

### 预检清单
- [ ] 确认影子运行结果文件可用
- [ ] 确认回测结果文件可用
- [ ] 验证最佳参数配置一致
- [ ] 检查比对工具可用性

### 执行清单
- [ ] 加载影子运行结果数据
- [ ] 加载回测结果数据
- [ ] 执行时间对齐
- [ ] 计算触发点一致性
- [ ] 计算交易次数一致性
- [ ] 计算净收益一致性
- [ ] 计算Sharpe一致性
- [ ] 生成一致性报告
- [ ] 分析一致性结果
- [ ] 识别不一致的原因
- [ ] 输出改进建议
- [ ] 保存比对结果

### 执行结果
- **执行时间**: 待执行
- **整体状态**: ⏳ **待开始**
- **一致性状态**: 待比对
- **误差率分析**: 待计算
- **改进建议**: 待生成

### 关键成果
- **一致性验证**: 待完成
- **误差率分析**: 待完成
- **逻辑正确性**: 待验证
- **改进建议**: 待生成

### 产出文件
- `artifacts/consistency/consistency_report.json` - 一致性报告
- `artifacts/consistency/error_rate_analysis.csv` - 误差率分析
- `artifacts/consistency/comparison_results.csv` - 比对结果
- `artifacts/consistency/improvement_suggestions.md` - 改进建议

### 下一步行动
1. **加载结果数据**: 加载影子运行和回测结果数据
2. **执行一致性比对**: 对比交易触发点、次数、净收益、Sharpe
3. **计算误差率**: 分析各项指标误差率
4. **生成改进建议**: 基于比对结果生成改进建议
5. **GO/NO-GO决策**: 基于一致性结果决定是否进入下一阶段

---

**备注**：本任务是对Task 1.6和Task 1.7的验证任务，通过影子运行与回测结果的一致性比对，验证系统逻辑正确性，确保影子运行与回测使用相同的算法逻辑，为进入测试网策略驱动做准备。
