# Task 1.7：回测与网格搜索验证

## 📋 任务信息
- **阶段**：阶段1 - 真实OFI+CVD核心
- **状态**：⏳ **待开始**
- **优先级**：P0（核心验证）
- **观测窗口**：回测验证周期
- **范围**：运行修复后的回测框架，执行网格搜索，验证参数优化效果
- **创建时间**：2025-10-22 08:00:00 UTC+09:00

## 🎯 目标（量化可验）
1) **回测框架验证**：运行修复后的回测框架，验证时间对齐、成本模型、列名统一等修复效果
2) **网格搜索执行**：遍历参数网格，找到最佳参数组合
3) **交易对覆盖**：≥6-10个交易对同时回测，确保数据多样性
4) **多时间窗口**：3个horizons（15s, 30s, 60s）同时验证
5) **产出文件**：生成metrics_summary.csv和best_params.yaml

## 🚫 非目标
- 不进行真实交易，仅回测验证
- 不修改网格搜索参数范围
- 不接入新的数据源

## 🔗 前置条件
- Task 1.6 6小时影子运行验证完成
- Blockers修复完成（时间对齐、成本模型、列名统一、网格搜索）
- 48小时数据收集完成，有充足的历史数据
- 回测框架修复完成，可正常运行

## 📦 范围与组件
### 7.1 回测框架（Backtest Framework）
- **数据加载**：从data/ofi_cvd加载历史数据
- **时间对齐**：修复后的merge_asof时间对齐机制
- **成本模型**：应用交易成本（spread + slippage + commission）
- **信号合并**：OFI、CVD、Fusion信号合并，使用统一列名
- **指标计算**：AUC、IC、Sharpe、回撤等关键指标

### 7.2 网格搜索（Grid Search）
- **参数遍历**：遍历所有参数组合
- **评分机制**：综合AUC和IC的评分函数
- **最佳选择**：选择评分最高的参数组合
- **结果保存**：保存所有参数组合的结果

### 7.3 多交易对验证（Multi-Symbol Validation）
- **交易对选择**：≥6-10个交易对（BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, SOLUSDT等）
- **数据质量**：确保每个交易对有足够的数据量
- **信号一致性**：验证不同交易对的信号质量
- **稳健性测试**：跨交易对的参数稳健性

### 7.4 多时间窗口验证（Multi-Horizon Validation）
- **时间窗口**：15s, 30s, 60s三个horizons
- **前瞻收益**：基于时间对齐的未来收益计算
- **标签生成**：正负收益的二分类标签
- **指标聚合**：跨时间窗口的指标聚合

## 📈 监控指标与SLO（门槛）
### 回测指标
- **AUC**：≥0.55（超过随机水平）
- **IC**：≥0.05（信息系数显著）
- **Sharpe比率**：≥0.5（风险调整后收益）
- **最大回撤**：≤10%（风险控制）

### 网格搜索指标
- **参数组合数**：≥100个组合
- **搜索覆盖率**：100%（遍历所有组合）
- **收敛性**：评分函数收敛
- **最佳参数**：找到评分最高的参数

### 准入门槛
- **跨交易对一致性**：所有交易对都有正收益
- **跨时间窗口稳定性**：3个horizons都表现良好
- **成本后收益**：扣除交易成本后仍有正收益
- **参数稳健性**：最佳参数在多个交易对上表现一致

## ⚡ 执行步骤（Checklists）
### 预检
- [ ] 确认回测框架修复完成，可正常运行
- [ ] 检查数据源完整性，确保有足够的历史数据
- [ ] 验证参数网格配置正确
- [ ] 检查输出目录权限，确保可写入结果文件

### 执行中
- [ ] 启动回测框架，加载历史数据
- [ ] 执行网格搜索，遍历所有参数组合
- [ ] 计算每个参数组合的指标
- [ ] 选择最佳参数组合
- [ ] 生成metrics_summary.csv
- [ ] 生成best_params.yaml

### 收尾
- [ ] 分析回测结果，验证修复效果
- [ ] 检查参数稳健性
- [ ] 输出GO/NO-GO建议
- [ ] 保存所有结果文件

## 📁 交付物
### 核心文件
- **回测框架**：`runner/backtest.py`（已修复）
- **网格搜索**：`runner/backtest.py`中的run_grid_search方法（已修复）
- **参数配置**：`configs/params_grid.json`
- **最佳参数**：`configs/best_params.yaml`

### 配置文件
- **回测配置**：`configs/backtest_config.yaml`
- **网格配置**：`configs/params_grid.json`
- **输出配置**：`configs/output_config.yaml`

### 输出文件
- **指标汇总**：`artifacts/backtest/metrics_summary.csv`
- **最佳参数**：`artifacts/backtest/best_params.yaml`
- **网格结果**：`artifacts/backtest/grid_results.json`
- **回测报告**：`artifacts/backtest/backtest_report.md`

## 👥 角色与分工
- **算法/量化**：@Quant（回测逻辑验证、参数优化）
- **数据/平台**：@DataEng（数据加载、性能优化）
- **SRE/监控**：@SRE（回测稳定性、资源监控）
- **PM/评审**：@PM（结果评审、GO/NO-GO决策）

## 🚨 风险与缓解
- **数据不足** → 检查数据完整性，选择数据充足的交易对
- **计算超时** → 设置超时保护，分批处理
- **内存不足** → 优化内存使用，分批加载数据
- **参数过拟合** → 使用交叉验证，选择稳健参数

## 🚪 准入门槛（进入下一阶段）
- 回测框架正常运行
- 网格搜索完成
- 找到最佳参数组合
- 生成所有必需文件

## 📋 附录
### A. 参数网格配置
```json
{
  "swing_L": [8, 12, 16],
  "z_hi": [1.5, 2.0, 2.5],
  "z_mid": [0.5, 1.0, 1.5],
  "ema_alpha": [0.15, 0.2, 0.25],
  "ofi_win": [50, 100, 150],
  "cvd_win": [50, 100, 150]
}
```

### B. 回测指标定义
```
Metrics:
  auc: Area Under Curve
  ic: Information Coefficient
  sharpe: Sharpe Ratio
  max_drawdown: Maximum Drawdown
  calmar: Calmar Ratio
  hit_rate: Hit Rate
  avg_return: Average Return
  std_return: Standard Deviation
```

### C. 回测执行命令
```bash
# 运行回测
python runner/backtest.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,SOLUSDT --start 2025-10-15 --end 2025-10-22 --horizons 15,30,60 --grid configs/params_grid.json --output artifacts/backtest

# 网格搜索
python runner/backtest.py --grid-search --symbols BTCUSDT,ETHUSDT,BNBUSDT --grid configs/params_grid.json --output artifacts/backtest

# 结果分析
python scripts/analyze_backtest.py --input artifacts/backtest/metrics_summary.csv --output artifacts/backtest/analysis_report.md
```

### D. 预期输出格式
```csv
symbol,horizon,auc,ic,sharpe,max_drawdown,calmar,hit_rate,avg_return,std_return
BTCUSDT,15,0.612,0.045,0.78,0.08,9.75,0.58,0.12,0.15
BTCUSDT,30,0.598,0.038,0.65,0.09,7.22,0.55,0.10,0.18
BTCUSDT,60,0.584,0.032,0.52,0.11,4.73,0.52,0.08,0.21
```

---

## 📊 执行状态

### 当前进展
- **计划执行时间**: 2025-10-22 09:00-12:00 JST (3小时回测周期)
- **目标交易对**: ≥6-10个交易对（BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, SOLUSDT等）
- **验证重点**: 回测框架修复效果、网格搜索功能、参数优化

### 预检清单
- [ ] 确认回测框架修复完成，可正常运行
- [ ] 检查数据源完整性，确保有足够的历史数据
- [ ] 验证参数网格配置正确
- [ ] 检查输出目录权限，确保可写入结果文件

### 执行清单
- [ ] 启动回测框架，加载历史数据
- [ ] 执行网格搜索，遍历所有参数组合
- [ ] 计算每个参数组合的指标
- [ ] 选择最佳参数组合
- [ ] 生成metrics_summary.csv
- [ ] 生成best_params.yaml
- [ ] 分析回测结果，验证修复效果
- [ ] 检查参数稳健性
- [ ] 输出GO/NO-GO建议
- [ ] 保存所有结果文件

### 执行结果
- **执行时间**: 待执行
- **整体状态**: ⏳ **待开始**
- **回测状态**: 待运行
- **网格搜索**: 待执行
- **参数优化**: 待完成

### 关键成果
- **回测框架**: 待验证
- **网格搜索**: 待执行
- **参数优化**: 待完成
- **指标计算**: 待生成

### 产出文件
- `artifacts/backtest/metrics_summary.csv` - 指标汇总
- `artifacts/backtest/best_params.yaml` - 最佳参数
- `artifacts/backtest/grid_results.json` - 网格结果
- `artifacts/backtest/backtest_report.md` - 回测报告

### 下一步行动
1. **启动回测框架**: 使用修复后的回测框架运行回测
2. **执行网格搜索**: 遍历所有参数组合，找到最佳参数
3. **生成结果文件**: 输出metrics_summary.csv和best_params.yaml
4. **结果分析**: 分析回测结果，验证修复效果
5. **GO/NO-GO决策**: 基于回测结果决定是否进入下一阶段

---

**备注**：本任务是对Task 1.5核心算法v1的验证任务，通过回测和网格搜索验证Blockers修复效果，确保回测框架稳定性和参数优化功能，为进入测试网策略驱动做准备。
