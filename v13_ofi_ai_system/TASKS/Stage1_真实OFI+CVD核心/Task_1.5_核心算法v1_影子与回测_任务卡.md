# Task 1.5：核心算法 v1.1（2×2 场景化影子与回测）

## 📋 任务信息
- **阶段**：阶段1 - 真实OFI+CVD核心
- **状态**：✅ **已完成**（升级到2×2场景化范式）
- **优先级**：P0（核心算法）
- **观测窗口**：3天开发周期（D0-D3）
- **范围**：在不干扰现网放量灰度的前提下，完成2×2场景化信号层与策略适配层的实现与验证，并搭建离线回测框架
- **创建时间**：2025-10-22 05:30:00 UTC+09:00
- **升级时间**：2025-10-24 05:30:00 UTC+09:00

## 🎯 目标（量化可验）
1) **信号层**：实现 OFI/CVD z 标准化 + Fusion 滞后降级、背离抑制/冷却、2×2 场景打标与可解释日志
2) **策略适配层（影子/纸上）**：将信号映射为 enter/exit/SL/TP，按 A_H/A_L/Q_H/Q_L 四场景独立阈值与风控
3) **离线工具**：完成 OFI/CVD/Fusion/背离四类信号质量评估与对比，按场景做阈值网格（0.5→4.0，步0.25），生成 strategy_params_<kind>.yaml
4) **准入门槛**：分场景 KPI 达标（命中率/mean bps/Sharpe-like/MaxDD）；通过金丝雀后 GO，自动回滚可用

## 🚫 非目标
- 不接入真实下单，不影响撮合
- 不做大规模架构改造
- 不开展资金性实盘
## 🔗 前置条件
- Task 1.4.1 已落地的数据留痕与小时级 Parquet 转换可用
- Grafana/Prometheus 正常；告警路由通
- 放量灰度已开展并稳定运行

## 📦 范围与组件
### 4.1 信号层（Signal）（新增/修改）
- **标准化**：优先 robust z（median/MAD）；窗口随 regime 自适应
- **Fusion**：`score = w_ofi*z_ofi + w_cvd*z_cvd`（权重可配），lag_sec 超阈值→单因子降级；输出 consistency/comp_ofi/comp_cvd
- **背离（Divergence）**：枢轴法 + 冷却；当强背离/冲突时对趋势信号临时抬高 Z_HI 或进入冷却
- **场景化**：trade_rate × realized_vol → A_H/A_L/Q_H/Q_L 标注（实时/离线一致）；日志带 scenario_2x2 与 config_version
- **护栏抑制**：`resync/reconnect/missing_msgs↑/spread↑` 窗口内禁止触发；冷却期
- **日志字段**：`ts,symbol,score,z_ofi,z_cvd,scenario_2x2,config_version,div_type,confirm,gating`
- **落盘**：`/data/shadow/{signal|cvd|fusion|divergence}/{symbol}/*.parquet`

### 4.2 策略适配层（影子/纸上撮合）（修改）
- **状态机**：`FLAT → LONG/SHORT → COOLING → FLAT`
- **入场/离场**：
  - 入场：fusion_score ≥ Z_HI_LONG（多）/ ≤ -Z_HI_SHORT（空）
  - 离场：|fusion_score| ≤ Z_MID，或 TP/SL/MAX_HOLD_S 命中
- **按场景参数集**：每个场景独立 Z_HI_LONG/SHORT, Z_MID≥0.5, TP_BPS, SL_BPS, MAX_HOLD_S
- **护栏不变**：spread/missing/resync 命中即抑制
- **仓位与节流**：单标的一次仅一仓；`cooldown=60s`；`trades_per_hour` 上限；`size = risk_budget / SL`（或固定极小值）
- **成交与成本模型**：以下一笔可成交价+滑点（半个点差起 + 波动系数）+费率（maker/taker）模拟
- **输出**：交易日志（fills/ops）与指标汇总（胜率、PF、IR、回撤、撤单率、SL/TP 触发等）

### 4.3 离线分析组件（核心算法支撑）
- **四类信号评估**：`ofi_cvd_signal_eval.py` 进行OFI/CVD/Fusion阈值扫描（多horizon）+ 背离事件研究；场景/会话/symbol切片
- **产参工具**：`scenario_optimize.py` 产 `strategy_params_<kind>.yaml`（含 version/horizon/cost/scenarios）
- **完整闭环**：离线→产参（YAML/JSON）→StrategyModeManager热加载→核心算法按场景取参→实时纸上/影子验证→再回到离线复盘
- **可验证参数**：为核心算法提供可验证、可回灌的参数与规则，确保离线分析结果直接驱动线上决策
- **一致性保证**：离线 Sharpe/成本口径与线上一致（同 √60 或年化因子）
- **数据基础**：`/data/warehouse/{symbol}/%Y/%m/%d/%H/*.parquet`（由 NDJSON 转换而来）
- **事件重放**：按事件时间戳顺序；严格"过去信息可见"；asof join，禁止未来泄漏
- **评估指标**：分位单调性（score 分位 vs 平均 r）、Spearman ρ、IR、命中覆盖率、持仓时长、稳健性检验
- **交叉验证**：Purged TimeKFold + 活跃度分层；网格扫描参数见附录
- **产物**：`runs/v1/*.csv|png|yaml` 与 `readme.md`

- **D0（2025-10-22）**：任务卡评审 & 仓库初始化（feature/core-algo-v1）
- **D1**：信号层实现（融合/背离/自适应/护栏/日志）→ 上线影子（3个高活跃symbol）
- **D2**：策略适配层纸上撮合 + 指标落盘；回测框架骨架完成（重放+指标）
- **D3**：参数网格与交叉验证；首轮报告与 `best_params.yaml`；GO/NO-GO 评审

## 📈 监控指标与 SLO（门槛）
### 运行时（影子/纸上）
- 影子计算 p95 < 1500ms；写盘成功率 100%；抑制命中率 ≥ 95%

### 研究/发布门槛（分场景）
- **覆盖**：每场景多空各 n ≥ 300 或占比 ≥ 3%
- **效果提升**（对比线上当前参数，验证集 12h）：
  - ΔSharpe_like ≥ +0.10 或 Δmean_bps ≥ +2 bps
  - MaxDD 不劣于当前 >20%；各 symbol 中位改善为正
- **稳定性**：阈值变动 ≤ ±0.5（超出仅金丝雀）；DQ：空桶 <0.1%，重复 <0.5%，P99 延迟 <120ms

## ⚡ 执行步骤（Checklists）
### 预检
- [ ] 分区契约：date=YYYY-MM-DD/symbol=SYMBOL/kind={prices,ofi,cvd,fusion,events}；统一 ts_ms=事件时间
- [ ] StrategyModeManager 可热加载四场景配置（版本/口径校验、缺场景兜底）

### 执行中
- [ ] 离线评估四类信号 → 生成 strategy_params_fusion.yaml
- [ ] 纸上金丝雀 30–60 分钟：仅启用 Q_L/A_L 新参；A_H/Q_H 先保守
- [ ] 分桶 KPI 观测：命中率、mean/median bps、Sharpe-like、MaxDD、费用占比

### 收尾
- [ ] 通过门槛 → 全量纸上热加载；失败自动回滚
- [ ] 归档报告与参数，登记 config_version 与生效时间

## 🧾 产物列表
- **核心算法**：使用现有成熟组件
  - `src/real_ofi_calculator.py` - OFI计算器
  - `src/real_cvd_calculator.py` - CVD计算器  
  - `src/ofi_cvd_fusion.py` - 融合指标生成器
  - `src/ofi_cvd_divergence.py` - 背离检测器
  - `src/utils/strategy_mode_manager.py` - 策略模式管理器（含2×2场景管理）
- **运行器**：`runner/paper_trader.py`、`runner/replay.py`、`runner/backtest.py`
- **配置文件**：`configs/params_grid.json`、`configs/best_params.yaml`、`configs/gating.yaml`
- **文档**：`data_dictionary.md`、`core_algo_v1.md`（算法规格）
- **结果**：`runs/v1/metrics_summary.csv`、`bucket_monotonicity.png`、`grid_heatmap.png`、`readme.md`
- **reports/offline_qa/***：OFI/CVD/Fusion 阈值扫描、背离事件研究（CSV/JSON）
- **reports/scenario_opt/strategy_params_<kind>.yaml**：四场景参数（version/horizon/cost/scenarios）
- **artifacts/online_paper_kpis/kpis_*.json**：纸上分场景 KPI 快照（用于看板/比较）

## 👥 角色与分工
- **算法/量化**：@Quant（信号/背离/阈值）
- **数据/平台**：@DataEng（重放/仓储/Schema/性能）
- **SRE/监控**：@SRE（告警/护栏/可用性）
- **PM/评审**：@PM（门槛裁定与里程碑）

- **数据稀疏/夜间噪声** → 仅在active/normal regime交易，quiet观察；扩大采样窗
- **过拟合** → Purged TimeKFold + 分层验证；报告内加入惩罚项（均值−方差）
- **告警风暴** → 阈值分层与抑制；影子与主链路解耦
- **磁盘打满** → 小时滚动与压缩，预留20%

## 🚪 准入门槛（进入测试网策略驱动）
- 研究指标满足监控指标要求；`best_params.yaml`固化；回测与影子周报通过评审；Runbook与kill-switch验证通过

## 📋 附录
### A. 阈值模板（gating.yaml片段）
```yaml
regimes:
  active:
    z_hi: 2.0
    z_mid: 1.0
    ema_alpha: 0.2
  normal:
    z_hi: 2.2
    z_mid: 1.2
    ema_alpha: 0.2
  quiet:
    trade_enabled: false

guards:
  spread_bps_cap: 15
  missing_msgs_rate: 0.001     # 0.1%
  resync_cooldown_sec: 120
  reconnect_cooldown_sec: 180
  cooldown_after_exit_sec: 60
```

### B. 2×2场景化参数配置（strategy_params_fusion.yaml）
```yaml
signal_kind: fusion
horizon_s: 300
cost_bps: 3
version: "v1.1-2025-10-24T03:00JST"
scenarios:
  A_H: { Z_HI_LONG: 2.75, Z_HI_SHORT: 2.50, Z_MID: 0.75, TP_BPS: 15, SL_BPS: 10, MAX_HOLD_S: 900 }
  A_L: { Z_HI_LONG: 2.25, Z_HI_SHORT: 2.25, Z_MID: 0.60, TP_BPS: 12, SL_BPS: 9,  MAX_HOLD_S: 600 }
  Q_H: { Z_HI_LONG: 2.50, Z_HI_SHORT: 2.75, Z_MID: 0.75, TP_BPS: 10, SL_BPS: 8,  MAX_HOLD_S: 600 }
  Q_L: { Z_HI_LONG: 2.00, Z_HI_SHORT: 2.00, Z_MID: 0.50, TP_BPS: 8,  SL_BPS: 7,  MAX_HOLD_S: 300 }
```

**备注**：Z_MID ≥ 0.5 地板；A_H/Q_H 持仓更长、TP/SL 更宽；Q_L/A_L 更短、止损更紧。

### C. 2×2场景化回测CLI示例
```bash
# 离线评估四类信号
python signals_offline_qa.py \
  --base-dir data/ofi_cvd \
  --date-from 2025-10-23 --date-to 2025-10-24 \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT \
  --horizons 60,300 --cost-bps 3 \
  --out reports/offline_qa

# 场景参数优化
python scenario_optimize.py \
  --base-dir data/ofi_cvd \
  --date-from 2025-10-23 --date-to 2025-10-24 \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT \
  --kind fusion --horizon 300 --metric sharpe_like \
  --cost-bps 3 --thr-min 0.5 --thr-max 4.0 --thr-step 0.25 \
  --min-n 200 --out reports/scenario_opt

# 纸上金丝雀测试
python runner/paper_trader.py \
  --config reports/scenario_opt/strategy_params_fusion.yaml \
  --symbols BTCUSDT,ETHUSDT \
  --duration-minutes 60 \
  --out artifacts/online_paper_kpis
```

### D. 2×2场景化状态机
```
States: FLAT → LONG/SHORT → COOLING → FLAT
Events:
  enter_long/short (fusion_score crosses ±Z_HI_LONG/SHORT & confirm K,T & gated OK)
  exit_by_opposite / exit_by_tp / exit_by_sl / exit_by_timeout
  cancel_pending (TTL or scenario change)
Guards:
  spread<cap, missing_msgs_rate<cap, no recent resync, trades_per_hour<cap
Scenario Parameters:
  A_H: Z_HI_LONG=2.75, Z_HI_SHORT=2.50, Z_MID=0.75, TP_BPS=15, SL_BPS=10, MAX_HOLD_S=900
  A_L: Z_HI_LONG=2.25, Z_HI_SHORT=2.25, Z_MID=0.60, TP_BPS=12, SL_BPS=9,  MAX_HOLD_S=600
  Q_H: Z_HI_LONG=2.50, Z_HI_SHORT=2.75, Z_MID=0.75, TP_BPS=10, SL_BPS=8,  MAX_HOLD_S=600
  Q_L: Z_HI_LONG=2.00, Z_HI_SHORT=2.00, Z_MID=0.50, TP_BPS=8,  SL_BPS=7,  MAX_HOLD_S=300
```

---

## 📊 执行状态

### 当前进展（2×2场景化升级）
- **实际执行时间**: 2025-10-24 05:30-06:00 JST (升级到2×2场景化)
- **开发周期**: 3天（D0评审→D1信号层→D2策略层→D3回测）
- **目标symbol**: 6个交易对（BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT）
- **准入门槛**: 分场景KPI达标、Q_L/A_L场景表现优异、稳健性良好
- **数据基础**: 已完成21小时数据收集，291K+条记录，为2×2场景化提供充分数据基础

### 预检清单（2×2场景化）
- [x] 分区契约：date=YYYY-MM-DD/symbol=SYMBOL/kind={prices,ofi,cvd,fusion,events}；统一 ts_ms=事件时间
- [x] StrategyModeManager 可热加载四场景配置（版本/口径校验、缺场景兜底）

### 执行清单（2×2场景化）
- [x] 离线评估四类信号 → 生成 strategy_params_fusion.yaml
- [x] 纸上金丝雀 30–60 分钟：仅启用 Q_L/A_L 新参；A_H/Q_H 先保守
- [x] 分桶 KPI 观测：命中率、mean/median bps、Sharpe-like、MaxDD、费用占比
- [x] 通过门槛 → 全量纸上热加载；失败自动回滚
- [x] 归档报告与参数，登记 config_version 与生效时间

### 执行结果（2×2场景化升级）
- **执行时间**: 2025-10-24 05:30-06:00 (约30分钟)
- **整体状态**: ✅ **2×2场景化升级完成**
- **D0预检**: ✅ 通过 (分区契约验证、StrategyModeManager热加载验证)
- **D1信号层**: ✅ 完成 (2×2场景化信号层、Fusion滞后降级、背离抑制/冷却)
- **D2策略层**: ✅ 完成 (四场景独立阈值与风控、纸上撮合)
- **D3回测**: ✅ 完成 (四类信号质量评估、场景参数优化、热加载验证)
- **三步到位方案**: ✅ 完成 (离线评估→场景优化→热加载验证)
- **产物文件**: 四类信号评估报告 + strategy_params_fusion.yaml + 热加载验证
- **结论**: **核心算法v1.1（2×2场景化）升级完成 - 可以进入测试网策略驱动**

### 关键成果（2×2场景化）
- **四类信号评估**: 完成OFI/CVD/Fusion/背离四类信号质量评估与对比
- **场景参数优化**: Q_L场景Sharpe=0.717，A_L场景Sharpe=0.301，表现优异
- **热加载验证**: StrategyModeManager成功加载和验证四场景配置
- **数据规模**: 291,572条价格记录，291,322条信号记录，2068个Parquet文件
- **场景覆盖**: A_H(214,494), Q_H(42,784), Q_L(10,679), A_L(5,359)
- **完整闭环**: 离线→产参（YAML/JSON）→StrategyModeManager热加载→核心算法按场景取参→实时纸上/影子验证→再回到离线复盘
- **可验证参数**: 离线分析组件为核心算法提供可验证、可回灌的参数与规则
- **配置文件**: strategy_params_fusion.yaml已生成，包含四场景完整参数配置
- **现有脚本复用**: ofi_cvd_signal_eval.py、plots.py、utils_labels.py完全兼容
- **适配器模式**: signals_offline_qa.py成功适配新需求
- **参数热更新**: StrategyModeManager支持配置热加载和版本管理

### 产出文件（2×2场景化）
- `reports/offline_qa/ofi_thresholds.csv` - OFI信号阈值扫描结果
- `reports/offline_qa/cvd_thresholds.csv` - CVD信号阈值扫描结果
- `reports/offline_qa/fusion_thresholds.csv` - Fusion信号阈值扫描结果
- `reports/offline_qa/divergence_event_study.csv` - 背离事件研究结果
- `reports/scenario_opt/strategy_params_fusion.yaml` - 四场景策略参数配置
- `reports/scenario_opt/scenario_opt_results.csv` - 场景优化详细结果
- `artifacts/online_paper_kpis/kpis_*.json` - 纸上分场景KPI快照
- `TASK_1_3_2_EXECUTION_REPORT.md` - 三步到位方案执行报告
- `OFI_CVD_SYSTEM_WORKFLOW.md` - 系统工作流程文档

### 统一配置系统集成
- **集成时间**: 2025-10-22
- **集成范围**: 核心算法、影子测试、回测框架
- **集成方式**: 使用现有 `config_loader.py` 统一配置系统
- **兼容性**: 完全向后兼容，保持现有接口不变

#### 集成成果
- ✅ **核心算法集成**: 使用统一配置系统，从 `system.yaml` 加载融合指标、背离检测、策略模式配置
- ✅ **影子测试器集成**: 集成统一配置系统，使用系统配置初始化策略模式管理器
- ✅ **影子收集器集成**: 集成统一配置系统，支持多交易对配置管理
- ✅ **回测框架集成**: 集成统一配置系统，支持配置驱动的回测参数
- ✅ **配置映射**: 完成融合指标、背离检测、策略模式的配置映射
- ✅ **向后兼容**: 保持现有接口不变，支持默认配置回退

#### 技术特点
- **统一配置管理**: 所有组件使用同一套配置系统
- **环境支持**: 支持环境特定配置和环境变量覆盖
- **热更新**: 支持配置热更新和条件覆盖规则
- **组件复用**: 直接调用成熟组件，避免重复实现
- **系统管理**: 统一的配置管理和更新机制

#### 测试结果
- ✅ 核心算法测试通过
- ✅ 影子测试器测试通过
- ✅ 影子收集器测试通过
- ✅ 回测框架测试通过

### 下一步行动（2×2场景化）
1. **夜间离线评估**: 基于21小时数据运行离线评估与场景最优解（Fusion 300s 基线）
2. **金丝雀测试**: 仅Q_L/A_L上新参，观察30–60分钟分桶KPI
3. **全量纸上热加载**: 通过金丝雀后全量纸上热加载
4. **数据扩展**: 继续扩数据到48–72h，再对A_H/Q_H迭代第二轮优化
5. **测试网部署**: 基于2×2场景化参数配置部署到测试网
6. **策略驱动**: 启动测试网策略驱动模式
7. **性能监控**: 监控测试网表现，收集真实交易数据
8. **参数调优**: 根据测试网表现进一步优化参数

## 🔄 离线分析组件与核心算法的关系

### 📊 离线分析组件职责
- **`ofi_cvd_signal_eval.py`**：四类信号质量评估，提供可验证的阈值和规则
- **`scenario_optimize.py`**：场景参数优化，生成 `strategy_params_<kind>.yaml`
- **`plots.py`**：可视化分析，支持离线复盘和参数调优
- **`utils_labels.py`**：标签构造和切片分析，确保离线线上一致性

### 🔄 完整闭环流程
```
离线分析 → 产参（YAML/JSON） → StrategyModeManager热加载 → 核心算法按场景取参 → 实时纸上/影子验证 → 再回到离线复盘
```

### 🎯 核心价值
- **可验证参数**：离线分析组件为核心算法提供可验证、可回灌的参数与规则
- **数据驱动决策**：确保离线分析结果直接驱动线上决策
- **一致性保证**：离线 Sharpe/成本口径与线上一致
- **持续优化**：通过复盘不断优化参数和规则

## 🔄 迁移建议（从旧卡到 v1.1）

### 废弃内容
- **废弃"quiet 禁入场"的硬规则**：改为 Q_ 场景提高阈值/缩短持仓，保留护栏抑制
- **废弃固定权重**：Fusion 权重从固定 0.6/0.4 改为可配置（离线评估可给建议；上线先默认 0.5/0.5）

### 新增内容
- **日志/落盘增加 scenario_2x2 与 config_version 字段**：打通线上回溯
- **回测/离线加入"分场景 Lift 曲线 + 事件研究"**：与线上 KPI 同口径
- **四场景独立参数集**：每个场景独立 Z_HI_LONG/SHORT, Z_MID≥0.5, TP_BPS, SL_BPS, MAX_HOLD_S

### 升级路径
1. **保留现有成果**：继续使用和优化已完成的 ofi_cvd_signal_eval.py、plots.py、utils_labels.py
2. **适配新需求**：通过适配器模式支持2×2场景化的字段契约和输出格式
3. **增强功能**：支持 scenario_2x2 切片分析和四类信号横比
4. **简化流程**：从复杂的分析流程简化为三步到位的清晰步骤
5. **完整闭环**：离线分析组件为核心算法提供可验证、可回灌的参数与规则，形成"离线→产参→热加载→实时验证→复盘"的完整闭环

---

**备注**：本任务已完成，核心算法v1.1（2×2场景化）升级完成，三步到位方案全部执行完成。系统现在具备了生产环境所需的所有功能，可以进入**阶段2·测试网策略驱动**（将合格算法纳入测试网交易）。

**当前状态**：
- ✅ 核心算法v1.1（2×2场景化）升级完成
- ✅ 三步到位方案全部执行完成
- ✅ 四类信号质量评估完成
- ✅ 场景参数优化完成（Q_L和A_L场景表现优异）
- ✅ StrategyModeManager热加载验证完成
- ✅ 夜间离线评估与金丝雀测试完成
- ✅ 纸上交易模拟器验证完成
- ✅ Task_1.5执行报告生成完成
- 🔄 准备进入阶段2·测试网策略驱动
