# Task 1.3.3 (v3 Final): 分析 OFI / CVD / Fusion 的预测能力

## 📋 任务信息

- **任务编号**: Task_1.3.3
- **所属阶段**: 阶段1 - 真实 OFI 核心
- **状态**: ✅ **已完成**
- **优先级**: 高
- **预计时间**: 3–4 小时（离线批评估）

## 🎯 目标（量化可验）

对 OFI、CVD、Fusion、背离 四类信号在多个前瞻窗口上的预测能力进行稳健评估（分类、排序、校准）。

输出总体与切片（Active/Quiet、Tokyo/London/NY、波动分位、品种）的指标与图表，并给出最佳/稳健阈值与可部署建议。

生成可复现的产物（CSV/JSON/图），并将关键元数据（config_fingerprint、cvd_direction、merge 对齐分布等）写入报告与 run_tag。

## 📥 输入与数据契约

来自 Task_1.3.1 / 1.3.2 的分区化数据（Snappy Parquet；1 分钟轮转）：

**prices**: ts_ms, event_ts_ms, price, ...

**ofi**: ts_ms, ofi_z, scale, regime, ...

**cvd**: ts_ms, z_cvd, scale, sigma_floor, ...

**fusion**: ts_ms, score（如上游未写该列，本任务会现场重算）

**events**: ts_ms, event_type, meta_json（背离/枢轴/异常）

**标签**：默认使用 mid（(best_bid+best_ask)/2）前瞻收益构造；有顶层队列量时可切换 microprice；均采用 forward asof 对齐，容差可配（默认 1500ms）。

## 🧠 指标与方法

**分类/排序**：AUC、PR-AUC、F1、Top-K 命中率（5%/10%）、Lift/Gain。

**相关性**：IC（Spearman），单调性（Q1→Q5 前瞻收益，Kendall τ 检验）。

**校准**：Brier、ECE；默认 Platt（可选 Isotonic），滑窗训练/验证。

**阈值扫描**：|Z|∈[0.5, 3.0], step=0.1；目标函数 PR-AUC 最大，平手用 Top-K 命中率裁决；输出"最佳/稳健"双阈值。

**Fusion**：fusion_raw = w_ofi*ofi_z + w_cvd*z_cvd（方向自检后参与融合）；默认 gate=0，由校准/排序承载识别力；可按切片另配 gate。

**切片**：regime（Active/Quiet）、ToD（Tokyo/London/NY）、波动分位、symbol（BTC/ETH…）。

## ⏱️ 前瞻窗口

**标准档**：60/180/300/900 秒（与 1 分钟落盘一致，鲁棒）

**低延时档（可选）**：5/10/30 秒（需 tick 级对齐与更小容差）

## 📦 产出与目录

```
artifacts/analysis/ofi_cvd/
  run_tag.txt                        # 启动指纹、方向、窗口等
  summary/metrics_overview.csv       # 各信号×各窗口的总表
  summary/slices_*.csv               # Active/Quiet、ToD、Vol、Symbol 切片
  summary/merge_time_diff_ms.csv     # 合并时差 p50/p90/p99
  summary/platt_samples.csv          # 训练/测试样本量
  charts/                            # ROC、PR、单调性、校准、Top-K、背离
  reports/report_{YYYYMMDD}.json     # 机器可读摘要（含阈值与建议）
```

## 🛠️ 运行方式（CLI 示例）

### Bash（WSL/Git Bash）

```bash
python -m v13_ofi_ai_system.analysis.ofi_cvd_signal_eval \
  --data-root data/ofi_cvd \
  --symbols ETHUSDT,BTCUSDT \
  --date-from 2025-10-21 --date-to 2025-10-22 \
  --horizons 60,180,300,900 \
  --labels mid \
  --use-l1-ofi --cvd-auto-flip \
  --fusion "w_ofi=0.6,w_cvd=0.4,gate=0" \
  --calibration platt \
  --calib-train-window 7200 --calib-test-window 1800 \
  --merge-tol-ms 1500 \
  --plots all \
  --out artifacts/analysis/ofi_cvd \
  --run-tag 20251022_eval_v3
```

### PowerShell

```powershell
python -m v13_ofi_ai_system.analysis.ofi_cvd_signal_eval `
  --data-root data\ofi_cvd `
  --symbols ETHUSDT,BTCUSDT `
  --date-from 2025-10-21 --date-to 2025-10-22 `
  --horizons 60,180,300,900 `
  --labels mid `
  --use-l1-ofi --cvd-auto-flip `
  --fusion "w_ofi=0.6,w_cvd=0.4,gate=0" `
  --calibration platt `
  --calib-train-window 7200 --calib-test-window 1800 `
  --merge-tol-ms 1500 `
  --plots all `
  --out artifacts\analysis\ofi_cvd `
  --run-tag 20251022_eval_v3
```

## ✅ DoD / Gate（硬验收）

### 全局主 Gate（二选一通过）

- **Fusion AUC ≥ 0.58**（任一主窗口）且 PR-AUC/Top-K 同向提升；ECE ≤ 0.10；
- **切片 Plan-B**：在 Active 或 Tokyo/London/NY 任一切片上 Fusion AUC ≥ 0.60 且 ECE ≤ 0.10，允许先在该切片放量，上线其余切片保守配置。

### 质量门槛（全部必须）

- merge_asof 匹配率 ≥ 80%；合并时差 p90 ≤ 500ms、p99 ≤ 900ms
- 生成 metrics_overview.csv / slices_*.csv / report_*.json / charts/ 全量产物
- 报告写入：config_fingerprint、cvd_direction(as_is|flipped)、platt_{train,test}_samples、merge_time_diff_ms_{p50,p90,p99}
- 单调性（Kendall τ）p < 0.05（≥1 个窗口）

## 📊 输出（report_*.json 示例）

```json
{
  "run_tag": "20251022_eval_v3",
  "config_fingerprint": "sha1:xxxx",
  "cvd_direction": "flipped",
  "windows": {"60s":{"AUC":0.60,"PR_AUC":0.11,"IC":0.04,"ECE":0.08}, "300s":{"AUC":0.62}},
  "slices": {"Active":{"AUC":0.63}, "London":{"AUC":0.61}},
  "thresholds": {"ofi":1.7, "cvd":1.5, "fusion":{"gate":0.0, "w_ofi":0.6, "w_cvd":0.4}},
  "merge_time_diff_ms": {"p50":155, "p90":335, "p99":758},
  "platt_samples": {"train":123293, "test":52840},
  "recommendation": "可在 Active/London 切片放量，其他切片保守。"
}
```

## 📝 执行步骤（Checklist）

1. 读取五类分区数据 → schema 校验 → forward asof 构造 mid/micro 标签；
2. OFI/CVD 方向自检（AUC(x) vs AUC(−x)）→ 以更优方向进入评估与融合；
3. 动态融合（gate=0）→ 校准（Platt/Isotonic） → 计算分类/排序/校准/单调性/Top-K；
4. 切片评估（regime/ToD/Vol/Symbol）→ 计算 ΔAUC 与稳定性；
5. 阈值扫描 与 "最佳/稳健"双阈值确定；
6. 生成 CSV/JSON/图表 → 写入 run_tag 与关键元数据；
7. 运行 DoD/Gate → 通过则回写阶段索引并进入 1.3.4。

## 📦 Allowed Files

- `v13_ofi_ai_system/analysis/ofi_cvd_signal_eval.py`（执行）
- `v13_ofi_ai_system/analysis/plots.py`（作图）
- `v13_ofi_ai_system/analysis/utils_labels.py`（标签/切片/校验）
- `v13_ofi_ai_system/data/ofi_cvd/...`（只读输入）

## 🔗 关联

- **上一任务**: [Task_1.3.2_创建OFI+CVD信号分析工具](./Task_1.3.2_创建OFI+CVD信号分析工具.md)
- **下一任务**: [Task_1.3.4_生成OFI+CVD验证报告](./Task_1.3.4_生成OFI+CVD验证报告.md)
- **阶段总览**: [📋V13_TASK_CARD.md](../../📋V13_TASK_CARD.md)

## ⚠️ 风险与回滚

- **校准漂移**：若 ECE>0.10 且样本骤降 → 回退"未校准评分 + 权重 0.5/0.5"，并标注需复核。
- **对齐异常**：若合并 p99>900ms 或匹配率<80% → 放宽/收紧 --merge-tol-ms 并复测。
- **方向偏置**：若翻转建议频繁变更 → 排查标签定义与 tick-rule 约束，必要时锁定方向至下一版本。

## 🔄 与旧版任务卡的差异

- 将"准确率"更换为更鲁棒的 AUC/PR-AUC/IC/单调性/校准 作为核心标准；
- 前瞻窗口从 5/10/30s 升级为 60/180/300/900s（并保留低延时档可选）；
- 引入 Plan-B 切片放量策略与 关键元数据写入；
- 明确 fusion gate=0 的默认实践；
- 补齐 产出目录、DoD/Gate、运行命令 与 可回滚策略。

---

**任务状态**: ✅ **已完成**  
**质量评分**: 9/10 (核心指标达标，图表修复完成，具备生产灰度条件)  
**是否可以继续下一个任务**: ✅ **可以继续Task_1.3.4，已具备生产灰度条件**