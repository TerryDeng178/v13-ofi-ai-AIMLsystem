# Task 1.2.12 — OFI-CVD 背离检测（v1.1 优化版）

## 📋 任务信息

- **任务编号**: Task_1.2.12
- **任务名称**: OFI-CVD 背离检测
- **所属阶段**: 1.2 真实 OFI + CVD 计算
- **优先级**: 高
- **任务状态**: ⏳ 待开始
- **相关依赖**:
  - ✅ 1.2.11 融合指标可用（fusion_score, consistency, 去噪机制）
  - ✅ 实时数据对齐：OFI/CVD/Price 统一采样 & 滞后控制（max_lag）
- **产出物**: 代码、单测、回测报告、Prometheus 指标、Grafana 面板

---

## 🎯 任务目标（明确且可验收）

实现面向反转与风险提示的背离检测：

- 支持价格 vs OFI、价格 vs CVD、**价格 vs Fusion（可选）**三条通道
- 提供正向（看涨）/负向（看跌）与隐藏背离（趋势延续）
- 生成事件化输出（含强度评分、参与的枢轴/窗口、理由标签）
- 具备去噪与频控，并通过回测量化有效性

---

## 🧩 范围说明

**包含**：背离检测逻辑、评分、事件去重/冷却、Prometheus 指标、单测与回测脚本、Grafana 面板配置建议

**不包含**：入场/风控策略实现（由上层策略消费本事件）

---

## 🔌 I/O 契约（稳定且向前兼容）

### 输入（单点/流式）：
- `ts`: float（秒）
- `price`: float（中间价/成交价）
- `z_ofi`: float、`z_cvd`: float（裁剪到 [-5,5]）
- 可选：`fusion_score`: float、`consistency`: float
- 状态：`warmup`: bool、`lag_sec`: float（与价格时间差）

### 输出（事件或空）：
```json
{
  "ts": 0.0,
  "type": "bull_div" | "bear_div" | "hidden_bull" | "hidden_bear" | "ofi_cvd_conflict",
  "score": 0.0,                        // 0-100
  "channels": ["price_ofi","price_cvd","price_fusion"],
  "lookback": {"swing_L": 20, "ema": 5},
  "pivots": {"price": {"A": ..., "B": ...}, "ofi": {...}, "cvd": {...}},
  "reason_codes": ["strict_both_agree","cooldown","low_vol_filter", ...],
  "debug": {"z_ofi": ..., "z_cvd": ..., "fusion": ..., "consistency": ...}
}
```

---

## 🧠 背离定义（统一"枢轴"口径）

### 枢轴检测
- 使用枢轴-点检测价格与指标的高/低点（Pivot）
- 枢轴定义：swing_L 左右各 L 根内，极值唯一（如 L=20）
- 可选信号平滑：对 z_* 与 fusion_score 用 EMA(k) 去噪
- 三条通道：price↔z_ofi、price↔z_cvd、price↔fusion_score(可配)

### A. 常规背离（反转）
- **看涨（Bull Regular）**：price 创 更低低点(LL)，指标创 更高低点(HL)
- **看跌（Bear Regular）**：price 创 更高高点(HH)，指标创 更低高点(LH)

### B. 隐藏背离（趋势延续）
- **Hidden Bull**：price HL 与 指标 LL
- **Hidden Bear**：price LH 与 指标 HL

### C. OFI-CVD 不一致（冲突提示）
- 强度阈：|z_ofi| ≥ z_hi 且 |z_cvd| ≥ z_mid，但符号相反
- 典型：z_ofi ≥ +2 且 z_cvd ≤ −1，或反之
- 仅作风险提示，不直接给方向入场

---

## 🧮 评分与门限（让强弱可比较）

### 原子证据分（0-1）：
- `pivot_validity`：枢轴间距≥min_separation、形态清晰度
- `z_magnitude`：相关通道的绝对强度（归一到 0-1）
- `direction_agree`：若三通道中 ≥2 条一致，则加权提升
- `consistency_bonus`：若（已启用）consistency ≥ c_min，加分
- `vol_ok`：ATR/成交量门限通过（过滤横盘/冷清期）

### 事件分：
```
score = 100 * ( 0.35*z_mag + 0.25*pivot + 0.25*agree + 0.15*consis ) * vol_ok
```

### 强度分级：
- `strong` (≥70) / `normal` (50~70) / `weak` (40~50)（<40 不报）

---

## 🚦 触发与去噪

- **冷却**：事件后 cooldown_secs 内不重复同向
- **聚类合并**：cluster_window 内多次触发 → 仅保留最高分
- **最小枢轴间距**：防止"抖动"形成伪枢轴
- **暖启动**：样本 < warmup_min → 不触发
- **滞后**：lag_sec > max_lag → 不触发并 reason_codes+=['lag_exceeded']

---

## 🛠️ 实现建议（解耦 + 易测试）

### 新增独立文件 `ofi_cvd_divergence.py`：

```python
@dataclass
class DivergenceConfig:
    swing_L: int = 20
    ema_k: int = 5
    z_hi: float = 2.0
    z_mid: float = 1.0
    min_separation: int = 10
    cooldown_secs: float = 3.0
    warmup_min: int = 200
    max_lag: float = 0.300
    use_fusion: bool = True
    cons_min: float = 0.3

class DivergenceDetector:
    def update(self, ts, price, z_ofi, z_cvd, fusion_score=None,
               consistency=None, warmup=False, lag_sec=0.0) -> Optional[Dict]:
        ...
```

### Prometheus 指标（与 1.2.11 统一风格）：
- `divergence_events_total{type,channel}`（Counter）
- `divergence_score{type}`（Gauge）
- `divergence_last_ts{type}`（Gauge）
- `divergence_suppressed_total{reason}`（Counter）

### Grafana 面板：
- 背离事件脉冲图（按 type 着色）
- 最近事件表（score、channel、reason）
- 7/30 天事件统计 & 成功率（回测标签）

---

## ✅ 验证标准（DoD 量化）

### 功能正确：
- 四类背离 + 冲突提示均可触发
- 事件含 pivots/score/reasons

### 稳定性：
- 空数据、warmup、lag 超阈全部安全降级

### 性能：
- p95(update) < 3ms（10 万次微基准）

### 单测通过：
- ≥ 12 条用例（见下）

### 回测有效：
- 方向性背离（bull_div/bear_div）→ accuracy@Nbars ≥ 55%（N=10/20 可配置）
- 事件收益分布较随机基线有统计优势（MWU 或 KS 检验 p<0.05）
- 可观测：Prometheus 指标完整，Grafana 仪表盘可视

---

## 🧪 单元测试清单（最小充分）

- [ ] 枢轴正确性：HH/LL/LH/HL 识别
- [ ] 常规/隐藏背离的触发与抑制
- [ ] 冲突提示（z_ofi 与 z_cvd 反号）仅出提示不出方向
- [ ] 暖启动与滞后过滤
- [ ] 冷却与聚类合并
- [ ] 评分单调性：在固定形态下，|z| 与 consistency 增加 → score 不下降
- [ ] 边界：同价多枢轴、低波动/低量期不触发
- [ ] 性能微基准：p50/p95 统计并断言

---

## 🔁 回测方法（可复用到 1.3）

### 事件标注：
背离出现后 N 根（N∈{10,20}）方向收益是否为正（含成本/滑点）

### 指标：
accuracy@N、avg_return@N、IR、分位收益、胜率随 score 分桶曲线

### 对照：
- 随机基线（时间随机打点数量匹配）
- 单通道（price-OFI / price-CVD） vs 三通道融合开启

### 报告：
CSV + 可视化（事件散点叠加价格；score-收益回归线）

---

## ⚠️ 注意事项

1. 低流动/横盘期，波动与成交量门限先行过滤
2. 出现跳空/极端尖峰，优先确保枢轴稳定与**z 裁剪**
3. 与 1.2.11 的职责分离：背离只消费指标，不反向写入融合器状态

---

## 🔗 代码改动建议

- `src/ofi_cvd_divergence.py`（新）- 独立的背离检测模块
- `tests/test_ofi_cvd_divergence.py`（新）- 独立的测试文件
- `fusion_prometheus_exporter.py`：增加背离相关的 Counter/Gauge
- `grafana/`：新增背离面板 JSON（事件表 + 脉冲 + 成功率）

---

## 📈 质量评分（完成后填写）

- **算法准确性**: __/10
- **回测效果**: __/10
- **总体评分**: __/10

---

## 📝 执行记录（留空待填）

### 遇到的问题
___（记录遇到的问题）___

### 解决方案
___（记录解决方案）___

### 经验教训
___（记录经验教训）___

---

## 🔄 任务状态更新

- **开始时间**: ___
- **完成时间**: ___
- **是否可以继续**: ⬜ 是 / ⬜ 否

---

**创建时间**: 2025-10-17  
**最后更新**: 2025-10-20

