# CVD 组件问题与修复方案（交付给 CURSOR 的实现指南）

> 目标：把当前 **5/8** 的测试通过率，提升到 **8/8（100%）**。  
> 范畴：仅涉及 **逐笔成交流（Binance aggTrade）→ CVD 计算器**链路与分析验证，不改动下单/策略层。

---

## 1) 症状与结论（TL;DR）

- **根因不在计算器本体**，而在**上游使用 `event_time_ms` 作为“唯一ID/排序键”**：  
  - 同毫秒多笔 → **大量“重复/无序”**；  
  - 造成 **CVD 连续性校验几乎全挂**、**Z 分布重尾**。
- 计算器本身的 **Z-score 设计（用 CVD 水平值标准化）** 会放大重尾，是**P1 需要优化**但非主因。
- 只要：
  1) **改用 `aggTradeId (a)` 作为唯一键** + **2s 水位线重排**；  
  2) **一致性检查改为“增量守恒 + 排序基于 (E, a)”**；  
  3) **Z-score 改为基于 ΔCVD 的稳健标准化**；  
  即可现实达成 **8/8 通过**。

---

## 2) 证据要点（来自最新报告/图）

- **Event ID 健康**：  
  - Duplicate（同 ID）≈ **3354**，Backward（倒序）≈ **43**，`>10s` 大跳 **0**（见 `event_id_diff.png`）。  
- **Interarrival（消息间隔）**：  
  - `P95 ≈ 2698ms`，`P99 ≈ 3435ms`，`Max ≈ 8388ms`（见 `interarrival_hist.png`）。  
- **Z 分布（现实现）**：  
  - `median(|Z|) ≈ 1.49`、`IQR ≈ 2.90`、`P(|Z|>2) ≈ 25.6%`、`P(|Z|>3) ≈ 6.3%`。  
- **CVD 连续性**：  
  - 逐笔守恒错误 **144/145** 次（几乎全红）。

> 直观结论：**唯一键/排序设计缺陷** + **Z 标准化对象选错（水平而非增量）**。

---

## 3) 修复路线（P0 → P1）

### P0（必做｜上游流 & 验证）
1) **唯一键切换 & 乱序重排**
   - 采集落盘字段（最小集合）：  
     `{"s":symbol, "a":agg_trade_id, "p":price, "q":qty, "m":isBuyerMaker, "E":event_time_ms, "T":trade_time_ms}`
   - **以 `a` 为唯一键去重**：维护 `last_a`，`a <= last_a` 直接丢弃并计数；  
   - **2s 水位线（watermark）乱序缓冲**：按 `(E, a)` 排序后写盘，等待上限 2000ms。

2) **一致性检查改造**
   - 统一定义增量：`Δcvd = (-1 if m else +1) * float(q)`；  
   - **逐笔守恒**：`cvd_t == cvd_{t-1} + Δcvd_t`；  
   - **首尾守恒**：`cvd_last - cvd_first == ΣΔcvd`；  
   - 报表固定输出 **Event ID 差值直方图** 与 **Interarrival 直方图**。

3) **验收阈值（P0 必过）**
   - `continuity_mismatch == 0`  
   - `agg_dup_rate == 0`、`backward_rate ≤ 0.5%`  
   - `p99_interarrival ≤ 5s` 且 `gaps_over_10s == 0`

---

### P1（应做｜计算器 Z-score 稳健化）
4) **Z-score 改到“增量域”**
   - 定义：`z_t = Δcvd_t / scale_t`；  
   - `scale_t` 用 **EWMA(|Δcvd|)**（或 **MAD(Δcvd)**）作为稳健尺度；  
   - **winsorize**：`|z| > 8` 截断到 8；  
   - **按“笔数”滚动**（非时间）：例如 half-life ≈ **300 笔**（折合 ~5min，视活跃度调整）；  
   - **暖启动/冻结**：有效样本 < 50 笔或 `scale≈0` ⇒ 暂不产出 z；  
   - **stale 抑制**：若与上笔 `event_time_ms` 间隔 > 5s，本笔**只更新状态不产出 z**（避免空窗后首笔虚高）。

5) **验收阈值（P1 过渡目标）**
   - `median(|Z|) ≤ 1.0`  
   - `P(|Z|>2) ≤ 10%`、`P(|Z|>3) ≤ 2%`  
   > 过渡目标通过后，再收紧到正式门槛（如 8% / 1%）。

---

## 4) 代码补丁片段（可直接交给 CURSOR）

### 4.1 采集层：去重 + 2s 水位线重排
```python
# stream/binance_trade_stream.py
import heapq, time, os
WATERMARK_MS = int(os.getenv("WATERMARK_MS", "2000"))

last_a = -1
buf = []  # heap of (E, a, msg)
metrics = {"agg_backward_count": 0, "late_write_count": 0}

def on_msg(msg):
    # msg fields: s, a, p, q, m, E, T
    a = int(msg["a"]); E = int(msg["E"])
    global last_a
    if a <= last_a:                 # duplicate or backward
        metrics["agg_backward_count"] += 1
        return
    heapq.heappush(buf, (E, a, msg))
    _flush_until(int(time.time()*1000) - WATERMARK_MS)

def _flush_until(wm):
    global last_a
    while buf and buf[0][0] <= wm:
        _, a0, m0 = heapq.heappop(buf)
        write_ndjson(m0)            # your existing writer
        last_a = a0

def on_close():
    # flush remainder
    _flush_until(10**18)
```

### 4.2 分析侧：增量守恒 + (E,a) 排序 & 指标
```python
# analysis/cvd_checks.py
from dataclasses import dataclass

@dataclass
class Row:
    E:int; a:int; q:float; m:bool

def cvd_and_checks(rows):
    rows = sorted(rows, key=lambda r:(r.E, r.a))
    cvd, mismatches = 0.0, 0
    deltas = []
    for r in rows:
        delta = (-1 if r.m else 1) * float(r.q)
        expected = cvd + delta
        # if row already carries cvd_t, compare with expected
        # if abs(r.cvd - expected) > 1e-9: mismatches += 1
        cvd = expected
        deltas.append(delta)
    conservation_err = abs(cvd - sum(deltas))
    return cvd, mismatches, conservation_err

def event_id_health(rows):
    dup = back = 0
    diffs = []
    for i in range(1, len(rows)):
        d = rows[i].a - rows[i-1].a
        dup += (d == 0); back += (d < 0); diffs.append(d)
    return {"dup": dup, "back": back, "diffs": diffs}
```

### 4.3 计算器：Z-score 基于 ΔCVD + EWMA(|Δ|)
```python
# calc/real_cvd_calculator.py  (在你的 RealCVDCalculator 基础上最小改)
import math

class RealCVDCalculator:
    __slots__ = ("cvd","ewma_abs","trades","alpha","winsor","freeze_min","last_E")

    def __init__(self, half_life_trades=300, winsor=8.0, freeze_min=50):
        self.cvd = 0.0
        self.ewma_abs = 0.0
        self.trades = 0
        self.alpha = 1 - math.exp(math.log(0.5)/max(1, half_life_trades))
        self.winsor = winsor
        self.freeze_min = freeze_min
        self.last_E = None

    def update_with_trade(self, qty: float, is_buyer_maker: bool, event_ms: int | None = None):
        delta = (-qty if is_buyer_maker else +qty)
        self.cvd += delta
        self.trades += 1
        self.ewma_abs = self.alpha*abs(delta) + (1-self.alpha)*self.ewma_abs

        # stale-window freeze
        z = None
        if self.last_E is not None and event_ms is not None and (event_ms - self.last_E) > 5000:
            z = None
        elif self.trades >= self.freeze_min and self.ewma_abs > 1e-9:
            z = delta / self.ewma_abs
            z = max(min(z, self.winsor), -self.winsor)

        self.last_E = event_ms if event_ms is not None else self.last_E
        return self.cvd, delta, z  # 暴露 delta/scale 给监控也可
```

---

## 5) 报告模板增补（固定到流水线）

- **Event ID 健康**  
  - 指标：`agg_dup_rate`、`backward_rate`、`late_write_count`  
  - 图：`event_id_diff.png`（Δa 直方图，标注 dup/back）  
- **Interarrival 健康**  
  - 指标：`p95_interarrival`、`p99_interarrival`、`max_interarrival`、`gaps_over_10s`  
  - 图：`interarrival_hist.png`（P95/P99 虚线）  
- **CVD 连续性**  
  - 指标：`continuity_mismatch`（逐笔）、`conservation_error`（首尾）  
- **Z 质量**  
  - 指标：`median_abs_z`、`iqr_z`、`p_gt2`、`p_gt3`、`z_freeze_count`  
  - 图：`hist_z.png`、`z_timeseries.png`  
- **延迟**（保持现有箱线）：`latency_box.png`

---

## 6) 测试流程与放行条件

1) **快测（30–60min）**  
   - 必须满足：  
     - `continuity_mismatch == 0`、`agg_dup_rate == 0`、`backward_rate ≤ 0.5%`  
     - `p99_interarrival ≤ 5s`、`gaps_over_10s == 0`  
     - `median(|Z|) ≤ 1.0`、`P(|Z|>2) ≤ 10%`、`P(|Z|>3) ≤ 2%`
2) **金测（24h）**  
   - 以上全部维持，产出 **Before / After** 对比表与全套图；  
   - 通过后进入 **融合/触发 A/B**（策略层）。

---

## 7) 风险与回滚

- 若个别时段/品种交易过稀疏，**上涨 `half_life_trades`** 或切到 **MAD 尺度**；  
- 若 `p99_interarrival` 偶有超 5s，但 `>10s`为 0：  
  - 计算器继续 **stale-freeze**，不产出 Z；  
  - 报告里记录 `gap_events`，不视为阻断项。  
- 回滚点清晰：  
  - 采集层可回退到“无水位线但保持 a 去重”；  
  - 计算器保留旧 `level-Z` 路径开关（配置 `z_mode`）。

---

### 最后一页（给 CURSOR 的执行摘要）

- **P0**：  
  - 采集：改唯一键→`a`；加 2s 水位线；落盘字段齐全；去重计数；  
  - 分析：排序 `(E,a)`；逐笔/首尾守恒校验；两张健康图；  
  - 阈值：连续性=0、dup=0、back≤0.5%、p99_gap≤5s、>10s=0。  
- **P1**：  
  - 计算器：`z = Δcvd / EWMA(|Δ|)`；winsor=8；half-life≈300 笔；freeze≥50 笔；stale>5s 不产 z；  
  - 阈值：`median|Z|≤1.0`、`P(|Z|>2)≤10%`、`P(|Z|>3)≤2%`。  
- **预期**：上述合入后，**8/8 指标通过**，可进入策略层实验。
