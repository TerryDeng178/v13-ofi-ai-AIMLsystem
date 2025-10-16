
# Cursor Prompt — OFI/CVD v3 系统评估→优化 一次性执行单

> 目标：把策略拉回“目标工况”（8–20 笔/日/标的、费用后 IR≥0、拒单率 5–25%）。
> 本执行单包含：参数落地（均衡版 B）、硬约束四件套、回测指标、A/B/C 影子盘与分桶归因。
> 适用仓库：`ofi_cvd_framework`

---

## ✅ 总任务清单（按顺序执行）
- [ ] A. 应用 **均衡版（B）** 参数
- [ ] B. 落地 **硬约束四件套**（if-guard，未通过直接拒单）
- [ ] C. 扩展回测指标（Sharpe/MDD/成本占比）
- [ ] D. 运行 **A/B/C 影子盘 ≥500 笔** 并输出 **分桶 IR 报表**
- [ ] E. 只保留 **正 IR 子桶**，在该子集上逐步放大 `size_max_usd`

---

## A) 参数（均衡版 B）— 粘贴到 `config/params.yaml`
```yaml
features:
  z_window: 1200
  ofi_window_seconds: 2
  vwap_window_seconds: 1800

signals:
  momentum:
    ofi_z_min: 2.1
    cvd_z_min: 1.4
    min_ret: 0.0
    thin_book_spread_bps_max: 1.4
    adaptive_thresholds: true
    quantile_window: 5400
    momentum_quantile_hi: 0.85
  divergence:
    cvd_z_max: 0.0
    ofi_z_max: 0.0
    reclaim_bars: 1              # 破位后一根回内侧
    divergence_quantile_hi: 0.75
  sizing:
    k_ofi: 0.10
    size_max_usd: 50000

risk:
  atr_stop_lo: 1.1
  atr_stop_hi: 1.6
  min_tick_sl_mult: 6
  time_exit_seconds_min: 120
  time_exit_seconds_max: 900
  slip_bps_budget_frac: 0.25
  fee_bps: 2.0

execution:
  slippage_budget_check: true
  max_slippage_bps: 7.0
  reject_on_budget_exceeded: true
  session_window_minutes: 15     # 背离仅在会话交接窗 ±15min
```

---

## B) 硬约束四件套（必须是 if-guard）

### B1. 流动性前置（`src/strategy.py`）
```python
spread_bps = (row["ask1"] - row["bid1"]) / row["price"] * 1e4
depth_now = row["bid1_size"] + row["ask1_size"]  # 可替换为 top-5 合计
depth_med = (df["bid1_size"] + df["ask1_size"]).rolling(300, min_periods=60).median().fillna(method="bfill").iloc[i]
if not (spread_bps <= params["signals"]["momentum"]["thin_book_spread_bps_max"] and depth_now >= depth_med):
    continue
```

### B2. 两段式收复（`src/signals.py`）
```python
hh = out["price"].rolling(60, min_periods=30).max()
ll = out["price"].rolling(60, min_periods=30).min()
new_high = out["price"] >= hh
new_low  = out["price"] <= ll

reclaim_high = new_high & (out["price"].shift(-params["signals"]["divergence"].get("reclaim_bars",1)) < hh)
reclaim_low  = new_low  & (out["price"].shift(-params["signals"]["divergence"].get("reclaim_bars",1)) > ll)

div_short = reclaim_high & ((out["cvd_z"] <= 0.0) | (out["ofi_z"] <= 0.0))
div_long  = reclaim_low  & ((out["cvd_z"] <= 0.0) | (out["ofi_z"] <= 0.0))

out.loc[div_long,  "sig_type"] = "divergence"
out.loc[div_long,  "sig_side"] = 1
out.loc[div_short, "sig_type"] = "divergence"
out.loc[div_short, "sig_side"] = -1
```

### B3. 滑点预算→拒单（`src/strategy.py`）
```python
exp_reward = max(abs(row["vwap"] - row["price"]), 0.5 * row["atr"])
entry_px, fee, slip = broker.simulate_fill(side, qty_usd, row["price"], row["atr"], exp_reward)
budget_bps = min(params["execution"]["max_slippage_bps"],
                 params["risk"]["slip_bps_budget_frac"] * (exp_reward / row["price"]) * 1e4)
if slip > budget_bps:
    continue
```

### B4. 最小 tick 止损（`src/risk.py`）
```python
def compute_levels(row, params):
    atr = float(row["atr"]); price = float(row["price"])
    lo = params["risk"]["atr_stop_lo"]
    tick = max(float(row.get("ask1", price) - row.get("bid1", price)), 1e-2)
    min_sl = max(params["risk"].get("min_tick_sl_mult", 6) * tick, 1e-2)
    atr_sl = max(lo * atr, min_sl)
    if row["sig_side"] > 0:
        sl = price - atr_sl; tp = max(row.get("vwap", price), price + 0.5*atr)
    else:
        sl = price + atr_sl; tp = min(row.get("vwap", price), price - 0.5*atr)
    return sl, tp
```

> 可选：动量单要求 `ΔOI>0` + 现货轻确认；背离单仅在**会话交接窗 ±session_window_minutes**允许（在 `strategy.py` 判断时间窗）。

---

## C) 回测指标（`src/backtest.py`）
```python
import numpy as np, json, os, pandas as pd
# 已有 summary 后追加：Sharpe/MDD/成本占比
if len(trades) > 1:
    rets = trades["pnl"] / trades["qty_usd"].replace(0, np.nan).abs()
    rets = rets.dropna().values
    sharpe = float((np.mean(rets) / (np.std(rets)+1e-9)) * np.sqrt(365*24*60*60)) if len(rets)>1 else 0.0
else:
    sharpe = 0.0
curve = trades["pnl"].cumsum() if len(trades) else pd.Series(dtype=float)
peak = np.maximum.accumulate(curve.fillna(0)) if len(curve) else curve
mdd = float((curve - peak).min()) if len(curve) else 0.0
cost = float(abs(trades.get("pnl",0)).sum() - trades["pnl"].sum()) if len(trades) else 0.0  # 近似
summary.update({"sharpe_approx": sharpe, "mdd": mdd, "cost_est": cost})
outdir = "examples/out"; os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
```

---

## D) A/B/C 影子盘与分桶 IR
- 组 A（严）：`ofi_z_min=2.4, cvd_z_min=1.6, thin_spread=1.2, min_tick_sl=7, max_slip=6`
- 组 **B（均衡）**：`ofi_z_min=2.1, cvd_z_min=1.4, thin_spread=1.4, min_tick_sl=6, max_slip=7`
- 组 C（松）：`ofi_z_min=1.8, cvd_z_min=1.2, thin_spread=1.6, min_tick_sl=5, max_slip=8`
- 统一：四件套全开；背离仅在会话交接窗 ±15 分钟。

**分桶维度：** `sig_type × OFI_z分位 × spread分位 × depth分位 × session(是/否)`  
**准入：** 仅保留净 IR>0 的子桶；负 IR 桶禁用。

---

## KPI（验收标准）
- 费用后 **IR≥0**；成本 +50% 压力回放后仍不为负
- 背离胜率 **≥30–45%**；动量胜率 **≥45–55%**（且盈亏比>1.2）
- 平均（手续费+滑点） **≤ 期望收益的 40%**
- 拒单率 **5–25%**
- 会话交接窗内成交占比 **≥60%**

---

## 回滚与灰度
- 分支：`upgrade/ofi-cvd-v3`；
- 条件：日内回撤>6% 或当日 IR<0 → 回滚到上一 tag；
- 先影子盘，正 IR 后 **仅在正 IR 子桶**上放大 `size_max_usd`（每次+20%）。
