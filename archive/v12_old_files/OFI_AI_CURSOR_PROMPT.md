# Cursor Prompt — OFI 策略 × AI 元模型（LSTM / Transformer / CNN / 集成）一键执行单

> 目标：在现有 **OFI/微观结构**策略上增加**AI 元模型**（仅做“出手过滤 & sizing”，不替代风控和执行），实现费用后 IR 提升、拒单率可控、端到端延迟达标。适配仓库：`eth_perp_ofi_sim*`（批量与实时版均可）。

---

## ✅ 总任务清单（按顺序执行）
- [ ] A. 创建特征流水线（100ms 微窗 × 最近 N 步）与标签（“三重障碍”meta-label）
- [ ] B. 训练 **TCN / 小型 Transformer / LSTM** 三套基线模型 + 简单集成（加权或 LR stacking）
- [ ] C. 线下评估（Purged Walk-Forward + Embargo），输出 **费用后 IR、分桶 IR、校准曲线**
- [ ] D. 在线推理接入（ONNX/TorchScript），作为**二层过滤 & sizing**，加 **EV≥2×成本** 硬门槛
- [ ] E. A/B/C 阈值 × （无模型 vs. 有模型）影子盘对比，**仅在正 IR 子桶放量**，灰度上线

---

## A) 数据与特征（src/features.py）
采样：100ms 微窗；序列长度：T ∈ [20, 100]（2–10s）。  
特征通道（示例）：OFI_z（1档或前3档加权）、OFI_roll、CVD_z、ΔCVD(近3)、spread_bps、depth_top5_sum、depth_imbalance、volume_spike、|price−VWAP|/σ_VWAP、session_flag、regime 分位、data_age_ms、e2e_budget_ms。

实现建议（伪代码）：
<code>
class FeatureWindow:
    def __init__(self, T=64, micro_ms=100):
        ...
    def update_from_event(self, e):  # ingest best/l2/trade
        ...
    def snapshot(self):
        # Return ndarray [T, C] normalized features; None if not ready.
        ...
</code>

---

## B) 标签（“三重障碍”，src/label.py）
- 目标（+1）：先到 0.8–1.0R；止损（0）：先到 1.1R；超时（0）：到期未触达。
- 严格事件驱动切片，不泄漏未来；类别不平衡用 class weights / focal loss(γ=1–2)。

伪代码：
<code>
def tri_barrier_label(quotes, t_entry, side, R_target=0.9, R_stop=1.1, horizon_ms=60000):
    ...
</code>

---

## C) 模型（TCN / 小型 Transformer / LSTM）与集成（src/models/*.py）
TCN：3–4 层 dilated conv（kernel=3, dilations={1,2,4,8}, channels=32–64），sigmoid 概率。  
Transformer：T=32–64, d=32–64, layers=2–3, heads=2–4。  
LSTM：1–2 层，hidden 32–64。  
集成：加权或 stacking（二层逻辑回归）。导出 ONNX/TorchScript 到 models/export/。

导出示例：
<code>
torch.onnx.export(model, dummy_input, "models/export/tcn.onnx",
                  input_names=["x"], output_names=["p"], opset_version=17)
</code>

---

## D) 训练与验证（train.py）
- Purged Walk-Forward CV + Embargo；AUC-PR、Brier、校准（Platt/Isotonic）  
- 策略指标：费用后 IR、胜率/盈亏比、拒单率、E2E 延迟达标率。  
- 分桶：sig_type × OFI_z分位 × spread分位 × depth分位 × session(是/否)。

示例命令：
<code>
python train.py --features data/features.parquet --labels data/labels.parquet \
  --model tcn --t 64 --c 24 --export models/export/tcn.onnx
</code>

---

## E) 在线推理接入（src/runtime_infer.py）— 二层过滤 & sizing
在线逻辑（伪代码）：
<code>
x = feat_window.snapshot()                 # [T, C]
P = model.predict(x)                       # meta-prob: reach target first
exp_reward_bps = estimate_reward_bps()     # |VWAP-价|/价 or 0.5×ATR/价
cost_bps = fee_bps + est_slip_bps
guards_ok = (spread_bps <= th_spread) and (data_age_ms <= th_age)

if (P > tau) and (exp_reward_bps >= 2 * cost_bps) and guards_ok:
    size = base_size * max(0.0, P - tau)
    allow_order(size)                      # 动量分批IOC；背离限价回踩（未成交即撤）
else:
    skip()
</code>

延迟目标（单次推理+预处理）：动量/清算簇 ≤ 5–15ms；收复 ≤ 15–30ms；回归 ≤ 30–50ms。

---

## F) 执行与风控（硬约束四件套）
- 流动性前置：点差 ≤ 阈 + 深度 ≥ rolling P50  
- 两段式收复：破位→下一根回内侧才允许反向  
- 预算→拒单：slip_bps ≤ min(max_slip, slip_frac × exp_reward_bps)  
- EV 硬门槛：exp_reward_bps ≥ 2×(手续费+滑点)；最小tick止损：SL ≥ 6×tick

---

## G) 评估与KPI（验收标准）
费用后 IR ≥ 0（成本 +50% 压力仍不为负）；  
胜率：背离 ≥ 30–45%，动量 ≥ 45–55%（盈亏比>1.2）；  
成本占比 ≤ 期望收益 × 40%；拒单率（严/均衡/松）：~80%/60%/40–60%；  
交接窗内成交占比 ≥ 60%（对背离类）。

---

## H) 实验矩阵（影子盘 ≥500 笔）
A（严）：ofi_z_min=2.3, cvd_z_min=1.6, thin_spread=1.4, max_slip=7, min_tick_sl=6, tau=0.70  
B（均衡）：ofi_z_min=2.1, cvd_z_min=1.4, thin_spread=1.6, max_slip=8, min_tick_sl=6, tau=0.65  
C（松）：ofi_z_min=1.9, cvd_z_min=1.3, thin_spread=1.6, max_slip=8, min_tick_sl=5, tau=0.60  
对照：无模型 vs. 有模型；输出分桶 IR、拒单率。

---

## I) 参考代码骨架（tcn.py & runtime_infer.py 摘要）
<code>
class TCN(nn.Module): ...
torch.onnx.export(...)
class MetaFilter:
    def allow(...):  # 返回 (allow, p, size)
        ...
</code>

---

## J) README 补充（回滚与灰度）
日内回撤>6% 或当日 IR<0 → 回滚至上一 tag；  
影子盘 → 小额实盘 → 仅在正 IR 子桶放量（每次 +20%）；  
监控：P 分布/校准、成本占比、E2E P95 延迟。

---

## K) 运行与产物
- train.py：输出 models/export/*.onnx/.pt  
- ab_shadow.py：reports/metrics.json、buckets.csv、calibration.png  
- runtime_infer.py：在线筛单 + sizing（仅提示允许下单，不下单）。

---

**一句话原则**：AI 只是**二层“出手过滤器 / 仓位调节器”**，风控与执行纪律永远优先；先把**费用后 IR + 延迟**跑通，再灰度。