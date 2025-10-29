你是资深量化平台测试工程师，目标是对“core_algo.py 驱动的信号流水线（Collector→OFI/CVD→Fusion/Divergence→StrategyMode→PaperTrader）”进行影子交易巡检，并根据硬性阈值给出 Go/No-Go 结论。

【环境与约束】
- 仅使用 Python 标准库；不要新增第三方依赖（如 numpy/pandas）。
- 不修改核心业务代码与配置；只在 repo 里新增 `tools/` 下的测试脚本。
- 读取现有输出：`runtime/ready/signal/<SYMBOL>/signals_*.jsonl` 与 `artifacts/gate_stats.jsonl`。
- 若没有可读数据，必须以 “BLOCKED: <原因>” 明确退出（进程码≠0），不要伪造数据。

【输入数据假定的 JSONL 字段（若缺失需健壮处理）】
- `ts_ms:int`, `symbol:str`, `score:float`, `z_ofi:float|None`, `z_cvd:float|None`,
  `regime:str`, `div_type:str|None`（e.g. bullish/bearish/None）, `confirm:bool`,
  `gating:bool`, `guard_reason:str|None`.

【需要你创建并运行的 4 个脚本】
把下面 4 个脚本放到 `tools/` 目录，并确保可直接 `python tools/<script>.py` 运行。所有脚本均打印关键结果到 stdout。

1) `tools/z_healthcheck.py`
   作用：读取最近 ~60 分钟 `signals_*.jsonl`，计算：
   - `P(|z_ofi|>2)%`、`P(|z_cvd|>2)%`
   - `weak_ratio%`（1.0≤|score|<1.8）、`strong_ratio%`（|score|≥1.8）
   - `confirm_ratio%`
   - 校验数值是否为有限数；统计缺失比例
   要求：实现纯 Python 百分位函数（不依赖三方库）。

2) `tools/signal_consistency.py`
   作用：评估信号一致性与冲突：
   - `divergence_vs_fusion_conflict%`：当 `div_type=='bullish'` 但 `score<0` 或 `div_type=='bearish'` 且 `score>0` 的占比
   - `strong_signal_5m_directional_accuracy%`（若有价格字段则用价格方向；否则跳过并标记 N/A）
   - `confirm_after_threshold_rate%`：达到阈值(|score|≥1.0)后最终 `confirm==True` 的占比（仅非 Gating）
   - 结果按最近 ~60 分钟聚合

3) `tools/storage_liveness.py`
   作用：存储健康巡检
   - 检查最近 10 分钟内，每分钟是否都有新的 `signals_*.jsonl` 分片（分钟维度就地换名到 ready）
   - 检查 `artifacts/gate_stats.jsonl` 是否 ≤60s 有一条心跳
   - 如发现 `spool/*.part` 长时间滞留（>90s），打印告警
   - 输出：`OK/ALERT` 级别摘要

4) `tools/latency_and_queue.py`
   作用：粗测滞后与队列健康（无需连接进程）
   - 事件滞后(`event_lag_ms`)：以 `now()-ts_ms/1000` 近 60 分钟 p50/p95
   - 从控制台或 `gate_stats` 心跳中提取 `JsonlSink` 的 `qsize/open/dropped`（若日志不可读则跳过）
   - 输出：`lag_p50_ms/lag_p95_ms`，以及最近观测到的 `dropped` 值（理应为 0）

【统一的 Go/No-Go 判定脚本】
创建 `tools/shadow_go_nogo.py`，依次调用上述 4 个脚本（可通过 import 调用其函数，或子进程方式执行并解析其 stdout）。
按以下硬阈值打分并在末尾打印一行：
`DECISION: GO` 或 `DECISION: NO-GO`，同时给出各项不达标的原因清单。

【硬性验收阈值（全部达标才 GO）】
数据质量：
- 3% ≤ `P(|z_ofi|>2)` ≤ 12%
- 3% ≤ `P(|z_cvd|>2)` ≤ 12%
- `strong_ratio%` ∈ [0.8%, 3.5%]
- `confirm_ratio%` > 0%
一致性：
- `divergence_vs_fusion_conflict%` < 2%
- （可选）若能计算：`strong_signal_5m_directional_accuracy%` ≥ 52%
性能/稳定性：
- `lag_p95_ms` ≤ 120
- `JsonlSink dropped` == 0（或短时波动后回落为 0）
存储/可观测性：
- 近 10 分钟每分钟都有 `signals_*.jsonl` 新分片
- `gate_stats.jsonl` ≤60s 有心跳
阻断条件（任一触发即 NO-GO）：
- 找不到 `runtime/ready/signal/*/signals_*.jsonl`
- 解析到 NaN/Inf
- 分片长时间仅在 `spool/*.part` 而非 `ready/`

【实现细节要求】
- 提供一个纯 Python 百分位函数 `percentile(values, p)`（排序后线性插值）。
- 统计时用“最近 N 分钟”的文件（例如取最近 60 分钟/最多 120 个分片），避免超大历史拖慢。
- 所有脚本遇到异常需打印 `ERROR:` 前缀并以非零码退出。
- 在 `shadow_go_nogo.py` 末尾打印一个 YAML 风格摘要，字段包含：
  - `z_health: {p_abs_gt2_ofi, p_abs_gt2_cvd, weak_ratio, strong_ratio, confirm_ratio}`
  - `consistency: {div_vs_fusion_conflict, strong_5m_acc or 'N/A'}`
  - `latency: {lag_p50_ms, lag_p95_ms}`
  - `storage: {minutes_covered, ready_rotation_ok, gate_stats_heartbeat_ok}`
  - `decision: GO/NO-GO`
并把同样的摘要写入 `artifacts/shadow_summary.yaml`（若目录不存在则创建）。

【完成后要做的事】
1) 生成以上 5 个脚本到 `tools/`，并给出关键函数的简短注释。
2) 实际运行它们（读取当前 repo 的真实输出），在终端展示各脚本结果。
3) 最后运行 `python tools/shadow_go_nogo.py`，打印汇总与 `DECISION: <...>`。
4) 若被阻断（没数据等），明确打印 `BLOCKED:` 并给出诊断建议（例如“先确保 V13_SINK=jsonl 且已运行 ≥10 分钟”）。

请严格按以上要求实现与执行。
