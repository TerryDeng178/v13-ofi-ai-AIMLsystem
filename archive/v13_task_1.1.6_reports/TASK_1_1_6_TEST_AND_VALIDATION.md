# Task 1.1.6 — 测试与验证（异步日志 + 轮转保留 + 30–60min 稳态）

## 🎯 Goal
在 **不改架构、不加依赖** 的前提下，完成：
1) **非阻塞日志**（`QueueHandler/QueueListener`），确保 WS 消费不被打印/写盘拖慢；  
2) **日志轮转与保留**（时间或大小轮转 + ≥7 个备份，自动清理过期）；  
3) **10s 周期 metrics 刷新**（覆盖写）；  
4) **30–60 分钟 soak test**，输出延迟分位、连续性、吞吐与队列水位等指标的**验收报告**。

---

## 🗂 Allowed files
- `v13_ofi_ai_system/src/binance_websocket_client.py`
- `v13_ofi_ai_system/src/utils/async_logging.py`  ← 新增一个工具模块（仅标准库）

> 说明：只允许新增这 1 个工具文件；其他文件不得改动。

---

## ✂️ Change Budget
- `files_changed_max: 2`  
- `total_changed_lines_max: 180`

---

## ✅ Acceptance Criteria（必须全部满足）
1) **非阻塞**：日志采用 `QueueHandler` + `QueueListener`；WS 消费主循环不直接写磁盘/格式化大字符串。  
   - 指标：`log_queue_depth_p95 <= 0`、`log_drops == 0`、`ingest_backlog_max == 0`。
2) **轮转与保留**：提供两套模式（启动参数切换）  
   - 时间轮转：`--rotate=interval --rotate-sec=60 --backups=7`（测试时观察切片）  
   - 大小轮转：`--rotate=size --max-bytes=5_000_000 --backups=7`（常规跑）  
   - 验证测试模式下切出 ≥2 个切片，并保证保留上限生效（不会无限增长）。
3) **metrics 周期刷新**：`metrics.json` 每 **10s** 覆盖写一次，字段至少包含：
   ```json
   {
     "window_sec": 10,
     "runtime_sec": ...,
     "total_messages": ...,
     "recv_rate": ...,
     "latency_ms": {"p50": ..., "p95": ..., "p99": ...},
     "continuity": {"breaks": 0, "resyncs": 0, "reconnects": 0},
     "batch_span": {"p95": ..., "max": ...},
     "log_queue": {"depth_p95": ..., "depth_max": ..., "drops": 0}
   }
   ```
4) **连续性与吞吐**（60min 可选，至少 30min）：  
   - **连续性合格**：采用 `pu == last_u` 判定连续；`breaks == 0`、`resyncs == 0`（如 >0，必须附触发日志与恢复时间）。  
   - **分位单调**：`p99 ≥ p95 ≥ p50 ≥ 0`。  
   - **吞吐不退化**：`recv_rate ≥ 1.0 /s` 或 ≥上版 90%。
5) **最终报告**：生成 `reports/Task_1_1_6_validation.json`，包含：  
   - 全程 `p50/p95/p99` 曲线点（每 1–2 分钟采样一次即可）；  
   - `breaks/resyncs/reconnects` 总计数与时间点；  
   - `batch_span_p95/max`；  
   - `log_queue_depth_p95/max` 与 `log_drops`；  
   - 日志切片文件清单与大小（证明轮转与保留）。

---

## 🧭 Plan first（提交前先给出计划）
1) **文件与函数**：在哪些函数插入队列日志、在哪里维护 `log_queue_depth`；指标采集点（`on_message` / `after_update` / `flush_metrics`）。  
2) **轮转策略**：`TimedRotatingFileHandler`（测试模式 60s）与 `RotatingFileHandler`（大小模式），参数切换方案。  
3) **指标实现**：队列深度统计（滑动窗口 `deque`）、`drops` 计数（队列 `put_nowait` 失败时 +1）、`ingest_backlog` 监测变量。  
4) **metrics 写盘**：10s 覆盖写，沿用现有 metrics 写盘函数或新增轻量方法。  
5) **报告生成**：在收尾时序列化汇总（或进程信号中断时也能 `flush`）。

---

## 🛠 实施要点（必须遵循）
- 仅用标准库：`logging.handlers.QueueHandler/QueueListener`、`queue.Queue(maxsize=...)`、`logging.handlers.{TimedRotatingFileHandler,RotinatingFileHandler}`。  
- **主线程/WS 线程只做 `logger.info/debug`**（入队）；**文件 IO 在 Listener 线程执行**。  
- **打印节流**：控制台摘要仍为 10s/次（固定前缀 `SUMMARY`），不要输出大表格。  
- **连续性规则**：首次对齐 `U ≤ X+1 ≤ u`；其后 **`pu == last_u`** 才应用，否则触发 **resync**。  
- **无依赖新增**：不得引入第三方库（如 `psutil`）。可选用 `tracemalloc` / `resource` 做轻量内存观测（非必需）。

---

## ▶️ 运行方式（示例）
**测试模式（时间轮转，每 60s 切片）**
```bash
python v13_ofi_ai_system/src/binance_websocket_client.py \
  --symbol ETHUSDT \
  --rotate interval --rotate-sec 60 --backups 7 \
  --print-interval 10 \
  --ndjson logs/ws_depth_ETHUSDT.ndjson.gz \
  --metrics logs/metrics.json \
  --run-minutes 30
```

**常规模式（大小轮转）**
```bash
python v13_ofi_ai_system/src/binance_websocket_client.py \
  --symbol ETHUSDT \
  --rotate size --max-bytes 5000000 --backups 7 \
  --print-interval 10 \
  --ndjson logs/ws_depth_ETHUSDT.ndjson.gz \
  --metrics logs/metrics.json \
  --run-minutes 60
```

---

## 🧪 验收脚本（可选，本地跑）
`tools/validate_task_1_1_6.py`
```python
# 不依赖第三方库
import json, time, pathlib

m = pathlib.Path("logs/metrics.json")
a = json.loads(m.read_text()); time.sleep(10); b = json.loads(m.read_text())

def ok(x):
    lat = x["latency_ms"]; cq = x["continuity"]; lq = x["log_queue"]
    return all([
        lat["p99"] >= lat["p95"] >= lat["p50"] >= 0,
        cq["breaks"] == 0 and cq["resyncs"] == 0,
        lq["depth_p95"] <= 0 and lq["drops"] == 0,
        x["recv_rate"] >= 1.0,
    ])

print("refresh:", a != b, "okA:", ok(a), "okB:", ok(b))
```

---

## 📦 提交产物（必须一次性交付）
1) **代码补丁**：仅 2 个文件的 diff（总改动 ≤180 行）；  
2) **运行命令**与 commit hash；  
3) **日志切片列表**：`logs/*.log`（至少 2 个不同时间段的文件名）；  
4) **metrics.json**：两次间隔 ≥10s 的快照；  
5) **NDJSON**：抽样 20 行纯 JSON；  
6) **最终报告**：`reports/Task_1_1_6_validation.json`（含各指标与切片证据）；  
7) **控制台摘要**：最后两条 `SUMMARY`。

---

## 🧿 Go / No-Go 门槛
- **Go（通过）**：`breaks==0 && resyncs==0`，分位单调，`recv_rate ≥ 1.0/s`（或 ≥上版 90%），`log_queue_depth_p95==0 && drops==0`，日志轮转实证（≥2 切片）。  
- **No-Go（不通过）**：任一硬指标不满足，或新增依赖/超出变更预算。

---

> 说明：本任务卡承接 **Task 1.1.5** 已完成的内容（NDJSON 字段完整、分位统计、正确的连续性口径、周期 metrics 输出），在此基础上进一步确保**可观测性与稳态表现**，并以**最小改动**完成工程化落地。
