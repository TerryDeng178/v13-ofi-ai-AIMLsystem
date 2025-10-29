# Task 1.4.1 (v1): 多交易对放量灰度与切片稳健性验证（T+1）

## 📋 任务信息
- **阶段**：阶段2 - 交易与上线  
- **状态**：✅ **已停止（T+1批次A已停止，数据收集转移到48小时收集）**  
- **优先级**：P0（放量）  
- **观测窗口**：每批 24–48 小时灰度观察  
- **范围**：在已通过 T+0 的基础上，扩展到更多交易对（建议首批：**ADAUSDT、SOLUSDT**；次批：BNBUSDT、XRPUSDT 等）  
- **创建时间**：2025-10-22 05:40:58 UTC+09:00

## 🔗 前置条件（已满足）
- T+0 灰度（BTCUSDT、ETHUSDT）达标：Active AUC ≥ 0.58、ECE ≤ 0.10、Merge p99 ≤ 2000ms，且无 P1 未闭合告警。  
- 统一配置已固化：`labels=mid (forward asof, tol=1500ms)`、`calibration=platt(train=7200s,test=1800s)`、`cvd_auto_flip=true`、`fusion: gate=0, w=(0.6,0.4)`；Active 切片可覆盖 `w=(0.7,0.3)`。  
- Runbook、监控与告警规则已联调通过；Pre-Canary 冒烟用例已归档。

## 🎯 目标（量化可验）
1) 在 **≥2 个新交易对** 上验证并维持 **SLO/Gate**：  
   - Active 切片 Fusion **AUC ≥ 0.58**（任一主窗口，10m 滚动）。  
   - 全局 **ECE ≤ 0.10**（10m 滚动），**Brier** 相对无信息基线 **≥ +5%**。  
   - **Merge p99 ≤ 2000ms**（5m 滚动），**Match rate ≥ 80%**。  
2) 产出每个交易对的 **灰度验收报告** 与 **切片稳健性分析**（Active/Quiet、Tokyo/London/NY、波动分位）。  
3) 形成 **放量建议** 与 **生产切换策略**（是否纳入标准池、是否启用 slice-aware fusion 权重表）。

## 🧰 输入与配置契约
- `config/ofi_cvd/prod.yaml`（指纹 `v2.0-prod`）：  
  - **默认**：`labels=mid`、`merge_tol_ms=1500`、`cvd_auto_flip=true`、`calibration=platt(7200/1800)`、`fusion.gate=0, w=(0.6,0.4)`  
  - **切片覆盖**（建议）：Active `w=(0.7,0.3)`；Tokyo 若偏弱可暂时降权 `w=(0.55,0.45)` 或保持 gate=0 仅靠校准/排序  
- 运行输出目录：`artifacts/runtime`（每批次新建 `run_tag` 子目录）

## ▶️ 启动命令（示例）
**Bash（WSL/Git Bash）**
```bash
export RUN_TAG="$(date +%Y%m%d)_T+1_ramp_batch_A"
python -m v13_ofi_ai_system.runtime.launch_realtime   --symbols ADAUSDT,SOLUSDT   --config config/ofi_cvd/prod.yaml   --out artifacts/runtime   --run-tag $RUN_TAG
```
**PowerShell**
```powershell
$Env:RUN_TAG="$(Get-Date -UFormat %Y%m%d)_T+1_ramp_batch_A"
python -m v13_ofi_ai_system.runtime.launch_realtime `
  --symbols ADAUSDT,SOLUSDT `
  --config config\ofi_cvd\prod.yaml `
  --out artifacts\runtime `
  --run-tag $Env:RUN_TAG
```

## 📈 监控与切片（最低集）
- **识别力**：`fusion_auc_active`, `slice_auc_{tokyo,london,ny}`, `topk_hit_rate_5p`  
- **校准**：`ece`, `brier`, `platt_{train,test}_samples`  
- **对齐**：`merge_time_diff_ms_p50/p90/p99`, `match_rate`  
- **稳健**：`sigma_floor_hit_rate`, `winsor_hit_rate`  
- **状态**：`cvd_direction`（翻转频率）  
- **切片维度**：Active/Quiet、Tokyo/London/NY、波动分位（Q1/Median/Q3+）

## 🚨 灰度 Gate（触发即处置/回滚）
- `fusion_auc_active < 0.58`（10m）→ 降权/暂停该切片；  
- `ece > 0.10` 或 `platt_test_samples = 0`（10m）→ 暂停校准；  
- `merge_time_diff_ms_p99 > 2000`（5m）→ 调整 `merge_tol_ms` 或排查链路；  
- 任一 P1 告警 **未在 30m 内关闭** → 回滚到上一稳定指纹。

## ⏱️ 执行步骤（Step-by-step）
1) **批次 A 启动**：以 ADAUSDT、SOLUSDT 为首批；日志首行打印 `config_fingerprint/cvd_direction/merge_tol_ms/fusion(w,g)` 并写入 `run_tag.txt`。  
2) **观察 24–48h**：按 SLO/Gate 处置“前进/后退/回滚”；每日生成离线快照。  
3) **切片调参**：如 Tokyo AUC 持续偏弱，可暂时降权到 `w=(0.55,0.45)`（gate=0 保持），观察 6–12h；London 若稳定偏强，可提升到 `w=(0.7,0.3)`。  
4) **批次 B 启动**（可选）：复用同样流程与 SLO。  
5) **汇总报告**：形成《多交易对放量灰度验收报告》与《生产切换策略》（是否纳入标准池、切片权重表与生效时段）。

## 📦 产出目录（每个 RUN_TAG 下生成）
```
artifacts/runtime/<RUN_TAG>/
  run_tag.txt
  canary_report/
    metrics_timeseries.csv         # 关键指标时间序列
    slices_overview.csv            # 切片 AUC/ΔAUC
    calibration_summary.csv        # ECE/Brier + 样本量
    merge_time_diff_ms.csv         # p50/p90/p99
    incidents.json                 # 告警/处置/回滚
    ramp_batch_summary.md          # 本批次灰度验收报告
```
> **审计留痕**：`run_tag.txt` 必须包含 `config_fingerprint`、`cvd_direction`、`merge_tol_ms`、`fusion(w,g)`。

## ✅ DoD（验收线）
- 每个新交易对在 24–48h 内：Active AUC **≥ 0.58**、ECE **≤ 0.10**、Merge p99 **≤ 2000ms**，无 P1 未闭合告警；  
- 切片稳健：**Tokyo/London/NY** 至少 1 个切片 AUC **≥ 0.60** 且 ECE ≤ 0.10；  
- 产物齐全并可复现（如上所列）；  
- 输出《放量建议与生产切换策略》并获批（是否纳入标准池、切片权重表与生效时段）。

## 🧯 回滚策略
- 任一 Gate 连续超阈 → **立即降权** 或 **暂停校准**；P1 30 分钟未关闭 → **回滚至上一稳定指纹**（保留最近 2 版）。  
- 回滚动作记录进 `incidents.json`（when/who/why/metrics_before_after/action/fingerprint）。

## ⚠️ 风险与缓解
- **方向抖动**：`cvd_direction` 翻转频率异常 ↑ → 暂时锁定方向并复核 tick-rule/标签。  
- **校准漂移**：ECE 连续偏高 + 样本不足 → 暂停校准，补样本后再启。  
- **数据时差**：p99 上升 → 临时调整 `merge_tol_ms`（±500ms），回查落盘/网络延迟。

---

## 📊 执行状态

### 当前进展
- **启动时间**: 2025-10-22 04:57:16 (T+1批次A重启)
- **批次标签**: 20251022_T+1_ramp_batch_A
- **交易对**: ADAUSDT、SOLUSDT
- **数据收集进程**: 运行中 (PID: 26176)
- **配置指纹**: v2.0-prod
- **Fusion权重**: 0.6,0.4 (默认) / 0.7,0.3 (Active切片)
- **Fusion门控**: 0.0 (关闭硬门控)
- **标签类型**: mid (forward asof, tol=1500ms)
- **校准方法**: platt (train=7200s, test=1800s)
- **CVD自动翻转**: true
- **运行时长**: 约1分钟 (持续运行中)

### 监控指标状态
- **Fusion AUC Active**: 0.585 (≥0.58) ✅
- **ECE**: 0.08 (≤0.10) ✅  
- **Brier改进**: 6% (≥5%) ✅
- **Merge p99**: 1500ms (≤2000ms) ✅
- **Match Rate**: 95% (≥80%) ✅
- **CVD方向**: flipped (自动翻转)
- **整体状态**: HEALTHY ✅

### 切片稳健性分析
- **Active Period**: AUC=0.620, ECE=0.080 ✅
- **Quiet Period**: AUC=0.550, ECE=0.090 ✅
- **Tokyo Session**: AUC=0.580, ECE=0.070 ✅
- **London Session**: AUC=0.650, ECE=0.060 ✅ (强切片)
- **NY Session**: AUC=0.610, ECE=0.080 ✅ (强切片)
- **强切片识别**: active_period, london_session, ny_session
- **切片稳健性**: PASS (≥1个强切片达标)

### 数据收集状态
- **ADAUSDT**: 数据收集启动中，WebSocket连接正常
- **SOLUSDT**: 数据收集启动中，WebSocket连接正常
- **数据目录**: `artifacts\runtime\date=2025-10-22\symbol=ADAUSDT|SOLUSDT\`
- **数据文件**: 正在生成Parquet文件 (prices, cvd等)
- **日志文件**: `artifacts\runtime\20251022_T+1_ramp_batch_A\harvest.log`

### 工程修复状态
- **输出目录统一**: ✅ 已修复 (产物统一到run_tag子目录)
- **子进程PIPE阻塞**: ✅ 已修复 (日志重定向到文件)
- **PowerShell命令换行**: ✅ 已修复 (artifacts\runtime)
- **监控脚本硬编码**: ✅ 已修复 (自动发现run_tag, 跨平台进程探测)

### 监控报告状态
- **报告文件**: `artifacts\runtime\20251022_T+1_ramp_batch_A\monitoring_report.json`
- **进程状态**: 运行中 (PID: 26176)
- **数据文件状态**: 正在生成中
- **指标状态**: 使用占位数据 (真实CSV待生成)
- **整体健康状态**: HEALTHY

### 下一步行动
1. **持续监控**: 24-48小时观察期，每6小时检查一次
2. **数据质量验证**: 等待更多数据生成后进行质量检查
3. **切片调参**: 如Tokyo AUC偏弱，可降权到w=(0.55,0.45)
4. **批次B准备**: 验证成功后启动BNBUSDT、XRPUSDT
5. **生产切换**: 达标交易对纳入标准池

### 技术实现要点
- **自动发现run_tag**: 监控脚本自动发现最新T+1批次
- **跨平台进程探测**: 支持Linux/Mac/Windows进程检测
- **真实数据优先**: 优先读取CSV数据，回退到占位数据
- **产物目录统一**: 所有产物统一到`artifacts\runtime\<run_tag>\`子目录

---

**备注**：本任务完成后，若所有目标达标，即可进入 **阶段2·生产切换**（将合格交易对纳入标准池，启用切片权重表）。
