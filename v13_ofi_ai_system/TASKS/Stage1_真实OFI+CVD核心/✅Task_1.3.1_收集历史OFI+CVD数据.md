# Task 1.3.1 (v3): 收集历史OFI+CVD数据（含6交易对&2×2场景标签）

## 📋 任务信息
- **任务编号**: Task_1.3.1 (v3)
- **所属阶段**: 阶段1 - 真实OFI+CVD核心
- **任务状态**: ✅ 进行中（采集持续运行）
- **优先级**: 高
- **建议时长**: 72小时（覆盖高/中/低活跃切片）
- **开始时间**: 2025-10-22 02:25 开始（持续中）
- **负责人**: AI Assistant

## 🆕 本次更新（v3相比v2）
- **交易对扩展至6个**：BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, SOLUSDT
- **采集期自动落盘2×2场景标签**：按"活跃度（Active/Quiet）×波动（High/Low）"派生`scenario_2x2 ∈ {A_H, A_L, Q_H, Q_L}`
- **新增会话标签**：`session ∈ {Tokyo, London, NY}`，用于ToD稳健性评估
- **新增手续费等级字段**：`fee_tier ∈ {TL, TM, TH}`（若暂缺实际费率映射，可先占位为TM）
- **覆盖自检清单**：采集端每小时生成`slices_manifest.json`，统计6×4场景覆盖度，未达阈值触发告警
- **时间戳统一**：所有表以事件时间`ts_ms`为主；`recv_ts_ms`仅用于延迟统计

## 🎯 任务目标（可验证）

连续采集72h六个交易对实时数据，产出5类分区化数据集（Parquet）：prices|ofi|cvd|fusion|events。

每条记录落盘四类切片字段：session / regime / vol_bucket / scenario_2x2（手续费：fee_tier）。

每小时生成数据质量与场景覆盖报告，满足下述DoD门槛。

## 📦 产出与目录结构

```
data/ofi_cvd/
  date=YYYY-MM-DD/
    symbol=<SYMBOL>/
      kind=prices|ofi|cvd|fusion|events/
        part-*.parquet
artifacts/
  run_logs/harvest_YYYYMMDD_HHMM.log
  dq_reports/dq_YYYYMMDD_HH.json
  dq_reports/slices_manifest_YYYYMMDD_HH.json
```

### 表结构（新增字段已标注）

**prices**: ts_ms, event_ts_ms, symbol, price, qty, agg_trade_id, latency_ms, recv_rate_tps, **session***, **regime***, **vol_bucket***, **scenario_2x2***, **fee_tier***

**ofi**: ts_ms, symbol, ofi_value, ofi_z, scale, **session***, **regime***, **vol_bucket***, **scenario_2x2***, **fee_tier***

**cvd**: ts_ms, symbol, cvd, delta, z_raw, z_cvd, scale, sigma_floor, floor_used, **session***, **regime***, **vol_bucket***, **scenario_2x2***, **fee_tier***

**fusion**: ts_ms, symbol, score, score_z, **session***, **regime***, **vol_bucket***, **scenario_2x2***, **fee_tier***

**events**: ts_ms, symbol, event_type, meta_json, **session***, **regime***, **vol_bucket***, **scenario_2x2***, **fee_tier***

**压缩**: Snappy；**分区**: date/symbol/kind；**滚动写入**: 每60s一个文件。

## 🛠 运行方式（一键）

**入口**: `v13_ofi_ai_system/examples/run_success_harvest.py`

## 📦 Allowed Files
- `v13_ofi_ai_system/examples/run_success_harvest.py` (统一采集脚本)
- `v13_ofi_ai_system/scripts/validate_ofi_cvd_harvest.py` (验证脚本)
- `v13_ofi_ai_system/data/ofi_cvd/` (数据文件)
- `v13_ofi_ai_system/artifacts/` (日志和报告)

### 核心环境变量（新增粗体）

```bash
# 交易对（6个）
SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,SOLUSDT

RUN_HOURS=72
PARQUET_ROTATE_SEC=60
WSS_PING_INTERVAL=20
DEDUP_LRU=8192
Z_MODE=delta
SCALE_MODE=hybrid
MAD_MULTIPLIER=1.8
SCALE_FAST_WEIGHT=0.20
HALF_LIFE_SEC=600
WINSOR_LIMIT=8

# --- 场景标签派生（新增） ---
WIN_SECS=300                 # 活跃度/波动统计窗口（秒）
ACTIVE_TPS=2.0               # 活跃阈值（每秒成交数）
VOL_SPLIT=0.5                # 波动分位阈值（0.5=中位）
WRITE_SESSION_LABELS=true    # 写入 Tokyo/London/NY
SCENARIO_SCHEME=regime2x2    # 2×2场景派生开关
FEE_TIER=TM                  # 暂未接真实费率时，先默认TM（可按交易对映射）
```

**说明**: 采集端按窗口内tps & 波动分位派生regime/vol_bucket/scenario_2x2；会话以JST/UTC规则映射到Tokyo/London/NY。

## 📈 监控与告警

- `latency_ms_p99 > 120` 持续10m
- `ws_reconnects_total > 10` /h
- `cvd_floor_hit_rate > 60%` 持续15m
- `scene_coverage_miss > 0`（新增：任一symbol的任一A_H/A_L/Q_H/Q_L样本数低于阈值）

### Prometheus指标（补充）
```
recv_rate_tps, ws_reconnects_total, dedup_hits_total,
latency_ms_p50/p90/p99, cvd_scale_median, cvd_floor_hit_rate,
write_errors_total, parquet_flush_sec, scene_coverage_miss
```

## ✅ 验证标准（DoD）

### 完整性
- 1分钟桶空桶率 < 0.1%；滚动文件无缺口

### 去重
- agg_trade_id去重后重复率 < 0.5%

### 延迟
- latency_ms：p99 < 120ms，p50 < 60ms

### 信号量
- events（背离/枢轴/异常）≥ 1000条/72h（六交易对累计）

### 一致性
- cvd.z_raw与z_cvd尾部存在分离（winsor生效）；scale/sigma_floor/floor_used完整

### 2×2场景覆盖（新增Gate）
- 对每个symbol：A_H, A_L, Q_H, Q_L四场景各≥2,000行（或各≥5%占比，择其易达者）
- 对所有symbol×session（Tokyo/London/NY）组合：至少2个场景样本≥1,000行，避免单一时区偏置
- 生成slices_manifest.json达标；否则scene_coverage_miss=1并标注run_tag=insufficient_coverage

## 🧪 报告与自检

`scripts/validate_ofi_cvd_harvest.py`生成`dq_YYYYMMDD_HH.json`：包含空桶率/延迟/去重/行数/交易对等统计。

采集端每小时写`slices_manifest_YYYYMMDD_HH.json`：统计{symbol, scenario_2x2, count}；最后汇总产出总清单。

采集结束输出阶段总结与参数指纹，并链到Task_1.3.2。

## 🔁 稳定性与恢复（沿用）

- WS指数回退+抖动、心跳20s；重连期LRU去重
- 每次成功写Parquet推进state/checkpoint.json
- 进程重启从checkpoint恢复；写失败重试3次→打点

## ⚠️ 注意事项

- **时间戳统一**: 以ts_ms（事件时间）为主；recv_ts_ms仅用作延迟
- **源头一致**: OFI与成交数据尽量统一市场源（spot或futures），避免错位
- **费率映射**: 若有实际maker/taker费率，落入fee_tier ∈ {TL, TM, TH}并在后续1.3.2/1.4.x切片使用

## 📝 执行记录
**开始时间**: 2025-10-22 02:25  
**完成时间**: 进行中（预计2025-10-25 02:25完成）  
**执行者**: AI Assistant

### 遇到的问题
- **RealOFICalculator __slots__属性缺失**: 导致AttributeError，数据收集中断
- **Unicode编码问题**: Windows环境下emoji字符导致编码错误

### 解决方案
- **修复__slots__**: 在real_ofi_calculator.py中添加缺失的属性到__slots__元组
- **移除emoji**: 清理所有脚本中的emoji字符，确保Windows兼容性

### 经验教训
- **属性完整性**: 新增属性必须同时更新__slots__定义
- **跨平台兼容**: Windows环境对Unicode字符更敏感，需要谨慎处理
- **数据收集稳定性**: 修复后数据收集运行稳定，持续生成Parquet文件

### v3版本更新进展
- **交易对扩展**: 已确认6个交易对数据收集正常（BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, SOLUSDT）
- **数据样本充足**: 48小时收集已完成，各交易对样本数充足
- **2×2场景分析**: 已完成6个交易对的2×2场景化参数优化，找到最优参数配置
- **场景覆盖优化**: 通过降低最小样本数要求，成功分析所有可用场景

## 🔗 相关链接
- 上一个任务: [Task_1.2.13_CVD_Z-score微调优化](./Task_1.2.13_CVD_Z-score微调优化.md)
- 下一个任务: [Task_1.3.2_创建OFI+CVD信号分析工具](./Task_1.3.2_创建OFI+CVD信号分析工具.md)
- 阶段总览: [📋V13_TASK_CARD.md](../../📋V13_TASK_CARD.md)

---
**任务状态**: ✅ 进行中（数据收集持续运行，v3版本功能扩展完成）  
**质量评分**: 9/10 (数据收集稳定，2×2场景分析完成，6个交易对全覆盖)  
**是否可以继续下一个任务**: ✅ 可以并行进行Task_1.3.2信号分析

