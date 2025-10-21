# Task 1.3.1 (v2): 收集历史OFI+CVD数据（可直接执行）

## 📋 任务信息
- **任务编号**: Task_1.3.1 (v2)
- **所属阶段**: 阶段1 - 真实OFI+CVD核心
- **任务状态**: ⏳ 待开始
- **优先级**: 高
- **预计时间**: 48-72小时（自动运行）
- **实际时间**: (完成后填写)

## 🎯 任务目标（更可验证）

连续采集48小时（建议72h）BTCUSDT、ETHUSDT（可增减）实时数据，覆盖高/中/低活跃时段。

产出5类分区化数据集（Parquet）：prices, ofi, cvd, fusion, events。

生成数据质量报告（完整性/去重/时钟/延迟/信号量）用于Task 1.3.2的分析输入。

## 📦 产出与目录结构

```
data/ofi_cvd/
  date=YYYY-MM-DD/
    symbol=BTCUSDT/
      kind=prices|ofi|cvd|fusion|events/
        part-*.parquet
    symbol=ETHUSDT/...
artifacts/
  run_logs/harvest_YYYYMMDD_HHMM.log
  dq_reports/dq_YYYYMMDD.json
```

### 表结构（最少字段）

**prices**: ts_ms, event_ts_ms, symbol, price, qty, agg_trade_id, latency_ms, recv_rate_tps

**ofi**: ts_ms, symbol, ofi_value, ofi_z, scale, regime

**cvd**: ts_ms, symbol, cvd, delta, z_raw, z_cvd, scale, sigma_floor, floor_used, regime

**fusion**: ts_ms, symbol, score, score_z, regime

**events**（背离/枢轴/异常）: ts_ms, symbol, event_type, meta_json

**压缩**: Snappy；**分区**: date、symbol、kind；**滚动写入**: 每60秒一个文件。

## 📝 任务清单
- [ ] 创建统一数据采集脚本run_success_harvest.py
- [ ] 设置环境变量和配置参数
- [ ] 运行10分钟预检（连续性/去重率/时间漂移/小样本DoD）
- [ ] 连续采集48-72小时BTCUSDT、ETHUSDT数据
- [ ] 生成5类分区化Parquet数据集
- [ ] 创建数据质量验证脚本
- [ ] 设置Prometheus监控和Grafana仪表板
- [ ] 生成数据质量报告
- [ ] 验证DoD验收标准

## 📦 Allowed Files
- `v13_ofi_ai_system/examples/run_success_harvest.py` (统一采集脚本)
- `v13_ofi_ai_system/scripts/validate_ofi_cvd_harvest.py` (验证脚本)
- `v13_ofi_ai_system/data/ofi_cvd/` (数据文件)
- `v13_ofi_ai_system/artifacts/` (日志和报告)

## 📚 依赖项
- **前置任务**: Task_1.2.13 (CVD Z-score微调优化)
- **依赖包**: pandas, pyarrow, prometheus_client, websockets

## ✅ 验证标准（DoD）

### 完整性
- 按1分钟桶聚合，空桶率 < 0.1%
- 文件滚动无缺口

### 去重
- 按agg_trade_id去重后，重复率 < 0.5%

### 延迟
- latency_ms p99 < 120ms；p50 < 60ms

### 信号量
- events（背离/枢轴/异常）合计 ≥ 1000条/72h（多交易对总计）

### 一致性
- cvd.z_raw与z_cvd在尾部存在分离（winsor生效）
- 包含scale/sigma_floor/floor_used诊断字段

## 🛠 运行方式（推荐一键脚本）

### 运行入口
`v13_ofi_ai_system/examples/run_success_harvest.py` 与CVD统一为一个harvest脚本/进程（或supervisor管两个进程），写入同一目录。

### 关键环境变量
```bash
SYMBOLS=BTCUSDT,ETHUSDT
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
```

### 运行前10分钟预检（必做）
- **连续性**: 无方向不一致、无异常重连风暴（>3次/10m）
- **去重率**: 重复样本 < 0.2%
- **时间漂移**: abs(event_ts_ms - recv_ts_ms) p99 < 120ms
- **小样本DoD**: prices、cvd均已落盘，schema正确

## 📈 监控与告警（上线即用）

### 告警规则
- `latency_ms_p99 > 120` 持续10m
- `ws_reconnects_total > 10` /h
- `cvd_floor_hit_rate > 60%` 持续15m（疑似市况/参数失配）
- 空桶率 > 0.5% /h

### 仪表板指标
- **24h趋势**: p(|Z|>2), p(|Z|>3), scale_median, recv_rate_tps（Overall/Active/Quiet）
- **数据健康**: 空桶热力图、去重率、写入时延

### Prometheus指标
```
recv_rate_tps, ws_reconnects_total, dedup_hits_total, 
latency_ms_p50/p90/p99, cvd_scale_median, cvd_floor_hit_rate, 
write_errors_total, parquet_flush_sec
```

## 🔁 稳定性与恢复

### WebSocket
- 指数回退 + 抖动、心跳（20s）
- 重连期间启用LRU去重

### Checkpoint
- 每次成功写Parquet后推进state/checkpoint.json（最后offset与时间）

### 自动恢复
- 进程重启从checkpoint恢复
- 写入失败重试3次 → 打点并跳过（不阻塞流）

## 🧪 收尾校验脚本

### 脚本位置
`scripts/validate_ofi_cvd_harvest.py`

### 功能
跑完给出JSON报告（用于DoD）：
- 空桶率计算
- 延迟统计（p50/p90/p99）
- 数据行数和分钟数统计
- 交易对列表

## 📊 DoD检查清单
- [ ] 代码无语法错误
- [ ] 通过lint检查
- [ ] 通过所有测试
- [ ] 无mock/占位/跳过
- [ ] 产出真实验证结果
- [ ] 性能达标
- [ ] 更新相关文档
- [ ] 数据完整性验证通过
- [ ] 延迟指标达标
- [ ] 信号量达标
- [ ] 监控指标正常

## 📝 执行记录
**开始时间**: (填写)  
**完成时间**: (填写)  
**执行者**: AI Assistant

### 遇到的问题
- (记录问题)

### 解决方案
- (记录解决方案)

### 经验教训
- (记录经验)

## 🗓 执行步骤（Checklist）

1. **锁定依赖版本**，打印"配置指纹"（Z_MODE/SCALE/MAD/FAST/HL）
2. **跑10分钟预检**，全部通过→开始48–72h采集
3. **采集中监控**上述指标与告警；异常触发则不中断采集但标注run_tag
4. **每6小时生成一次DQ报告**；任务结束汇总总报告与样本快照（前/后1h）
5. **产出归档**：打tag（日期+参数指纹），写入阶段总览与下一任务链接

## ⚠️ 注意事项（风险与规避）

### 时钟
- 以event_ts_ms为主时钟，recv_ts_ms仅用于延迟
- 若系统时间漂移>200ms，立即记录并上报

### 磁盘
- 确保写入吞吐≥ 10MB/s
- 磁盘占用预估BTC/ETH 72h ≈ 2–5 GB（视行情）

### 权限
- 目录可写，异常退出时不会遗留半写文件（先写临时名，flush后rename）

## 🔗 相关链接
- 上一个任务: [Task_1.2.13_CVD_Z-score微调优化](./Task_1.2.13_CVD_Z-score微调优化.md)
- 下一个任务: [Task_1.3.2_创建OFI+CVD信号分析工具](./Task_1.3.2_创建OFI+CVD信号分析工具.md)
- 阶段总览: [📋V13_TASK_CARD.md](../../📋V13_TASK_CARD.md)

---
**任务状态**: ⏳ 待Task_1.2.13完成后开始  
**质量评分**: (完成后填写)  
**是否可以继续下一个任务**: ❓ 待测试通过后确定

