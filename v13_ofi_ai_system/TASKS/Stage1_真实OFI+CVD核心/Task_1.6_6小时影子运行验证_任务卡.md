# Task 1.6：6小时影子运行验证

## 📋 任务信息
- **阶段**：阶段1 - 真实OFI+CVD核心
- **状态**：⏳ **待开始**
- **优先级**：P0（核心验证）
- **观测窗口**：6小时连续运行
- **范围**：在修复Blockers后，运行6小时影子测试，验证系统稳定性和信号质量
- **创建时间**：2025-10-22 07:00:00 UTC+09:00

## 🎯 目标（量化可验）
1) **系统稳定性**：6小时连续运行无异常，计算周期p95 < 1500ms，写盘成功率 = 100%
2) **信号质量验证**：验证修复后的时间对齐、成本模型、列名统一等Blockers修复效果
3) **交易对覆盖**：3个活跃交易对（BTCUSDT, ETHUSDT, BNBUSDT）同时运行
4) **健康度监控**：产出运行健康度报告，包含关键指标趋势和异常检测
5) **影子成交明细**：生成详细的影子交易日志，包含入场/出场/撤单等动作记录

## 🚫 非目标
- 不进行真实交易，仅影子/纸上撮合
- 不修改核心算法参数
- 不接入新的数据源

## 🔗 前置条件
- Task 1.5 核心算法v1开发完成
- Blockers修复完成（时间对齐、成本模型、列名统一、网格搜索）
- 48小时数据收集进行中，有足够的历史数据支撑
- 统一配置系统集成完成

## 📦 范围与组件
### 6.1 影子运行器（Shadow Runner）
- **数据源**：使用真实数据收集器的历史数据
- **信号计算**：OFI、CVD、Fusion、Divergence信号实时计算
- **状态机**：FLAT → LONG/SHORT → COOLING → FLAT
- **触发条件**：score上穿±z_hi，需K笔&T秒确认，guard检查通过
- **交易动作**：enter/exit/cancel/SL/TP意图生成
- **日志记录**：每笔交易动作的详细记录

### 6.2 影子收集器（Shadow Collector）
- **数据加载**：从data/ofi_cvd加载最新数据
- **信号处理**：Robust Z-score计算，时间列统一
- **数据流转**：确保ts_ms、z_ofi、z_cvd列名一致性
- **性能监控**：计算周期、内存使用、数据质量指标

### 6.3 健康度监控（Health Monitoring）
- **系统指标**：CPU使用率、内存使用率、磁盘I/O
- **业务指标**：信号计算周期、写盘成功率、数据延迟
- **质量指标**：信号强度分布、交易触发频率、异常检测
- **告警机制**：关键指标超阈值时触发告警

### 6.4 成交明细记录（Trade Details）
- **交易日志**：每笔影子交易的完整记录
- **动作类型**：enter_long/short、exit_by_opposite/tp/sl/timeout、cancel_pending
- **信号信息**：触发时的score、z_ofi、z_cvd、regime状态
- **时间戳**：精确到毫秒的交易时间记录

## 📈 监控指标与SLO（门槛）
### 运行时指标
- **计算周期**：p95 < 1500ms，p99 < 3000ms
- **写盘成功率**：100%（无失败记录）
- **内存使用**：峰值 < 2GB，平均 < 1GB
- **CPU使用率**：平均 < 50%，峰值 < 80%

### 业务指标
- **信号质量**：z_ofi、z_cvd分布合理，无异常值
- **交易触发**：每小时交易次数 5-50次（避免过度交易或信号缺失）
- **数据延迟**：信号计算延迟 < 5秒
- **异常检测**：无数据中断、无计算错误、无内存泄漏

### 准入门槛
- **连续运行**：6小时无异常退出
- **数据完整性**：所有时间窗口都有信号输出
- **交易合理性**：交易动作符合策略逻辑，无异常交易

### 度量口径（统一）
- **统计窗口**：rolling=5min, step=1min；并输出6小时全局统计
- **计算周期**：信号计算从收到最新一条行情到信号（或意图）写盘完成的耗时
- **数据延迟**：从行情入队时间到信号落盘时间
- **写盘成功**：允许重试≤3次且零数据丢失视为成功；超过阈值计入write_fail_events并判定失败
- **交易频率**：按symbol分布输出trades_per_hour_by_symbol，低活跃时段（预设白名单时间段）不作为否决项，仅记录偏差原因
- **预热排除**：首10min样本不计入SLO计算，但在报告中单列"WARMUP段"

### 触发与去重策略
- **触发参数**：K=3, T=2s；阈值穿越采用"首次穿越计一次，回落复位后方可再次触发"
- **成本模型版本**：成交价格口径（price_ref=mid或last）、滑点实现写入明细
- **去重规则**：一次阈值穿越只计一次，避免重复触发

### 告警与处置
- **WARN**：任一SLO在5分钟窗口内超阈值→10分钟内恢复则合格
- **CRIT**：连续10分钟超阈或出现不可恢复错误→触发on-call（@SRE），记录处置结论
- **自动恢复**：允许单次自动拉起，累计中断<60秒仍视为达标

## ⚡ 执行步骤（Checklists）
### 预检
- [ ] 确认Blockers修复完成，所有组件可正常运行
- [ ] 检查数据源可用性，确保有足够的历史数据
- [ ] 验证统一配置系统加载正常
- [ ] 检查输出目录权限，确保可写入日志文件
- [ ] 固化运行基线（CPU/内存/磁盘/容器限额、Python版本、依赖哈希、时区/时钟同步）
- [ ] 生成run_id、git_commit、config_sha256等可复现信息
- [ ] 设置告警渠道（Pager/Slack/Email）和响应时限

### 执行中
- [ ] 启动3个交易对的影子运行器
- [ ] 启动影子收集器，开始数据加载和信号计算
- [ ] 启动健康度监控，记录关键指标
- [ ] 每10分钟输出一次运行状态摘要
- [ ] 排除首10分钟预热段，从第11分钟开始统计SLO指标
- [ ] 监控告警状态，记录WARN/CRIT事件和处置结论
- [ ] 实时监控write_fail_events，确保重试≤3次内成功

### 收尾
- [ ] 生成6小时运行健康度报告
- [ ] 生成影子成交明细CSV文件
- [ ] 分析关键指标趋势和异常点
- [ ] 输出GO/NO-GO建议
- [ ] 包含可复现信息（run_id、git_commit、config_sha256、env_spec、tz、ntp_status）
- [ ] 生成config_snapshot.yaml
- [ ] 统计trades_per_hour_by_symbol，分析低活跃时段偏差
- [ ] 记录告警时间线与处置结论

## 📁 交付物
### 核心文件
- **影子运行器**：`runner/shadow_runner.py`（已修复）
- **影子收集器**：`runner/shadow_collector.py`（已修复）
- **健康度监控**：`monitoring/health_monitor.py`（新建）
- **成交明细记录**：`logging/trade_logger.py`（新建）

### 配置文件
- **运行配置**：`configs/shadow_run_config.yaml`
- **监控配置**：`configs/health_monitoring.yaml`
- **日志配置**：`configs/logging_config.yaml`

### 输出文件
- **健康度报告**：`artifacts/shadow_run/health_report.json`
- **成交明细**：`artifacts/shadow_run/trade_details.csv`
- **运行日志**：`artifacts/shadow_run/run_log.txt`
- **指标趋势**：`artifacts/shadow_run/metrics_trends.png`
- **配置快照**：`artifacts/shadow_run/config_snapshot.yaml`
- **告警记录**：`artifacts/shadow_run/alert_timeline.json`

## 👥 角色与分工
- **算法/量化**：@Quant（信号质量验证、交易逻辑检查）
- **数据/平台**：@DataEng（数据加载、性能优化）
- **SRE/监控**：@SRE（健康度监控、告警设置）
- **PM/评审**：@PM（结果评审、GO/NO-GO决策）

## 🚨 风险与缓解
- **数据中断** → 自动重连机制，数据缓存策略
- **内存泄漏** → 定期内存检查，自动重启机制
- **计算超时** → 超时保护，降级处理
- **磁盘满** → 日志轮转，自动清理

## 🚪 准入门槛（进入下一阶段）
- 6小时连续运行无异常
- 所有SLO指标达标
- 信号质量符合预期
- 交易逻辑正确执行

## 📋 附录
### A. 健康度监控指标
```yaml
system_metrics:
  cpu_usage: < 50%
  memory_usage: < 1GB
  disk_io: < 100MB/s
  network_latency: < 100ms

business_metrics:
  signal_calculation_p95: < 1500ms
  write_success_rate: 100%
  data_delay: < 5s
  trade_frequency: 5-50/hour
```

### B. 交易动作类型
```
Actions:
  enter_long: 做多入场
  enter_short: 做空入场
  exit_by_opposite: 反向信号出场
  exit_by_tp: 止盈出场
  exit_by_sl: 止损出场
  exit_by_timeout: 超时出场
  cancel_pending: 撤单
```

### C. 运行监控命令
```bash
# 启动6小时影子运行
python scripts/run_6h_shadow.py --symbols BTCUSDT,ETHUSDT,BNBUSDT --duration 6

# 实时监控
python scripts/monitor_shadow.py --run-id shadow_run_20251022

# 健康度检查
python scripts/health_check.py --report artifacts/shadow_run/health_report.json
```

### D. 预期输出示例
```json
{
  "run_id": "shadow_run_20251022_070000",
  "started_at": "2025-10-22T07:00:00Z",
  "git_commit": "abc123def456",
  "config_sha256": "sha256:789...",
  "env_spec": {
    "cpu_limit": "2.0",
    "mem_limit": "4GB",
    "python_version": "3.9.7"
  },
  "tz": "UTC+9",
  "ntp_status": "synced",
  "duration_hours": 6,
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "health_status": "healthy",
  "total_trades": 156,
  "success_rate": 100.0,
  "avg_calculation_time_ms": 1200,
  "memory_peak_gb": 1.2,
  "trigger_policy": {
    "K": 3,
    "T": 2,
    "price_ref": "mid",
    "cost_model_ver": "v1.0"
  },
  "trades_per_hour_by_symbol": {
    "BTCUSDT": 25.3,
    "ETHUSDT": 18.7,
    "BNBUSDT": 12.1
  },
  "warmup_excluded": true,
  "write_fail_events": 0,
  "alert_timeline": []
}
```

---

## 📊 执行状态

### 当前进展
- **计划执行时间**: 2025-10-22 08:00-14:00 JST (6小时连续运行)
- **目标交易对**: 3个活跃交易对（BTCUSDT, ETHUSDT, BNBUSDT）
- **验证重点**: Blockers修复效果、系统稳定性、信号质量

### 预检清单
- [ ] 确认Blockers修复完成，所有组件可正常运行
- [ ] 检查数据源可用性，确保有足够的历史数据
- [ ] 验证统一配置系统加载正常
- [ ] 检查输出目录权限，确保可写入日志文件

### 执行清单
- [ ] 启动3个交易对的影子运行器
- [ ] 启动影子收集器，开始数据加载和信号计算
- [ ] 启动健康度监控，记录关键指标
- [ ] 每10分钟输出一次运行状态摘要
- [ ] 生成6小时运行健康度报告
- [ ] 生成影子成交明细CSV文件
- [ ] 分析关键指标趋势和异常点
- [ ] 输出GO/NO-GO建议

### 执行结果
- **执行时间**: 待执行
- **整体状态**: ⏳ **待开始**
- **健康度状态**: 待监控
- **交易统计**: 待统计
- **异常记录**: 待记录

### 关键成果
- **系统稳定性**: 待验证
- **信号质量**: 待验证
- **交易逻辑**: 待验证
- **性能指标**: 待记录

### 产出文件
- `artifacts/shadow_run/health_report.json` - 健康度报告
- `artifacts/shadow_run/trade_details.csv` - 成交明细
- `artifacts/shadow_run/run_log.txt` - 运行日志
- `artifacts/shadow_run/metrics_trends.png` - 指标趋势图
- `artifacts/shadow_run/config_snapshot.yaml` - 配置快照
- `artifacts/shadow_run/alert_timeline.json` - 告警记录

### 下一步行动
1. **启动6小时影子运行**: 使用修复后的组件运行6小时连续测试
2. **健康度监控**: 实时监控系统指标和业务指标
3. **结果分析**: 分析运行结果，验证Blockers修复效果
4. **报告生成**: 生成详细的健康度报告和成交明细
5. **GO/NO-GO决策**: 基于运行结果决定是否进入下一阶段

---

**备注**：本任务是对Task 1.5核心算法v1的验证任务，通过6小时连续运行验证Blockers修复效果，确保系统稳定性和信号质量，为进入测试网策略驱动做准备。
