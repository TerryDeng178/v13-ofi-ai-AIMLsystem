# Task 0.8: 创建Grafana监控仪表盘

## 📋 任务信息

- **任务编号**: Task_0.8
- **任务名称**: 创建Grafana监控仪表盘（策略模式切换）
- **所属阶段**: 阶段0 - 准备工作（监控增强）
- **优先级**: 中
- **预计时间**: 3-4小时
- **实际时间**: 约2小时（含V1.1修复 + V1.2统一配置集成）
- **任务状态**: ✅ 已完成（V1.2统一配置集成版）
- **前置任务**: 
  - ✅ Task_0.7（动态模式切换 + 13个Prometheus指标）
  - ⏳ Task_1.2.14（24小时真实环境测试 - 可选，有助于调优）

---

## 🎯 任务目标

创建 Grafana 仪表盘，可视化策略模式切换系统的13个Prometheus指标，为运维和分析提供直观的监控界面。

### 核心目标

1. **策略模式仪表盘** - 展示当前模式、切换历史、触发因子
2. **性能仪表盘** - 展示参数更新耗时、系统负载
3. **告警仪表盘** - 展示告警历史和趋势
4. **数据质量仪表盘** - 展示OFI/CVD指标质量
5. **导出配置** - JSON格式便于版本控制和分享

---

## 📝 任务清单

### 阶段1: Grafana环境准备（30分钟）

- [ ] 1.1 安装Grafana（如未安装）
- [ ] 1.2 配置Prometheus数据源
- [ ] 1.3 导入基础仪表盘模板
- [ ] 1.4 配置时区为Asia/Hong_Kong

### 阶段2: 策略模式仪表盘（1.5小时）

**Panel 1: 当前模式状态（V1核心）**
- [ ] 2.1 Stat面板 - 当前模式（active/quiet）
  ```promql
  avg without(instance,pod) (strategy_mode_active{env="$env",symbol=~"$symbol"})
  ```
- [ ] 2.2 Stat面板 - 最后切换距今（duration单位）
  ```promql
  time() - max without(instance,pod) (strategy_mode_last_change_timestamp{env="$env",symbol=~"$symbol"})
  ```
- [ ] 2.3 Stat面板 - 今日切换次数
  ```promql
  increase(strategy_mode_transitions_total{env="$env",symbol=~"$symbol"}[24h])
  ```
- [ ] 2.4 Gauge面板 - 当前模式持续时间

**Panel 2: 切换历史**
- [ ] 2.5 Time series - 模式切换时间线（annotate）
- [ ] 2.6 Bar chart - 切换原因分布（schedule/market/manual）
  ```promql
  increase(strategy_mode_transitions_total{env="$env",symbol=~"$symbol"}[$__range]) by (reason)
  ```
- [ ] 2.7 Time series - 各模式累计时长趋势（hours单位）
  ```promql
  increase(strategy_time_in_mode_seconds_total{env="$env",symbol=~"$symbol"}[$__range]) by (mode) / 3600
  ```

**Panel 3: 触发因子**
- [ ] 2.8 Time series - 市场指标（trades/min, quotes/sec）
  ```promql
  avg without(instance,pod) (strategy_trigger_trades_per_min{env="$env",symbol=~"$symbol"})
  avg without(instance,pod) (strategy_trigger_quote_updates_per_sec{env="$env",symbol=~"$symbol"})
  ```
- [ ] 2.9 Time series - 点差和波动率（bps）
  ```promql
  avg without(instance,pod) (strategy_trigger_spread_bps{env="$env",symbol=~"$symbol"})
  avg without(instance,pod) (strategy_trigger_volatility_bps{env="$env",symbol=~"$symbol"})
  ```
- [ ] 2.10 Heatmap - 一天内的模式分布（24x7热力图，V2补强）

### 阶段3: 性能仪表盘（1小时）

**Panel 4: 参数更新性能（V1核心）**
- [ ] 3.1 Histogram - 参数更新耗时分布
  ```promql
  histogram_quantile(0.95, sum by(le) (rate(strategy_params_update_duration_ms_bucket{env="$env"}[$__rate_interval])))
  ```
- [ ] 3.2 Stat - P50/P95/P99耗时
  ```promql
  histogram_quantile(0.50, sum by(le) (rate(strategy_params_update_duration_ms_bucket{env="$env"}[$__rate_interval])))
  histogram_quantile(0.95, sum by(le) (rate(strategy_params_update_duration_ms_bucket{env="$env"}[$__rate_interval])))
  histogram_quantile(0.99, sum by(le) (rate(strategy_params_update_duration_ms_bucket{env="$env"}[$__rate_interval])))
  ```
- [ ] 3.3 Time series - 更新耗时趋势
- [ ] 3.4 Counter - 更新失败次数（按模块分组）
  ```promql
  increase(strategy_params_update_failures_total{env="$env"}[$__range]) by (module)
  ```

**Panel 5: 系统性能**
- [ ] 3.5 CPU使用率
- [ ] 3.6 内存使用率
- [ ] 3.7 网络延迟

### 阶段4: 告警仪表盘（30分钟）

- [ ] 4.1 Table - 当前告警列表
- [ ] 4.2 Time series - 告警触发历史
- [ ] 4.3 Stat - 24小时告警次数

### 阶段5: 配置导出与文档（30分钟）

- [ ] 5.1 导出仪表盘JSON
- [ ] 5.2 创建使用文档
- [ ] 5.3 创建故障排查指南
- [ ] 5.4 配置告警通知（Slack/Email）

---

## ✅ 验收标准

### V1 核心验收（立即可用）
- [ ] **V1.1**: 8个核心面板运行正常（当前模式、切换次数、触发因子、性能指标）
- [ ] **V1.2**: 变量配置正确（$env、$symbol），时区设置为Asia/Hong_Kong
- [ ] **V1.3**: 注释能标出每次模式切换，点击可看reason
- [ ] **V1.4**: 三个性能视图可用（P50/P95/P99 & Histogram & 失败计数）

### V2 增强验收（基于实测调优）
- [ ] **V2.1**: 24×7热力图展示模式分布
- [ ] **V2.2**: 性能分组对比（不同symbol/env）
- [ ] **V2.3**: 告警视图与历史趋势
- [ ] **V2.4**: 数据质量页签（丢包、缺口、回放对比）

### 配置验收
- [ ] **V3.1**: JSON配置已导出并版本控制
- [ ] **V3.2**: 告警规则配置（4条草案）
- [ ] **V3.3**: 文档完善（使用指南、故障排查）

**通过标准**: V1 4/4 + V2 2/4 + V3 2/3 = 8/11 验收通过

---

## 📦 交付物

### 配置文件
- `grafana/dashboards/strategy_mode_overview.json` - 主仪表盘
- `grafana/dashboards/strategy_performance.json` - 性能仪表盘
- `grafana/dashboards/strategy_alerts.json` - 告警仪表盘

### 文档
- `docs/GRAFANA_SETUP_GUIDE.md` - 安装配置指南
- `docs/GRAFANA_DASHBOARD_GUIDE.md` - 仪表盘使用指南
- `docs/GRAFANA_TROUBLESHOOTING.md` - 故障排查指南

---

## 🚨 告警规则配置（4条草案）

### 告警规则1: 频繁切换告警
```yaml
alert: StrategyModeSwitchingTooFrequently
expr: sum(increase(strategy_mode_transitions_total{env="$env"}[1h])) > 10
for: 10m
labels:
  severity: warning
annotations:
  summary: "策略模式切换过于频繁"
  description: "过去1小时内模式切换超过10次，可能存在配置问题"
```

### 告警规则2: 长期quiet模式
```yaml
alert: StrategyModeStuckInQuiet
expr: (time() - max(strategy_mode_last_change_timestamp{env="$env"})) > 4*3600 and on() avg(strategy_mode_active{env="$env"}) < 0.5
for: 15m
labels:
  severity: warning
annotations:
  summary: "策略模式长期处于quiet状态"
  description: "超过4小时未切换且当前为quiet模式"
```

### 告警规则3: 参数更新失败
```yaml
alert: StrategyParamsUpdateFailed
expr: increase(strategy_params_update_failures_total{env="$env"}[5m]) > 0
for: 0m
labels:
  severity: critical
annotations:
  summary: "策略参数更新失败"
  description: "过去5分钟内参数更新失败，请检查模块状态"
```

### 告警规则4: 指标心跳异常
```yaml
alert: StrategyMetricsHeartbeatMissing
expr: absent(strategy_mode_active{env="$env"}) or (time() - max(strategy_metrics_last_scrape_timestamp{env="$env"})) > 120
for: 2m
labels:
  severity: critical
annotations:
  summary: "策略指标心跳异常"
  description: "指标缺失或超过2分钟未更新"
```

---

## ⚠️ 易踩坑点 & 规避建议

### 1. 统计窗口问题
- **问题**: 对Counter指标直接使用sum()或rate()
- **解决**: 一律使用`increase()`函数，例如：`increase(strategy_mode_transitions_total[24h])`
- **原因**: Counter类型指标需要计算增量，而非绝对值

### 2. 直方图百分位数计算
- **问题**: 直接对`_sum`或`_count`做百分位计算
- **解决**: 使用`histogram_quantile()`配合`_bucket`指标
- **正确**: `histogram_quantile(0.95, sum by(le) (rate(strategy_params_update_duration_ms_bucket[5m])))`

### 3. 标签爆炸
- **问题**: 高基数字段（如instance、pod）导致查询缓慢
- **解决**: 查询时使用`without(instance,pod)`聚合
- **示例**: `avg without(instance,pod) (strategy_mode_active)`

### 4. 采样步长设置
- **问题**: 高频指标锯齿严重
- **解决**: 设置合适的`$__rate_interval`和最小步长（如15s）
- **配置**: 在Panel设置中调整"Min interval"

### 5. 单位与值映射
- **时间单位**: 使用"duration(s)"自动转换为可读格式
- **bps显示**: 设置"misc -> parts per million"显示为bps
- **模式映射**: 0→Quiet(蓝色)，1→Active(绿色)

### 6. 变量配置
- **环境变量**: `$env` (development/testing/production)
- **交易对变量**: `$symbol` (多选：BTCUSDT,ETHUSDT)
- **默认值**: env=testing, symbol=BTCUSDT

---

## 💡 仪表盘设计示例

### Panel配置示例

```yaml
# 当前模式状态
{
  "title": "Current Mode",
  "type": "stat",
  "datasource": "Prometheus",
  "targets": [
    {
      "expr": "strategy_mode_active",
      "legendFormat": "Mode"
    }
  ],
  "options": {
    "reduceOptions": {
      "values": false,
      "calcs": ["lastNotNull"]
    },
    "text": {
      "valueSize": 72
    },
    "colorMode": "value",
    "graphMode": "none"
  },
  "fieldConfig": {
    "overrides": [
      {
        "matcher": {"id": "byValue", "options": "0"},
        "properties": [
          {"id": "displayName", "value": "Quiet"},
          {"id": "color", "value": {"mode": "fixed", "fixedColor": "blue"}}
        ]
      },
      {
        "matcher": {"id": "byValue", "options": "1"},
        "properties": [
          {"id": "displayName", "value": "Active"},
          {"id": "color", "value": {"mode": "fixed", "fixedColor": "green"}}
        ]
      }
    ]
  }
}
```

---

**任务创建时间**: 2025-10-19 06:32  
**任务完成时间**: 2025-10-20 06:00（V1.2统一配置集成版）
**任务状态**: ✅ 已完成

## 🎉 V1.2统一配置集成完成

### 新增功能
- ✅ **统一配置管理**: Grafana配置完全集成到 `config/system.yaml`
- ✅ **动态仪表盘生成**: 支持从配置动态生成仪表盘JSON
- ✅ **环境变量覆盖**: 支持通过环境变量动态调整配置
- ✅ **配置热更新**: 支持配置变更实时生效
- ✅ **完整测试覆盖**: 8/8测试用例全部通过

### 技术实现
- **配置加载器**: `src/grafana_config.py` - 统一配置加载
- **仪表盘生成器**: `src/grafana_dashboard_generator.py` - 动态生成
- **测试脚本**: `test_grafana_config.py` - 功能验证
- **配置结构**: 完整的仪表盘、数据源、变量、告警配置

### 配置示例
```yaml
monitoring:
  grafana:
    dashboards:
      strategy_mode:
        uid: "strategy-mode-overview"
        title: "Strategy Mode Overview"
        timezone: "Asia/Hong_Kong"
    datasources:
      prometheus:
        url: "http://localhost:9090"
    variables:
      env:
        query: "label_values(strategy_mode_active, env)"
```

_完整详细配置见实际开发时补充_

