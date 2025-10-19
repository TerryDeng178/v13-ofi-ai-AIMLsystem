# Grafana仪表盘使用指南

## 📋 概述

本指南介绍如何使用V13策略模式切换系统的Grafana监控仪表盘。

## 🚀 快速开始

### 1. 导入仪表盘

```bash
# 方法1: 通过Grafana UI导入
1. 登录Grafana (http://localhost:3000)
2. 点击 "+" -> "Import"
3. 上传以下JSON文件：
   - grafana/dashboards/strategy_mode_overview.json
   - grafana/dashboards/strategy_performance.json
   - grafana/dashboards/strategy_alerts.json

# 方法2: 通过Provisioning自动导入
# 将JSON文件放入Grafana的provisioning目录
```

### 2. 配置数据源

确保Prometheus数据源已配置：
- **名称**: Prometheus
- **URL**: http://localhost:9090
- **Access**: Server (default)

### 3. 设置变量

仪表盘包含以下变量：

| 变量名 | 类型 | 说明 | 默认值 |
|--------|------|------|--------|
| `$env` | Query | 环境选择 | testing |
| `$symbol` | Query | 交易对选择 | BTCUSDT |

## 📊 仪表盘说明

### 1. Strategy Mode Overview（策略模式概览）

**核心面板**：

#### 当前模式状态
- **Current Mode**: 显示当前策略模式（Active/Quiet）
- **Last Switch Ago**: 距离上次切换的时间
- **Switches Today**: 今日切换次数
- **Switch Reason Distribution**: 切换原因分布

#### 触发因子监控
- **Market Triggers**: 市场指标（成交笔数/分钟、报价更新/秒）
- **Spread & Volatility**: 点差和波动率（bps）
- **Volume USD**: 成交量（美元）

#### 模式时长趋势
- **Mode Duration Trend**: 各模式累计时长（小时）

**关键PromQL查询**：
```promql
# 当前模式
avg without(instance,pod) (strategy_mode_active{env="$env",symbol=~"$symbol"})

# 今日切换次数
increase(strategy_mode_transitions_total{env="$env",symbol=~"$symbol"}[24h])

# 切换原因分布
increase(strategy_mode_transitions_total{env="$env",symbol=~"$symbol"}[$__range]) by (reason)
```

### 2. Strategy Performance（策略性能）

**性能指标**：

#### 参数更新耗时
- **P50/P95/P99**: 参数更新耗时百分位数
- **Duration Trend**: 耗时趋势图
- **Duration Histogram**: 耗时分布直方图

#### 更新失败监控
- **Update Failures**: 总失败次数
- **Failures by Module**: 按模块分组的失败次数

**关键PromQL查询**：
```promql
# P95耗时
histogram_quantile(0.95, sum by(le) (rate(strategy_params_update_duration_ms_bucket{env="$env"}[$__rate_interval])))

# 失败次数
increase(strategy_params_update_failures_total{env="$env"}[$__range]) by (module)
```

### 3. Strategy Alerts（策略告警）

**告警监控**：

#### 实时告警状态
- **Frequent Switching Alert**: 频繁切换告警
- **Long Quiet Period Alert**: 长期quiet告警
- **Parameter Update Failures**: 参数更新失败告警
- **Metrics Heartbeat Status**: 指标心跳状态

#### 告警趋势
- **Alert Trends**: 告警触发趋势
- **Alert History**: 告警历史日志

## 🎯 使用场景

### 场景1: 24小时长测监控

1. **设置时间范围**: 选择24小时时间窗口
2. **关注核心指标**:
   - 当前模式状态
   - 切换频率（应≤10次/小时）
   - 参数更新性能（P95≤100ms）
   - 失败次数（应为0）

3. **关键告警**:
   - 频繁切换告警
   - 长期quiet告警
   - 参数更新失败告警

### 场景2: 性能调优

1. **性能分析**:
   - 查看参数更新耗时分布
   - 识别性能瓶颈（P95>100ms）
   - 分析失败模式

2. **优化验证**:
   - 对比优化前后的性能指标
   - 验证优化效果

### 场景3: 故障排查

1. **问题定位**:
   - 查看告警历史
   - 分析触发因子异常
   - 检查模式切换原因

2. **根因分析**:
   - 时间线分析
   - 相关性分析

## ⚠️ 注意事项

### 1. 数据源配置
- 确保Prometheus正确配置并运行
- 验证指标名称与查询匹配
- 检查标签维度一致性

### 2. 时间范围设置
- 长测期间建议设置24小时窗口
- 性能分析建议设置1小时窗口
- 故障排查建议设置6小时窗口

### 3. 变量使用
- 环境变量`$env`用于区分不同环境
- 交易对变量`$symbol`支持多选
- 合理设置默认值

### 4. 告警配置
- 告警规则需要配置通知渠道
- 建议设置Slack或邮件通知
- 定期检查告警规则有效性

## 🔧 故障排查

### 问题1: 面板显示"No Data"
**可能原因**：
- Prometheus数据源未配置
- 指标名称错误
- 时间范围设置不当

**解决方案**：
1. 检查Prometheus连接状态
2. 验证指标名称：`strategy_mode_active`
3. 调整时间范围

### 问题2: 变量不生效
**可能原因**：
- 变量查询语法错误
- 标签值不存在

**解决方案**：
1. 检查变量查询语法
2. 验证标签值：`label_values(strategy_mode_active, env)`

### 问题3: 告警不触发
**可能原因**：
- 告警规则配置错误
- 阈值设置不当
- 数据不足

**解决方案**：
1. 验证告警规则语法
2. 调整告警阈值
3. 确保数据充足

## 📞 技术支持

如遇到问题，请：
1. 检查本文档的故障排查部分
2. 查看Grafana日志
3. 联系技术支持团队

## 🔧 V1.1 修复说明

### 已修复的关键问题

#### 1. 聚合口径优化
- **当前模式**: 改用`max without(instance,pod,symbol)`避免多选symbol时的"半激活"值
- **切换次数**: 统一使用`sum without(instance,pod,symbol)`聚合，避免Stat面板多值异常
- **心跳监控**: 改用`strategy_metrics_last_scrape_timestamp`监控指标更新而非切换频率

#### 2. 单位显示修复
- **bps显示**: 从ppm改为"short + bps后缀"，避免单位混淆（1 bps = 1e-4，1 ppm = 1e-6）
- **时间范围**: Overview仪表盘默认时间范围从1小时调整为6小时，更利于观测节律

#### 3. 注释功能增强
- **颜色区分**: 模式切换注释按原因分色显示（schedule=蓝色，market=橙色，manual=绿色）
- **交互优化**: 点击注释可查看详细切换信息

#### 4. 告警规则完善
- **标签统一**: 所有告警规则添加`env: testing`标签，便于多环境筛选
- **阈值优化**: 心跳异常告警使用正确的指标监控

### 验证清单

使用以下PromQL验证修复效果：

```promql
# 验证当前模式聚合
max without(instance,pod,symbol) (strategy_mode_active{env="testing",symbol=~"BTCUSDT|ETHUSDT"})

# 验证切换次数聚合
sum without(instance,pod,symbol) (increase(strategy_mode_transitions_total{env="testing",symbol=~"BTCUSDT|ETHUSDT"}[24h]))

# 验证心跳监控
time() - max without(instance,pod) (strategy_metrics_last_scrape_timestamp{env="testing"})
```

---

**文档版本**: V1.1  
**最后更新**: 2025-10-19  
**维护者**: V13 Team
