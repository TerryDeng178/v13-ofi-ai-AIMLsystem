# Grafana仪表盘快速验收清单

## ✅ V1 核心验收（立即可用）

### V1.1: 8个核心面板运行正常

- [ ] **当前模式面板**: 显示Active/Quiet状态，颜色映射正确
- [ ] **最后切换时间**: 显示距离上次切换的时长（duration单位）
- [ ] **今日切换次数**: 显示24小时内的切换次数
- [ ] **切换原因分布**: 显示schedule/market/manual分布
- [ ] **模式时长趋势**: 显示active/quiet累计时长（hours单位）
- [ ] **市场触发因子**: 显示trades/min、quotes/sec
- [ ] **点差波动率**: 显示spread_bps、volatility_bps
- [ ] **成交量**: 显示volume_usd

**验证方法**: 导入仪表盘后，检查每个面板是否有数据且显示正常

### V1.2: 变量配置正确

- [ ] **$env变量**: 可选择development/testing/production
- [ ] **$symbol变量**: 可选择BTCUSDT/ETHUSDT等交易对
- [ ] **默认值**: env=testing, symbol=BTCUSDT
- [ ] **时区设置**: Asia/Hong_Kong

**验证方法**: 点击仪表盘右上角的变量下拉框，确认选项和默认值

### V1.3: 注释功能正常

- [ ] **模式切换注释**: 时间线上显示切换点
- [ ] **注释内容**: 显示切换原因（schedule/market/manual）
- [ ] **注释交互**: 点击注释可查看详细信息

**验证方法**: 触发一次模式切换，检查时间线是否出现注释点

### V1.4: 性能视图可用

- [ ] **P50/P95/P99**: 显示参数更新耗时的百分位数
- [ ] **Histogram**: 显示耗时分布直方图
- [ ] **失败计数**: 显示更新失败次数（按模块分组）

**验证方法**: 触发参数更新，检查性能指标是否正常显示

## 🚨 告警规则验收

### 告警规则1: 频繁切换告警
- [ ] **规则表达式**: `sum(increase(strategy_mode_transitions_total{env="testing"}[1h])) > 10`
- [ ] **持续时间**: 10分钟
- [ ] **严重级别**: Warning

### 告警规则2: 长期quiet告警
- [ ] **规则表达式**: `(time() - max(strategy_mode_last_change_timestamp{env="testing"})) > 4*3600 and on() avg(strategy_mode_active{env="testing"}) < 0.5`
- [ ] **持续时间**: 15分钟
- [ ] **严重级别**: Warning

### 告警规则3: 参数更新失败告警
- [ ] **规则表达式**: `increase(strategy_params_update_failures_total{env="testing"}[5m]) > 0`
- [ ] **持续时间**: 0分钟（立即触发）
- [ ] **严重级别**: Critical

### 告警规则4: 指标心跳异常告警
- [ ] **规则表达式**: `absent(strategy_mode_active{env="testing"}) or (time() - max(strategy_mode_last_change_timestamp{env="testing"})) > 120`
- [ ] **持续时间**: 2分钟
- [ ] **严重级别**: Critical

## 📊 数据验证

### 指标存在性检查
```bash
# 检查核心指标是否存在
curl -G 'http://localhost:9090/api/v1/query' --data-urlencode 'query=strategy_mode_active'
curl -G 'http://localhost:9090/api/v1/query' --data-urlencode 'query=strategy_mode_transitions_total'
curl -G 'http://localhost:9090/api/v1/query' --data-urlencode 'query=strategy_params_update_duration_ms_bucket'
```

### 标签维度检查
```bash
# 检查标签维度
curl -G 'http://localhost:9090/api/v1/label/__name__/values'
curl -G 'http://localhost:9090/api/v1/label/env/values'
curl -G 'http://localhost:9090/api/v1/label/symbol/values'
```

## 🔧 配置验收

### JSON配置完整性
- [ ] **strategy_mode_overview.json**: 主仪表盘配置完整
- [ ] **strategy_performance.json**: 性能仪表盘配置完整
- [ ] **strategy_alerts.json**: 告警仪表盘配置完整
- [ ] **strategy_alerts.yaml**: 告警规则配置完整

### 文档完整性
- [ ] **GRAFANA_DASHBOARD_GUIDE.md**: 使用指南完整
- [ ] **GRAFANA_QUICK_CHECKLIST.md**: 验收清单完整
- [ ] **故障排查指南**: 包含常见问题解决方案

## 🎯 24小时长测准备

### 长测前检查
- [ ] 所有面板显示正常
- [ ] 告警规则已配置
- [ ] 通知渠道已设置
- [ ] 时间范围设置为24小时
- [ ] 刷新间隔设置为30秒

### 长测期间监控
- [ ] 切换频率监控（应≤10次/小时）
- [ ] 性能指标监控（P95≤100ms）
- [ ] 失败次数监控（应为0）
- [ ] 告警状态监控

### 长测后分析
- [ ] 导出仪表盘数据
- [ ] 分析性能趋势
- [ ] 检查告警历史
- [ ] 生成测试报告

## 📝 验收记录

### 验收人员
- **姓名**: ___________
- **日期**: ___________
- **环境**: ___________

### 验收结果
- **V1.1**: ✅/❌ (8个核心面板)
- **V1.2**: ✅/❌ (变量配置)
- **V1.3**: ✅/❌ (注释功能)
- **V1.4**: ✅/❌ (性能视图)
- **告警规则**: ✅/❌ (4条规则)
- **配置完整性**: ✅/❌ (JSON+文档)

### 问题记录
```
问题1: ________________
解决方案: ______________

问题2: ________________
解决方案: ______________
```

### 验收结论
- [ ] **通过**: 所有V1验收项目通过，可用于24小时长测
- [ ] **有条件通过**: 部分项目通过，需要修复后重新验收
- [ ] **不通过**: 需要重新配置和测试

---

## ✅ V1.1 修复验收

### 关键修复验证
- [ ] **当前模式聚合**: 多选symbol时不再出现0.5等"半激活"值
- [ ] **切换次数聚合**: Stat面板显示单值，不再出现多值异常
- [ ] **bps单位显示**: 点差和波动率显示为"X.X bps"而非"X.X ppm"
- [ ] **心跳监控**: 使用正确的scrape时间戳监控指标更新
- [ ] **注释颜色**: 模式切换注释按原因分色显示
- [ ] **时间范围**: Overview仪表盘默认6小时窗口

### 验证PromQL
```bash
# 测试当前模式聚合
max without(instance,pod,symbol) (strategy_mode_active{env="testing",symbol=~"BTCUSDT|ETHUSDT"})

# 测试切换次数聚合  
sum without(instance,pod,symbol) (increase(strategy_mode_transitions_total{env="testing",symbol=~"BTCUSDT|ETHUSDT"}[24h]))

# 测试心跳监控
time() - max without(instance,pod) (strategy_metrics_last_scrape_timestamp{env="testing"})
```

**验收标准**: V1 4/4 + V1.1 6/6 + 告警规则 4/4 + 配置 2/2 = 16/16 验收通过
