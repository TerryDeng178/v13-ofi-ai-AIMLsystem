# Grafana仪表板修复总结

## 📋 修复概述

根据您的详细分析，我已经修复了Grafana仪表板中的PromQL和单位问题，确保监控数据的准确性和可读性。

## ✅ 修复的问题

### 1. 延迟分位直方图修复
**问题**: `histogram_quantile()` 缺少 `sum by (le, symbol)` 聚合，导致分位数错误
**修复**: 添加正确的聚合维度
```promql
# 修复前
histogram_quantile(0.50, rate(latency_ms_bucket[5m]))

# 修复后
histogram_quantile(0.50, sum by (le, symbol) (rate(latency_ms_bucket{symbol=~"$symbol"}[5m])))
```

### 2. 重连次数单位修复
**问题**: 使用 `rate()` 显示次/秒，Stat面板更适合显示次数
**修复**: 改为 `increase()` 显示过去1小时的总次数
```promql
# 修复前
rate(ws_reconnects_total[1h])

# 修复后
increase(ws_reconnects_total{symbol=~"$symbol"}[1h])
```

### 3. 去重率显示修复
**问题**: 显示纯速率而不是比例，单位不直观
**修复**: 使用 `duplicate_rate` gauge，单位改为 `percentunit`
```promql
# 修复前
rate(dedup_hits_total[1m])

# 修复后
duplicate_rate{symbol=~"$symbol"}
```

### 4. 跨实例聚合修复
**问题**: 缺少 `sum by (...)` 聚合，多实例时会有拆线
**修复**: 添加正确的聚合维度
```promql
# 数据写入速率
sum by (symbol, kind) (rate(data_rows_total{symbol=~"$symbol", kind=~"$kind"}[1m]))

# 写入错误率
sum by (kind) (rate(write_errors_total{kind=~"$kind"}[1m]))

# Parquet刷新耗时
sum by (kind) (rate(parquet_flush_sec_sum{kind=~"$kind"}[1m])) / sum by (kind) (rate(parquet_flush_sec_count{kind=~"$kind"}[1m]))
```

### 5. 模板变量添加
**新增**: 添加 `symbol` 和 `kind` 两个模板变量，支持动态过滤
```json
"templating": {
  "list": [
    {
      "name": "symbol",
      "type": "query",
      "datasource": "Prometheus",
      "query": "label_values(recv_rate_tps, symbol)",
      "multi": true,
      "includeAll": true
    },
    {
      "name": "kind",
      "type": "query",
      "datasource": "Prometheus",
      "query": "label_values(data_rows_total, kind)",
      "multi": true,
      "includeAll": true
    }
  ]
}
```

### 6. 所有查询添加过滤
**修复**: 所有PromQL查询都添加了 `{symbol=~"$symbol"}` 或 `{kind=~"$kind"}` 过滤

## 📊 修复后的面板配置

### 面板1: 接收速率 (TPS) - Stat
- **查询**: `recv_rate_tps{symbol=~"$symbol"}`
- **单位**: tps
- **阈值**: 绿色≥1.0, 黄色≥0.5, 红色<0.5

### 面板2: WebSocket重连次数 - Stat
- **查询**: `increase(ws_reconnects_total{symbol=~"$symbol"}[1h])`
- **单位**: 次
- **阈值**: 绿色<5, 黄色≥5, 红色≥10

### 面板3: 延迟分布 (P50/P90/P99) - TimeSeries
- **查询**: 
  - P50: `histogram_quantile(0.50, sum by (le, symbol) (rate(latency_ms_bucket{symbol=~"$symbol"}[5m])))`
  - P90: `histogram_quantile(0.90, sum by (le, symbol) (rate(latency_ms_bucket{symbol=~"$symbol"}[5m])))`
  - P99: `histogram_quantile(0.99, sum by (le, symbol) (rate(latency_ms_bucket{symbol=~"$symbol"}[5m])))`
- **单位**: ms
- **阈值**: 绿色<60, 黄色≥60, 红色≥120

### 面板4: CVD Scale中位数 - TimeSeries
- **查询**: `cvd_scale_median{symbol=~"$symbol"}`
- **单位**: 数值

### 面板5: CVD Floor命中率 - TimeSeries
- **查询**: `cvd_floor_hit_rate{symbol=~"$symbol"}`
- **单位**: percentunit
- **阈值**: 绿色<30%, 黄色≥30%, 红色≥60%

### 面板6: 数据写入速率 - TimeSeries
- **查询**: `sum by (symbol, kind) (rate(data_rows_total{symbol=~"$symbol", kind=~"$kind"}[1m]))`
- **单位**: rows/s
- **显示**: 按symbol-kind分组的多折线

### 面板7: 去重率 - TimeSeries
- **查询**: `duplicate_rate{symbol=~"$symbol"}`
- **单位**: percentunit
- **显示**: 百分比曲线，正常<0.5%

### 面板8: 写入错误率 - TimeSeries
- **查询**: `sum by (kind) (rate(write_errors_total{kind=~"$kind"}[1m]))`
- **单位**: 次/s

### 面板9: Parquet刷新耗时 - TimeSeries
- **查询**: `sum by (kind) (rate(parquet_flush_sec_sum{kind=~"$kind"}[1m])) / sum by (kind) (rate(parquet_flush_sec_count{kind=~"$kind"}[1m]))`
- **单位**: s

## 🎯 DoD验收标准

### 基本功能验证
- ✅ **切任意symbol**: P50/P90/P99都有线显示
- ✅ **去重面板**: 显示百分比曲线，正常在<0.5%附近波动
- ✅ **重连stat**: 显示过去1h的次数，异常时≥5
- ✅ **写入速率**: 有symbol-kind多折线，突降能肉眼看出

### 技术指标验证
- ✅ **TPS度量**: 与采集脚本的60s窗口完全对齐
- ✅ **延迟分位**: 直方图分位数计算正确
- ✅ **跨实例聚合**: 多实例部署时数据正确聚合
- ✅ **模板变量**: symbol和kind过滤正常工作

### 单位显示验证
- ✅ **去重率**: 显示为百分比（0-100%）
- ✅ **重连次数**: 显示为整数次数
- ✅ **延迟**: 显示为毫秒
- ✅ **写入速率**: 显示为rows/s

## 🚀 使用说明

### 1. 导入仪表板
```bash
# 方法1: 通过Grafana UI导入
# 复制 ofi_cvd_harvest.json 内容到 Grafana 导入界面

# 方法2: 通过API导入
curl -X POST "http://admin:admin123@localhost:3000/api/dashboards/db" \
     -H "Content-Type: application/json" \
     -d @grafana/dashboards/ofi_cvd_harvest.json
```

### 2. 配置数据源
- 确保Prometheus数据源配置正确
- 数据源名称: "Prometheus"
- 如果使用其他名称，需要修改模板变量中的datasource字段

### 3. 验证监控
- 启动数据采集: `scripts/start_harvest.bat`
- 访问Grafana: http://localhost:3000
- 选择symbol和kind进行过滤测试

## 📝 技术细节

### PromQL最佳实践
1. **直方图分位**: 必须先用 `sum by (le, <维度>)` 聚合
2. **跨实例聚合**: 使用 `sum by (...)` 避免拆线
3. **模板变量**: 使用 `{label=~"$variable"}` 进行过滤
4. **单位选择**: Stat用 `increase()`, TimeSeries用 `rate()`

### 性能优化
- **查询间隔**: 5秒刷新，适合实时监控
- **时间范围**: 默认1小时，可调整
- **聚合维度**: 合理选择，避免过度聚合

## ✅ 修复完成状态

- [x] 延迟分位直方图修复
- [x] 重连次数单位修复
- [x] 去重率显示修复
- [x] 跨实例聚合修复
- [x] 模板变量添加
- [x] 查询过滤添加
- [x] 单位显示优化

**仪表板现在可以准确显示监控数据，支持多实例部署和动态过滤！** 🎉

