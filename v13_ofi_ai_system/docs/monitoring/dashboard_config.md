# CVD监控仪表盘配置

## 📊 核心监控卡片

### 1. Z-score质量监控
```yaml
card_name: "Z-score质量"
metrics:
  - name: "P(|Z|>2)"
    target: "≤8%"
    current: "5.73%"
    status: "✅ 达标"
    alert_thresholds:
      yellow: 10%
      red: 15%
  
  - name: "P(|Z|>3)"
    target: "≤2%"
    current: "4.65%"
    status: "🎯 优化中"
    alert_thresholds:
      yellow: 5%
      red: 8%
  
  - name: "median(|Z|)"
    target: "≤1.0"
    current: "0.0013"
    status: "✅ 优秀"
```

### 2. 尺度分母健康监控
```yaml
card_name: "尺度分母健康"
metrics:
  - name: "scale_p5"
    description: "尺度5%分位数"
    trend: "监控地板是否过低"
  
  - name: "scale_p50"
    description: "尺度中位数"
    trend: "监控正常波动范围"
  
  - name: "scale_p95"
    description: "尺度95%分位数"
    trend: "监控极端波动"
```

### 3. 空窗后Z分布监控
```yaml
card_name: "空窗后Z分布"
metrics:
  - name: "post_stale_3trades_z"
    description: "空窗后首3笔|Z|分布"
    purpose: "验证冻结效果"
    chart_type: "histogram"
```

### 4. 到达节奏监控
```yaml
card_name: "到达节奏"
metrics:
  - name: "p99_interarrival"
    target: "≤5s"
    current: "4.2s"
    status: "✅ 正常"
  
  - name: "gaps_over_10s"
    target: "=0"
    current: "0"
    status: "✅ 优秀"
```

## 🚨 告警配置

### 实时告警规则
```yaml
alerts:
  - name: "Z-score尾部异常"
    condition: "p_gt3 > 8%"
    severity: "critical"
    action: "立即检查数据源和参数"
  
  - name: "Z-score尾部警告"
    condition: "p_gt3 > 5%"
    severity: "warning"
    action: "关注趋势，准备调参"
  
  - name: "尺度地板过低"
    condition: "scale_p5 < 0.1"
    severity: "warning"
    action: "检查MAD_MULTIPLIER设置"
  
  - name: "数据完整性异常"
    condition: "parse_errors > 0 OR queue_dropped_rate > 0%"
    severity: "critical"
    action: "立即检查网络和解析逻辑"
```

## 📈 趋势分析

### 关键指标趋势
- **P(|Z|>2)**: 目标保持≤8%，当前5.73% ✅
- **P(|Z|>3)**: 目标≤2%，当前4.65%，持续优化中
- **scale分布**: 监控分母稳定性
- **空窗冻结效果**: 验证软冻结逻辑有效性

### 历史对比
- Step 1.3 → Step 1.6: P(|Z|>3) 从9.87% → 4.65% (改善53%)
- Step 1.3 → Step 1.6: P95(|Z|) 从8.0 → 2.71 (改善66%)
