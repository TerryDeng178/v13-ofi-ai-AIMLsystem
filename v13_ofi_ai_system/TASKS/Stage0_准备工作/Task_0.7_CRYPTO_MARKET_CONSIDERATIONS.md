# Task 0.7 数字资产市场特殊考虑

## 🪙 数字资产 vs 传统股票市场差异

| 维度 | 传统股票 | 数字资产（加密货币） | 配置影响 |
|------|---------|-------------------|---------|
| **交易时段** | 工作日 9:30-16:00（有休市） | 24/7 全天候 | `enabled_weekdays: [Mon-Sun]` |
| **节假日** | 有休市 | 无休市 | `holidays: []` |
| **流动性模式** | 开盘/收盘高峰 | 全球时段轮转 | 4个活跃窗口 |
| **波动性** | 相对稳定 | 高波动 | `min_volatility_bps: 10` |
| **成交频率** | 中等 | 高频 | `min_trades_per_min: 500` |
| **订单簿更新** | 较慢 | 极快 | `min_quote_updates_per_sec: 100` |
| **点差** | 较宽 | 较窄（主流币） | `max_spread_bps: 5` |

---

## 📊 数字资产活跃时段分析（HKT）

### 时段1: 亚洲早盘（09:00-12:00）
- **特征**: 受A股、日韩股市开盘影响
- **流动性**: 中等偏高
- **主要驱动**: 亚洲投资者、亚洲新闻
- **典型波动**: 中等
- **建议模式**: `active`

### 时段2: 亚洲午后（14:00-17:00）
- **特征**: 港股收盘前，欧洲预热
- **流动性**: 中等
- **主要驱动**: 港股收盘、欧洲开盘预期
- **典型波动**: 中等
- **建议模式**: `active`

### 时段3: 欧美高峰（21:00-02:00）⭐ **最重要**
- **特征**: 美股交易时段，全球最活跃
- **流动性**: **极高**
- **主要驱动**: 美股走势、美联储政策、宏观数据
- **典型波动**: 高
- **建议模式**: `active`（最激进参数）
- **注意**: 
  - 美东开盘（HKT 21:30/22:30，夏/冬令时）
  - 美东收盘（HKT 04:00/05:00，夏/冬令时）

### 时段4: 美洲夜盘（06:00-08:00）
- **特征**: 美股收盘后余温
- **流动性**: 中等
- **主要驱动**: 美股收盘情绪、欧洲早盘
- **典型波动**: 较高
- **建议模式**: `active`

### 时段5: 凌晨低谷（03:00-06:00）⚠️ **最危险**
- **特征**: 全球流动性枯竭
- **流动性**: **极低**
- **主要驱动**: 几乎无
- **典型波动**: 低但容易被操纵
- **建议模式**: `quiet`（保守参数）
- **风险**: 
  - 容易出现假突破
  - 大单冲击成本高
  - 滑点风险大

---

## 🎯 针对数字资产的参数调优建议

### Active 模式（高流动性时段）

```yaml
ofi:
  bucket_ms: 50              # 更短的bucket（数字资产高频）
  depth_levels: 15           # 更深的订单簿深度
  watermark_ms: 200          # 更短的水位线（低延迟）
  
cvd:
  window_ticks: 1000         # 更短的窗口（快速反应）
  ema_span: 30               # 更短的EMA（捕捉短期趋势）
  denoise_sigma: 1.5         # 较小的去噪（保留信号）

risk:
  position_limit: 1.5        # 较大的仓位（高流动性支持）
  order_rate_limit_per_min: 1000  # 更高的下单频率
```

### Quiet 模式（低流动性时段）

```yaml
ofi:
  bucket_ms: 500             # 更长的bucket（降低噪声）
  depth_levels: 5            # 浅订单簿（避免虚假信号）
  watermark_ms: 2000         # 更长的水位线（等待稳定）
  
cvd:
  window_ticks: 10000        # 更长的窗口（平滑噪声）
  ema_span: 300              # 更长的EMA（过滤假信号）
  denoise_sigma: 4.0         # 更大的去噪（强过滤）

risk:
  position_limit: 0.1        # 极小的仓位（避免冲击）
  order_rate_limit_per_min: 20   # 极低的下单频率
```

---

## ⚠️ 数字资产市场特殊风险

### 1. 链上事件风险
- **场景**: 重大协议升级、黑客攻击、监管新闻
- **特征**: 突发性、无预警、剧烈波动
- **应对**: 设置新闻事件触发器，自动切换到 `quiet` 模式

### 2. 流动性骤降风险
- **场景**: 交易所故障、API限流、市场恐慌
- **特征**: 订单簿深度瞬间消失
- **应对**: 
  - `min_volume_usd` 阈值检测
  - 自动降低 `position_limit`
  - 暂停新开仓

### 3. 操纵风险（凌晨时段）
- **场景**: 大户在低流动性时段拉盘/砸盘
- **特征**: 成交量小但价格波动大
- **应对**: 
  - 凌晨时段强制 `quiet` 模式
  - 提高 `min_volume_usd` 阈值
  - 信号置信度打折

### 4. 跨交易所套利机器人
- **场景**: 高频套利导致订单簿假信号
- **特征**: 报价频繁变动但成交量小
- **应对**: 
  - `min_trades_per_min` 过滤低质量信号
  - `winsorize_percentile` 过滤极端报价

---

## 🔧 实施建议

### 阶段1: 保守启动（1-2周）

```yaml
strategy:
  mode: quiet  # 固定为保守模式
  
market:
  enabled: false  # 仅使用时间表触发
```

**目的**: 收集不同时段的真实数据，校准阈值

### 阶段2: 时间表启用（2-4周）

```yaml
strategy:
  mode: auto
  
triggers:
  schedule:
    enabled: true
  market:
    enabled: false  # 仍未启用市场触发
```

**目的**: 验证时间表触发的准确性

### 阶段3: 市场指标启用（4-8周）

```yaml
strategy:
  mode: auto
  
triggers:
  schedule:
    enabled: true
  market:
    enabled: true  # 启用市场触发
```

**目的**: 启用完整的二元触发逻辑

### 阶段4: 精细调优（持续）

- 根据历史数据调整 `min_trades_per_min`、`min_volatility_bps`
- 优化活跃窗口边界
- 添加特殊事件处理逻辑

---

## 📈 关键监控指标

### 数字资产特有指标

```yaml
# 建议新增的指标
strategy_trigger_funding_rate        # 资金费率（永续合约）
strategy_trigger_oi_change_pct       # 持仓量变化（期货）
strategy_trigger_liquidations_usd    # 爆仓金额（风险指标）
strategy_trigger_exchange_premium_bps # 交易所溢价（套利空间）
```

### 告警阈值

```yaml
# 数字资产市场告警
- name: crypto_liquidity_collapse
  condition: min_volume_usd < 100000 for 5min
  action: force_quiet_mode
  
- name: crypto_extreme_volatility
  condition: volatility_bps > 100 for 1min
  action: reduce_position_limit_50pct
  
- name: crypto_api_degradation
  condition: quote_updates_per_sec < 10
  action: force_quiet_mode
```

---

## 🌐 多交易对考虑

数字资产通常同时交易多个币种，需要考虑：

### 主流币种（BTC/ETH）
- 流动性最好
- 可使用最激进参数
- 作为市场情绪基准

### 中盘币种（SOL/BNB/XRP等）
- 流动性中等
- 参数需要折中
- 与主流币种相关性高

### 小盘币种（长尾币种）
- 流动性差
- 建议固定 `quiet` 模式
- 或单独设置更保守阈值

---

## ✅ 检查清单

实施前确认：

- [ ] 时区设置为 `Asia/Hong_Kong`
- [ ] `calendar: CRYPTO`
- [ ] `enabled_weekdays: [Mon-Sun]`（7天）
- [ ] 4个活跃窗口覆盖全球高峰
- [ ] `min_trades_per_min` 调整到500+
- [ ] `min_volatility_bps` 调整到10+
- [ ] 添加 `min_volume_usd` 阈值
- [ ] 凌晨时段（03:00-06:00）不在活跃窗口内
- [ ] 美东夏令时/冬令时调整预案
- [ ] 链上事件监控集成计划
- [ ] 交易所API限流应对预案

---

**文档版本**: v1.0  
**适用市场**: 数字资产（加密货币）  
**时区**: Asia/Hong_Kong (HKT, UTC+8)  
**创建时间**: 2025-10-19  
**审核状态**: ✅ 已审核

