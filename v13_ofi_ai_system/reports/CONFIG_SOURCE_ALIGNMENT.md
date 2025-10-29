# 配置真源对齐说明

## 概述

本文档说明配置系统中的"单一真源"原则，以及当前存在的一些多真源冲突点及修复建议。

---

## ✅ 已对齐的配置（单一真源）

### 日志配置
- **真源：** `logging.level`
- **使用位置：** 所有组件从统一配置加载器获取

### 数据源配置
- **真源：** `data_source.default_symbol`, `data_source.websocket.connection.base_url`
- **使用位置：** binance_trade_stream, websocket 客户端等

---

## ⚠️ 需要收敛的配置（多真源冲突）

### 1. Fusion 阈值配置

**现状：**
- `components.fusion.thresholds.*` (在 defaults.yaml 中定义)
  - `fuse_buy: 1.0`
  - `fuse_strong_buy: 2.3`
- `fusion_metrics.thresholds.*` (在 system.yaml 中定义) ✅ **当前使用的真源**
  - `fuse_buy: 0.95`
  - `fuse_strong_buy: 1.70`

**建议：**
- **统一真源：** `fusion_metrics.thresholds.*`
- **操作：** 移除 `components.fusion.thresholds` 或将其标记为废弃，所有代码.读取 `fusion_metrics.thresholds.*`
- **验证脚本：** 当前 `runtime_validation.py` 已使用 `fusion_metrics.thresholds.*`

---

### 2. Strategy 市场阈值配置

**现状：**
- `components.strategy.triggers.market.min_trades_per_min: 100.0` (在 defaults.yaml 中定义)
- `strategy_mode.triggers.market.min_trades_per_min: 60` (在 system.yaml 中定义) ✅ **当前使用的真源**

**建议：**
- **统一真源：** `strategy_mode.triggers.market.*`
- **操作：** 移除 `components.strategy.triggers.market.*` 或将其标记为废弃
- **验证脚本：** 当前 `runtime_validation.py` 已使用 `strategy_mode.triggers.market.*`

---

## 📋 配置键命名规范

为确保单一真源，所有配置键应遵循以下命名规范：

### 顶层键命名
- `fusion_metrics.*` - Fusion 相关配置（单一真源）
- `strategy_mode.*` - 策略模式配置（单一真源）
- `components.*` - 组件配置（应逐步迁移到对应顶层键）
- `logging.*` - 日志配置（单一真源）
- `monitoring.*` - 监控配置（单一真源）

### 路径规范
- 使用点号分隔的路径：`fusion_metrics.thresholds.fuse_buy`
- 避免简写键名：使用 `strategy_mode.triggers.market.min_trades_per_min` 而不是 `strategy.min_trades_per_min`

---

## 🔍 验证与检查

### 运行时验证脚本检查项

`tools/runtime_validation.py` 中的 `test_monitoring_binding()` 方法会：
1. ✅ 从统一真源读取阈值配置
2. ✅ 检查配置键冲突（多真源警告）
3. ✅ 验证阈值类型正确性

### 配置验证脚本检查项

建议在 `tools/validate_config.py` 中添加：
- 检查同名关键字段是否在多棵树出现
- 输出警告或错误，提示收敛到单一真源

---

## 🚀 迁移计划（建议）

### Phase 1: 标记废弃（本次）
- [x] 更新运行时验证脚本，使用统一真源路径
- [x] 添加配置键冲突检查
- [ ] 在 `components.fusion.thresholds` 和 `components.strategy.*` 添加废弃注释

### Phase 2: 代码迁移（下轮）
- [ ] 将所有读取 `components.fusion.thresholds.*` 的代码改为 `fusion_metrics.thresholds.*`
- [ ] 将所有读取 `components.strategy.*` 的代码改为 `strategy_mode.*`

### Phase 3: 清理配置（下轮）
- [ ] 移除 `defaults.yaml` 中的 `components.fusion.thresholds`
- [ ] 移除 `defaults.yaml` 中的 `components.strategy.*` 相关配置

---

## 📝 参考

- **GCC 验收清单：** `reports/MERGE_GATE_CHECKLIST.md`
- **运行时验证结果：** `reports/runtime_validation_results.json`
- **有效配置导出：** `reports/effective-config.json`

---

**最后更新：** 2025-10-30  
**状态：** ⚠️ 待迁移（当前验证已对齐到统一真源）

