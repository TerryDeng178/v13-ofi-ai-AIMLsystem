# 旧键删除说明

## 变更摘要

根据配置系统统一化要求，以下旧键已从配置文件中移除：

### 已删除的配置路径

1. **`components.fusion.thresholds.*`** → 已迁移到 `fusion_metrics.thresholds.*`
   - `components.fusion.thresholds.fuse_buy` → `fusion_metrics.thresholds.fuse_buy`
   - `components.fusion.thresholds.fuse_sell` → `fusion_metrics.thresholds.fuse_sell`
   - `components.fusion.thresholds.fuse_strong_buy` → `fusion_metrics.thresholds.fuse_strong_buy`
   - `components.fusion.thresholds.fuse_strong_sell` → `fusion_metrics.thresholds.fuse_strong_sell`

2. **`components.strategy.triggers.market.*`** → 已迁移到 `strategy_mode.triggers.market.*`
   - `components.strategy.triggers.market.min_trades_per_min` → `strategy_mode.triggers.market.min_trades_per_min`
   - `components.strategy.triggers.market.min_quote_updates_per_sec` → `strategy_mode.triggers.market.min_quote_updates_per_sec`
   - 其他 `components.strategy.*` 子键

### 迁移时间表

- **Shim 映射保留期**：当前版本继续支持 Shim 自动映射，但会显示废弃警告
- **完全移除日期**：下一版本（v1.3）将彻底移除 Shim 映射

### 如何迁移

1. **更新配置文件**：
   ```yaml
   # 旧配置（已废弃）
   components:
     fusion:
       thresholds:
         fuse_buy: 0.95
   
   # 新配置（单一真源）
   fusion_metrics:
     thresholds:
       fuse_buy: 0.95
   ```

2. **验证迁移**：
   ```bash
   python tools/validate_config.py --strict
   ```

3. **如遇冲突**：
   - 临时放行：`ALLOW_LEGACY_KEYS=1 python tools/validate_config.py --strict`
   - 生产环境：不允许设置 `ALLOW_LEGACY_KEYS=1`，必须完全迁移

### 影响范围

- **配置文件**：`config/defaults.yaml`, `config/system.yaml`
- **代码引用**：所有使用 `components.fusion.*` 或 `components.strategy.*` 的地方
- **验证工具**：`tools/validate_config.py` 会检测冲突

### 回滚方案

如需回滚到旧配置：

1. 恢复包含旧键的配置文件版本
2. 确保 `ALLOW_LEGACY_KEYS=1` 仅在非生产环境使用
3. 参考 Git 历史记录恢复相关配置段

---

**更新日期**：2025-10-30  
**负责人**：系统配置团队

