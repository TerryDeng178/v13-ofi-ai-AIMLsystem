# 严格运行模式迁移完成报告

**完成时间**: 2025-01-XX  
**状态**: ✅ 已完成

## 迁移概览

所有关键组件入口已迁移到严格运行模式（Strict Runtime Mode），默认从运行时包加载配置，拒绝旁路读取全局配置。

---

## 已迁移组件

### 1. ✅ CoreAlgorithm (`core/core_algo.py`)

**修改内容**:
- 默认使用严格运行时模式（环境变量 `V13_STRICT_RUNTIME=true`）
- 运行时包路径: `dist/config/core_algo.runtime.current.yaml`
- 环境变量覆盖支持: `V13_CORE_ALGO_RUNTIME_PACK`
- 兼容模式开关: `V13_COMPAT_GLOBAL_CONFIG=true` 或命令行 `--compat-global-config`
- 场景快照指纹验证: 自动验证 `scenarios_snapshot_sha256`

**日志输出**:
```
[严格模式] 加载运行时配置包: dist/config/core_algo.runtime.current.yaml
  版本: 1.0.0
  Git SHA: dee5fb37e
  组件: core_algo
  来源统计: {'defaults': 10, 'system': 4, ...}
  场景快照指纹: 623caa34... (如果适用)
```

---

### 2. ✅ Paper Trading Simulator (`paper_trading_simulator.py`)

**修改内容**:
- 初始化方法添加 `compat_global_config` 参数
- 命令行参数: `--compat-global-config`（临时过渡选项）
- 运行时包路径: `dist/config/strategy.runtime.current.yaml`
- 环境变量覆盖支持: `V13_STRATEGY_RUNTIME_PACK`
- 配置包装器: `RuntimeConfigWrapper` 保持与 `UnifiedConfigLoader` 的接口兼容

**使用示例**:
```bash
# 严格模式（默认）
python paper_trading_simulator.py

# 兼容模式（临时过渡）
python paper_trading_simulator.py --compat-global-config
```

---

### 3. ⚠️ Run Success Harvest (`deploy/run_success_harvest.py`)

**状态**: 待迁移

**说明**: 该组件目前直接使用硬编码配置参数，暂未接入统一配置系统。如需迁移，需要：
1. 识别所有硬编码参数
2. 映射到统一配置系统
3. 选择合适的运行时包（可能需要新建 `harvester` 组件）

---

## 严格模式特性

### 启动日志

严格模式会在启动时打印以下信息：

```
[严格模式] 加载运行时配置包: dist/config/{component}.runtime.current.yaml
  版本: 1.0.0
  Git SHA: dee5fb37e
  组件: {component}
  来源统计: {'defaults': X, 'system': Y, 'overrides': Z, 'env': W, 'locked': V}
  场景快照指纹: abc12345... (如果适用)
```

### 场景快照指纹验证

对于 `strategy` 组件，自动验证场景文件的 SHA256 哈希：
- 运行时包中嵌入: `scenarios_snapshot_sha256`
- 启动时读取实际场景文件并计算哈希
- 不一致时抛出 `StrictRuntimeConfigError` 并拒绝启动

### 降级机制

如果严格模式加载失败（例如运行时包不存在），自动降级到兼容模式：
```
[WARNING] 严格模式加载失败，降级到兼容模式: {error}
```

**商业策略**: 在生产环境，应确保运行时包存在，避免意外降级。

---

## 环境变量配置

### 控制开关

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `V13_STRICT_RUNTIME` | `true` | 是否启用严格运行时模式 |
| `V13_COMPAT_GLOBAL_CONFIG` | `false` | 是否启用兼容模式（临时过渡） |

### 运行时包路径覆盖

| 环境变量 | 默认路径 | 说明 |
|---------|---------|------|
| `V13_CORE_ALGO_RUNTIME_PACK` | `dist/config/core_algo.runtime.current.yaml` | CoreAlgorithm 运行时包路径 |
| `V13_STRATEGY_RUNTIME_PACK` | `dist/config/strategy.runtime.current.yaml` | Strategy/Paper Trading 运行时包路径 |

---

## CI/CD 集成

### 构建验证流程

写完配置文件后，CI自动执行6个阶段的验证：

1. **Stage 1 - Dry run config validation**
   ```bash
   python tools/conf_build.py all --base-dir config --dry-run-config
   ```

2. **Stage 2 - Acceptance tests**
   ```bash
   pytest tests/test_config_system*.py -v
   ```

3. **Stage 3 - Build runtime packs**
   ```bash
   python tools/conf_build.py all --base-dir config
   ```

4. **Stage 4 - Verify runtime packs**
   - 检查所有 `.runtime.current.yaml` 文件存在
   - 验证运行时包结构（`__meta__`, `__invariants__`）

5. **Stage 5 - Dry-run verify all components**
   ```bash
   python tools/conf_build.py all --base-dir config --dry-run-config
   ```

6. **Stage 6 - Upload artifacts**
   - 上传所有运行时包到 CI artifacts

### 主分支强制规则

在主分支（`main`/`master`）构建时：
- ✅ `fail_on_unconsumed=True` - 未消费键会阻断发布
- ✅ `allow_env_override_locked=False` - 禁止环境变量覆盖锁定参数

---

## Windows 兼容性

### Current 软链方案

在 Windows 上，`current` 链接使用文件复制而非符号链接（`shutil.copy2`），确保：
1. 所有Windows版本兼容
2. 文件内容一致性（复制包括元数据）
3. 自动覆盖旧版本

**实现位置**: `v13conf/packager.py::save_runtime_pack()`

```python
if platform.system() == 'Windows':
    import shutil
    if current_path.exists():
        current_path.unlink()
    shutil.copy2(output_path, current_path)
```

---

## 发布说明要点

### 未消费键 = 阻断发布（主分支）

在主分支构建时，如果发现未消费的配置键，构建会失败。

**常见拼写陷阱示例**:
- ❌ `fuseStrongBuy` → ✅ `fuse_strong_buy`
- ❌ `minConsistency` → ✅ `min_consistency`
- ❌ `wOfi` → ✅ `w_ofi`
- ❌ `Z_HI` → ✅ `z_hi` (在divergence组件中)

**解决方法**:
1. 检查拼写是否正确
2. 确认键是否在对应的 Pydantic schema 中定义
3. 如果确实不需要，从配置文件中移除

### 环境变量覆盖锁定参数

**默认行为**: 禁止环境变量覆盖 OFI 锁定参数（`allow_env_override_locked=False`）

**紧急场景**: 如需临时覆盖，手动设置环境变量：
```bash
export V13_COMPAT_GLOBAL_CONFIG=true
export V13_STRICT_RUNTIME=false
```

**注意**: 此选项为临时过渡，未来版本将移除。

---

## 测试验证

### 快速验收清单

1. ✅ **干运行验证**
   ```bash
   python tools/conf_build.py all --base-dir config --dry-run-config
   ```

2. ✅ **验收测试**
   ```bash
   pytest tests/test_config_system*.py -v
   ```

3. ✅ **构建验证**
   ```bash
   python tools/conf_build.py all --base-dir config
   ```

4. ✅ **运行时包验证**
   ```bash
   ls -la dist/config/*.runtime.current.yaml
   ```

5. ✅ **严格模式测试**
   ```bash
   python core/core_algo.py  # 默认严格模式
   python paper_trading_simulator.py  # 默认严格模式
   ```

---

## 已知限制

1. **run_success_harvest.py**: 暂未迁移，仍使用硬编码参数
2. **其他脚本入口**: 需要逐个识别并迁移
3. **兼容模式**: 将在未来版本中移除 `--compat-global-config` 选项

---

## 下一步行动

1. ✅ 完成核心组件迁移
2. ⚠️ 迁移剩余组件（run_success_harvest.py 等）
3. ⚠️ 生产环境部署验证
4. ⚠️ 监控和日志收集
5. ⚠️ 文档更新（用户手册）

---

## 总结

**迁移状态**: 🎉 **核心组件已迁移到严格运行模式**

**关键成果**:
- ✅ CoreAlgorithm 和 Paper Trading Simulator 已支持严格模式
- ✅ 场景快照指纹验证已启用
- ✅ CI/CD 构建验证流程已完善（6个阶段）
- ✅ Windows 兼容性已确认

**剩余工作**:
- ⚠️ 迁移 run_success_harvest.py 和其他脚本入口
- ⚠️ 生产环境灰度验证
- ⚠️ 移除兼容模式（未来版本）

