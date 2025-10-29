# V13 统一配置管理系统

## 概述

本系统实现了"统一配置单一事实来源 → 按组件产出交付包（runtime yaml）"的配置管理架构。

### 核心特性

1. **四层合并机制**：defaults.yaml < system.yaml < overrides.local.yaml < 环境变量
2. **OFI参数锁定**：`locked_ofi_params.yaml` 中的参数具有最高优先级（可被环境变量突破）
3. **来源追踪**：每个配置键都记录来源层，便于调试和审计
4. **不变量校验**：自动检查权重和、阈值关系、范围等约束
5. **组件交付包**：每个组件生成独立的运行时配置包，包含元信息和校验摘要

## 目录结构

```
v13_ofi_ai_system/
├── config/                          # 配置源目录
│   ├── defaults.yaml                # 默认配置（生产就绪值）
│   ├── system.yaml                  # 系统级配置
│   ├── overrides.local.yaml         # 本地覆盖（.gitignore）
│   ├── locked_ofi_params.yaml       # OFI锁定参数
│   └── .gitignore                   # 忽略overrides.local.yaml
├── v13conf/                         # 配置管理库
│   ├── loader.py                    # 配置加载器
│   ├── normalizer.py                # 键名归一化
│   ├── invariants.py                # 不变量校验
│   └── packager.py                  # 交付包打包器
├── tools/
│   ├── conf_build.py                # CLI构建工具
│   └── conf_schema/                 # Pydantic Schema定义
├── dist/config/                     # 构建产物目录
│   ├── fusion.runtime.yaml
│   ├── ofi.runtime.yaml
│   └── ...
ών── tests/
    └── test_config_system.py        # 验收测试
```

## 快速开始

### 1. 构建所有组件运行时包

```bash
cd v13_ofi_ai_system
python tools/conf_build.py all --base-dir config
```

### 2. 仅构建fusion组件

```bash
python tools/conf_build.py fusion --base-dir config
```

### 3. 打印有效配置和来源

```bash
python tools/conf_build.py fusion --print-effective
```

### 4. 干运行（仅验证，不生成文件）

```bash
python tools/conf_build.py fusion --dry-run-config
```

### 5. 打印有效配置（脱敏、折叠）

```bash
# 默认模式（折叠大列表、脱敏敏感信息、摘要统计）
python tools/conf_build.py fusion --print-effective

# 详细模式（完整内容、逐键来源）
python tools/conf_build.py fusion --print-effective --verbose
```

### 6. 紧急场景：允许环境变量覆盖OFI锁定

```bash
python tools/conf_build.py fusion --allow-env-override-locked
```

## 配置优先级

**默认优先级**（推荐）：
1. **defaults.yaml** - 基础默认值（生产就绪参数）
2. **system.yaml** - 系统级配置覆盖
3. **overrides.local.yaml** - 本地开发覆盖（不提交Git）
4. **环境变量 V13__*** - 运行时动态覆盖
5. **locked_ofi_params.yaml** - OFI参数锁定（**最高优先级**，环境变量无法突破）

**紧急场景**：
使用 `--allow-env-override-locked` 标志允许环境变量突破OFI锁定参数（用于紧急排障）

```bash
python tools/conf_build.py fusion --allow-env-override-locked
```

**说明**: 
- 默认情况下，OFI锁定参数优先级最高，确保生产稳定性
- 仅在紧急排障场景使用环境变量突破功能
- 文档和实现已统一，避免认知冲突

## 环境变量格式

环境变量使用双下划线分隔路径：

```bash
# 设置Fusion强买入阈值
export V13__components__fusion__thresholds__fuse_strong_buy=2.5

# 设置权重（会自动解析为浮点数）
export V13__components__fusion__weights__w_ofi=0.65
export V13__components__fusion__weights__w_cvd=0.35

# 布尔值
export V13__logging__debug=true

# 列表（逗号分隔）
export V13__strategy__enabled_symbols=BTCUSDT,ETHUSDT
```

## 新功能特性

### 未消费键检测 ✅

自动检测配置中的拼写错误和悬空配置：

```bash
python tools/conf_build.py fusion
# 构建时会自动检测未消费键，并在 __invariants__ 中报告
```

### 打印优化（脱敏、折叠） ✅

- **脱敏**: 自动隐藏敏感信息（api_key, secret等）
- **折叠**: 大列表/字典自动折叠（>10个元素）
- **来源统计**: 默认只显示计数，`--verbose` 显示逐键来源

### 策略场景文件一致性 ✅

构建strategy组件时，自动将场景文件快照打包到运行时包，避免运行时路径漂移。

### 运行时严格模式 ✅

组件应使用 `load_strict_runtime_config()` 从运行时包加载配置，拒绝旁路覆盖：

```python
from v13conf.strict_mode import load_strict_runtime_config

config = load_strict_runtime_config("dist/config/fusion.runtime.yaml")
```

兼容模式（临时，排障用）：
```python
config = load_strict_runtime_config(
    "dist/config/fusion.runtime.yaml",
    compat_global_config=True  # 未来版本将删除
)
```

## 验收测试

运行验收测试验证配置系统：

```bash
cd v13_ofi_ai_system
pytest tests/test_config_system.py -v
```

### 测试覆盖

- ✅ 基础配置加载
- ✅ Fusion生产参数（±2.3 / 0.20 / 0.65）
- ✅ OFI锁定参数（z_window=80, ema_alpha=0.30, z_clip=3.0）
- ✅ 权重和为1.0约束
- ✅ 不变量校验
- ✅ 环境变量覆盖优先级
- ✅ 运行时包构建
- ✅ 阈值不变量
- ✅ 一致性不变量

## 生产参数基线

所有默认配置基于**最优配置报告**（`reports/🌸OPTIMAL_CONFIGURATION_REPORT.md`）中的生产就绪值：

### Fusion配置
- `fuse_strong_buy/sell`: ±2.3（控制强信号密度）
- `min_consistency`: 0.20（提高弱信号稳定性）
- `strong_min_consistency`: 0.65（提高强信号确认要求）

### OFI配置（锁定）
- `z_window`: 80
- `ema_alpha`: 0.30
- `z_clip`: 3.0

### CVD配置
- 支持分品种配置（BTCUSDT更严格，ETHUSDT稍宽松）

## 运行时包结构

生成的运行时包包含：

```yaml
__meta__:
  version: "1.0.0"
  git_sha: "abc1234"
  build_ts: "2025-10-29T12:00:00Z"
  component: "fusion"
  source_layers:
    defaults: 15
    system: 3
    overrides: 0
    env: 0
    locked: 5
  checksum: "a1b2c3d4..."

__invariants__:
  validation_passed: true
  errors: []
  checks:
    weights_sum_to_one:
      applicable: true
      w_ofi: 0.6
      w_cvd: 0.4
      sum: 1.0
      valid: true
    thresholds_valid:
      applicable: true
      buy_valid: true
      sell_valid: true
      all_valid: true

fusion:
  thresholds:
    fuse_buy: 1.0
    fuse_sell: -1.0
    fuse_strong_buy: 2.3
    fuse_strong_sell: -2.3
  # ...
```

## 集成到组件

### 旧方式（Deprecated）

```python
from unified_config_loader import UnifiedConfigLoader
cfg = UnifiedConfigLoader().load_all()
```

### 新方式（推荐）

```python
import yaml
from pathlib import Path

# 加载运行时包
runtime_pack_path = Path("dist/config/fusion.runtime.yaml")
with open(runtime_pack_path, 'r', encoding='utf-8') as f:
    pack = yaml.safe_load(f)

# 打印元信息
print(f"版本: {pack['__meta__']['version']}")
print(f"Git SHA: {pack['__meta__']['git_sha']}")
print(f"来源统计: {pack['__meta__']['source_layers']}")

# 使用组件配置
fusion_config = pack['fusion']
```

## 故障排查

### 问题：权重和不等于1.0

**错误信息**：
```
Fusion weights must sum to 1.0, got w_ofi=0.7, w_cvd=0.4, sum=1.1
建议: 请调整权重使 w_ofi + w_cvd = 1.0
```

**解决方案**：检查 `config/defaults.yaml` 或环境变量中的权重设置。

### 问题：OFI锁定参数被覆盖

**原因**：环境变量优先级高于锁定参数。

**解决方案**：
1. 检查是否有 `V13__components__ofi__*` 环境变量
2. 如需强制使用锁定参数，移除相关环境变量

### 问题：不变量校验失败

**查看详细信息**：
```bash
python tools/conf_build.py fusion --print-effective
 fructools/conf_build.py all --dry-run-config
```

## 严格运行模式

### 组件入口迁移

所有关键组件已迁移到严格运行模式（Strict Runtime Mode），默认从运行时包加载配置。

**已迁移组件**:
- ✅ `core/core_algo.py` - CoreAlgorithm
- ✅ `paper_trading_simulator.py` - Paper Trading Simulator

**使用方法**:
```bash
# 严格模式（默认）
python core/core_algo.py
python paper_trading_simulator.py

# 兼容模式（临时过渡）
python core/core_algo.py --compat-global-config
python paper_trading_simulator.py --compat-global-config
```

**环境变量控制**:
- `V13_STRICT_RUNTIME=true` (默认) - 启用严格模式
- `V13_COMPAT_GLOBAL_CONFIG=false` (默认) - 禁用兼容模式
- `V13_CORE_ALGO_RUNTIME_PACK` - 覆盖CoreAlgorithm运行时包路径
- `V13_STRATEGY_RUNTIME_PACK` - 覆盖Strategy运行时包路径

**场景快照指纹验证**:
- 启动时自动验证场景文件的 SHA256 哈希
- 打印指纹前8位便于排查: `场景快照指纹: abc12345...`
- 不一致时拒绝启动并抛出 `StrictRuntimeConfigError`

### 发布说明

#### 未消费键 = 阻断发布（主分支）

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

#### 环境变量覆盖锁定参数

**默认行为**: 禁止环境变量覆盖 OFI 锁定参数（`allow_env_override_locked=False`）

**生产环境强制规则**: 在主分支构建时，`allow_env_override_locked` 强制为 `False`，确保锁定参数不会被意外覆盖。

**紧急场景**: 如需临时覆盖，手动设置环境变量：
```bash
export V13_COMPAT_GLOBAL_CONFIG=true
export V13_STRICT_RUNTIME=false
```

**注意**: `--compat-global-config` 选项为临时过渡，未来版本将移除。

### Windows 兼容性

在 Windows 上，`current` 链接使用文件复制而非符号链接（`shutil.copy2`），确保：
1. 所有Windows版本兼容
2. 文件内容一致性（复制包括元数据）
3. 自动覆盖旧版本

## 后续计划

- [x] CI集成（自动构建和验证）✅
- [x] 配置版本管理（semver + git_sha）✅
- [ ] 热重载支持
- [ ] 配置变更审计日志

