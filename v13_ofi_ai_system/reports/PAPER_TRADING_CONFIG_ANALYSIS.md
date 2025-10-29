# 纸上交易模拟器配置加载问题分析

## 📋 配置加载问题概览

**检查时间**: 2025-10-29  
**检查范围**: `paper_trading_simulator.py` 组件配置加载机制  
**检查目标**: 识别配置加载问题和缺失的配置文件

## 🔍 发现的问题

### ⚠️ 关键问题1: 场景参数配置文件缺失

**问题描述**:
```python
# 第43行代码
self.config_path = config_path or str(PROJECT_ROOT / "reports/scenario_opt/strategy_params_fusion_clean.yaml")
```

**问题分析**:
1. **路径不存在**: `reports/scenario_opt/strategy_params_fusion_clean.yaml` 文件不存在
   - ✅ 检查结果: `reports/scenario_opt/` 目录不存在
   - ✅ 检查结果: 整个 `v13_ofi_ai_system` 目录下没有任何 `strategy_params` 相关的YAML文件

2. **运行影响**: 
   - 初始化时会抛出 `Exception("场景参数加载失败")` 
   - 模拟器无法加载2x2场景的参数（止盈止损、Z阈值等）
   - 导致策略执行失败

### ⚠️ 关键问题2: 配置加载器路径不一致

**代码位置**: 第117行
```python
cfg = UnifiedConfigLoader(base_dir=os.environ.get("CONFIG_DIR", "config"))
```

**问题分析**:
1. **默认目录**: `UnifiedConfigLoader` 默认从 `config/` 目录加载
   - ✅ `config/` 目录存在
   - ✅ `config/defaults.yaml` 存在
   - ✅ `config/system.yaml` 存在

2. **加载优先级**:
   - `defaults.yaml` → `overrides.local.yaml` → 环境变量
   - 当前只有 `defaults.yaml` 和 `system.yaml`

### ⚠️ 关键问题3: 两个不同的defaults.yaml文件

**发现**:
1. `config/defaults.yaml` (90行) - Core Algorithm配置
2. `deploy/defaults.yaml` (48行) - 部署运行时配置

**问题分析**:
1. **内容不同**: 
   - `config/defaults.yaml`: 核心算法配置（融合、背离、策略、护栏、信号输出）
   - `deploy/defaults.yaml`: 部署配置（符号列表、轮转、超时、CVD参数）

2. **用途混淆**: 
   - 不清楚 `paper_trading_simulator.py` 应该加载哪个
   - `UnifiedConfigLoader` 默认在 `config/` 目录
   - 但是 `deploy/defaults.yaml` 可能更适合纸上交易

## 📁 加载的配置文件列表

### 当前实际加载的配置文件

#### 1. **UnifiedConfigLoader 加载**
优先级从高到低:
1. ✅ `config/defaults.yaml` (存在)
2. ⚠️ `config/overrides.local.yaml` (不存在)
3. ⚠️ 环境变量覆盖 `V13__*`

#### 2. **StrategyModeManager.load_scenario_params() 加载**
1. ❌ `reports/scenario_opt/strategy_params_fusion_clean.yaml` (不存在)
   - 这是关键问题！

## 🎯 配置加载流程

### 配置加载流程图

```
paper_trading_simulator.py 初始化
├── UnifiedConfigLoader(base_dir="config")
│   ├── 加载 config/defaults.yaml
│   ├── 加载 config/overrides.local.yaml (可选)
│   └── 加载环境变量 V13__* (可选)
│   └── 创建统一配置对象 cfg
│
├── CoreAlgorithm(symbol, signal_config, config_loader=cfg)
│   └── 使用 cfg 加载核心算法配置
│
├── StrategyModeManager(config_loader=cfg)
│   └── 使用 cfg 加载策略模式配置
│
└── manager.load_scenario_params(self.config_path)
    └── ❌ 加载 reports/scenario_opt/strategy_params_fusion_clean.yaml
        └── 失败！文件不存在
```

## 📝 配置内容分析

### config/defaults.yaml 配置内容

```yaml
# 融合信号配置
fusion:
  thresholds:
    fuse_buy: 1.0
    fuse_sell: -1.0
    fuse_strong_buy: 2.3
    fuse_strong_sell: -2.3
  consistency:
    min_consistency: 0.20
    strong_min_consistency: 0.65
  smoothing:
    z_window: 60
    winsorize_percentile: 95
    mad_k: 2.0

# 背离检测配置
divergence:
  min_strength: 0.90
  min_separation_secs: 120
  count_conflict_only_when_fusion_ge: 1.0
  lookback_periods: 20

# 策略模式配置
strategy:
  mode: 'auto'
  hysteresis:
    window_secs: 60
    min_active_windows: 3
    min_quiet_windows: 6
  triggers:
    schedule: {...}
    market: {...}

# 护栏配置
guards:
  spread_bps_cap: 50
  max_missing_msgs_rate: 0.01
  max_event_lag_sec: 3.0
  exit_cooldown_sec: 30
  reverse_prevention_sec: 300
  warmup_period_sec: 300
```

### deploy/defaults.yaml 配置内容

```yaml
# 基础配置
SYMBOLS: "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT"
RUN_HOURS: 24
OUTPUT_DIR: ""

# 场景标签配置
WIN_SECS: 300
ACTIVE_TPS: 0.1
VOL_SPLIT: 0.5
FEE_TIER: "TM"

# CVD计算参数
CVD_SIGMA_FLOOR_K: 0.3
CVD_WINSOR: 2.5

# 融合计算参数
W_OFI: 0.6
W_CVD: 0.4
FUSION_CAL_K: 1.0
```

## 🚨 需要解决的问题

### 优先级1: 创建场景参数配置文件

需要创建文件: `reports/scenario_opt/strategy_params_fusion_clean.yaml`

**文件内容结构**:
```yaml
signal_kind: "oficvd_fusion"
horizon_s: 60
cost_bps: 3.0

scenarios:
  Q_H:
    long:
      Z_HI: 2.2
      Z_HI_LONG: 2.2
      TP_BPS: 15
      SL_BPS: 10
    short:
      Z_HI: 2.2
      Z_HI_SHORT: 2.2
      TP_BPS: 15
      SL_BPS: 10
  A_H:
    long:
      Z_HI: 1.8
      Z_HI_LONG: 1.8
      TP_BPS: 12
      SL_BPS: 9
    short:
      Z_HI: 1.8
      Z_HI_SHORT: 1.8
      TP_BPS: 12
      SL_BPS: 9
  A_L:
    long:
      Z_HI: 1.5
      Z_HI_LONG: 1.5
      TP_BPS: 10
      SL_BPS: 8
    short:
      Z_HI: 1.5
      Z_HI_SHORT: 1.5
      TP_BPS: 10
      SL_BPS: 8
  Q_L:
    long:
      Z_HI: 1.2
      Z_HI_LONG: 1.2
      TP_BPS: 8
      SL_BPS: 6
    short:
      Z_HI: 1.2
      Z_HI_SHORT: 1.2
      TP_BPS: 8
      SL_BPS: 6
```

### 优先级2: 统一配置系统问题

**问题**: `paper_trading_simulator.py` 同时使用了两套配置系统
1. `UnifiedConfigLoader` - 从 `config/defaults.yaml` 加载
2. `load_scenario_params()` - 从自定义路径加载场景参数

**建议**:
1. 将所有场景参数迁移到 `config/system.yaml`
2. 使用统一的配置加载机制
3. 或者创建符号链接/配置文件映射

### 优先级3: defaults.yaml 分离问题

**建议**:
1. 将 `config/defaults.yaml` 重命名为 `config/core_algorithm.yaml`
2. 保留 `deploy/defaults.yaml` 作为部署配置
3. 在 `paper_trading_simulator.py` 中明确指定配置目录

## 📋 配置加载问题清单

| 问题 | 优先级 | 状态 | 影响 |
|------|--------|------|------|
| 场景参数文件缺失 | P0 | ❌ 未解决 | 无法加载2x2场景参数，初始化失败 |
| config/defaults.yaml 冲突 | P1 | ⚠️ 待解决 | 配置来源不明确 |
| overrides.local.yaml 缺失 | P2 | ✅ 可接受 | 可选文件，不影响运行 |
| 环境变量覆盖未配置 | P2 | ✅ 可接受 | 可选配置 |
| 配置日志不完整 | P3 | ⚠️ 待改进 | 难以追踪配置来源 |

## 🎯 解决方案建议

### 方案1: 创建缺失的场景参数文件（推荐）

**步骤**:
1. 创建目录: `mkdir -p reports/scenario_opt`
2. 创建文件: `reports/scenario_opt/strategy_params_fusion_clean.yaml`
3. 填充上述内容结构
4. 测试加载

### 方案2: 修改配置路径

**方案A**: 使用相对路径
```python
self.config_path = config_path or "reports/scenario_opt/strategy_params_fusion_clean.yaml"
```

**方案B**: 从config目录加载
```python
self.config_path = config_path or str(PROJECT_ROOT / "config/scenario_params.yaml")
```

### 方案3: 统一配置系统（长期）

将所有配置统一到 `config/system.yaml`，使用配置分层机制。

## ✅ 验证清单

- [ ] 创建场景参数配置文件
- [ ] 测试配置加载成功
- [ ] 验证场景参数正确应用
- [ ] 检查日志输出
- [ ] 验证纸上交易模拟器可正常运行

## 📊 配置系统架构建议

```
config/
├── system.yaml              # 系统主配置
├── core_algorithm.yaml      # 核心算法配置（重命名 defaults.yaml）
├── scenario_params.yaml     # 场景参数（新文件）
└── environments/
    ├── development.yaml
    ├── testing.yaml
    └── production.yaml

reports/
└── scenario_opt/
    └── strategy_params_fusion_clean.yaml  # 实际场景参数（由优化脚本生成）
```

## 🔗 相关文件

- `config/defaults.yaml` - Core Algorithm配置
- `deploy/defaults.yaml` - 部署运行时配置
- `config/unified_config_loader.py` - 统一配置加载器
- `src/utils/strategy_mode_manager.py` - 场景参数加载逻辑

## 🎯 结论

**核心问题**: 缺少场景参数配置文件 `reports/scenario_opt/strategy_params_fusion_clean.yaml`

**下一步**: 创建该文件并填充场景参数内容，确保纸上交易模拟器可以正常加载配置。

