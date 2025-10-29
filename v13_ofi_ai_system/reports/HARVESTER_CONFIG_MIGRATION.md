# Harvester配置系统迁移完成报告

**完成时间**: 2025-01-XX  
**状态**: ✅ 主要迁移已完成，构造函数需要完善

## 完成的修改

### ✅ 1. 配置Schema定义

**位置**: `tools/conf_schema/components_harvester.py`

**内容**:
- 定义了完整的Harvester配置Schema（Pydantic模型）
- 包含所有配置项：paths, buffers, files, concurrency, timeouts, thresholds, dedup, scenario

### ✅ 2. 配置默认值

**位置**: `config/defaults.yaml`

**内容**:
- 添加了harvester组件的完整默认配置
- 包含所有必要的配置项及生产就绪的默认值

### ✅ 3. 构建系统集成

**修改位置**:
- `tools/conf_build.py`: 添加了`harvester`到COMPONENTS列表
- `v13conf/packager.py`: 添加了harvester到component_map
- `v13conf/unconsumed_keys.py`: 添加了harvester的所有配置键到SCHEMA_CONSUMED_KEYS
- `tools/conf_schema/__init__.py`: 导出了HarvesterConfig

### ✅ 4. 入口函数修改

**位置**: `deploy/run_success_harvest.py::main()`

**修改内容**:
- 添加了`--config`, `--dry-run-config`, `--compat-global-config`命令行参数
- 使用`load_component_runtime_config`加载配置
- 支持dry-run验证
- 打印有效配置

 improve

### ✅ 5. OFI对齐阈值修复

**位置**: `deploy/run_success_harvest.py::_process_trade_data()`

**修改内容**:
- 将`os.getenv('OFI_MAX_LAG_MS', '800')`改为`getattr(self, "ofi_max_lag_ms", 800)`
- 使用实例字段而非环境变量

## 待完善的修改

### ⚠️ 构造函数完整支持

**位置**: `deploy/run_success_harvest.py::__init__()`

**当前状态**: 
- 已添加`cfg`参数，但默认仍使用旧的环境变量读取逻辑
- 需要完成完整重写，将cfg为None时的逻辑也改为使用配置字段

**建议**: 
- 当cfg为None时，使用默认配置值（从schema获取）
- 移除所有`os.getenv()`读取，统一使用配置字段

## 验证步骤

### 1. 构建运行时包

```bash
cd v13_ofi_ai_system
python tools/conf_build.py harvester --base-dir config --dry-run-config
python tools/conf_build.py harvester --base-dir config
```

**预期**: 
- 构建成功
- 产出`dist/config/harvester.runtime.{semver}.{gitsha8}.yaml`
- 包含所有harvester配置项

### 2. Dry-run验证

```bash
python deploy/run_success_harvest.py --dry-run-config
```

**预期**: 
- 加载运行时包成功
- 打印有效配置和来源统计
- 退出（不运行采集器）

### 3. 启动采集器

```bash
python deploy/run_success_harvest.py --config dist/config/harvester.runtime.current.yaml
```

**预期**: 
- 使用运行时包配置启动
- 不再依赖环境变量（harvester相关）
- 配置值来自运行时包

## 配置项映射

| 原环境变量 | 配置路径 | 默认值 |
|-----------|---------|--------|
| `EXTREME_TRAFFIC_THRESHOLD` | `components.harvester.thresholds.extreme_traffic_threshold` | 30000 |
| `EXTREME_ROTATE_SEC` | `components.harvester.thresholds.extreme_rotate_sec` | 30 |
| `MAX_ROWS_PER_FILE` | `components.harvester.files.max_rows_per_file` | 50000 |
| `SAVE_CONCURRENCY` | `components.harvester.concurrency.save_concurrency` | 2 |
| `HEALTH_CHECK_INTERVAL` | `components.harvester.timeouts.health_check_interval` | 25 |
| `STREAM_IDLE_SEC` | `components.harvester.timeouts.stream_idle_sec` | 120 |
| `TRADE_TIMEOUT` | `components.harvester.timeouts.trade_timeout` | 150 |
| `ORDERBOOK_TIMEOUT` | `components.harvester.timeouts.orderbook_timeout` | 180 |
| `BACKOFF_RESET_SECS` | `components.harvester.timeouts.backoff_reset_secs` | 300 |
| `DEDUP_LRU` | `components.harvester.dedup.lru_size` | 32768 |
| `QUEUE_DROP_THRESHOLD` | `components.harvester.dedup.queue_drop_threshold` | 1000 |
| `OFI_MAX_LAG_MS` | `components.harvester.thresholds.ofi_max_lag_ms` | 800 |
| `PARQUET_ROTATE_SEC` | `components.harvester.files.parquet_rotate_sec` | 60 |
| `WIN_SECS` | `components.harvester.scenario.win_secs` | 300 |
| `ACTIVE_TPS` | `components.harvester.scenario.active_tps` | 0.1 |
| `VOL_SPLIT` | `components.harvester.scenario.vol_split` | 0.5 |
| `FEE_TIER` | `components.harvester.scenario.fee_tier` | TM |

## 总结

✅ **已完成**: Schema定义、默认配置、构建集成、入口修改、部分字段迁移  
⚠️ **待完善**: 构造函数完整重写（移除所有os.getenv读取）

**下一步**: 完成构造函数的完整重写，确保所有配置项都从cfg读取，完全脱离环境变量依赖。

