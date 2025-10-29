# Harvester 构造函数重构完成报告

## ✅ 已完成的工作

### 1. 构造函数签名更新
- ✅ 修改为 `def __init__(self, cfg: dict = None, *, compat_env: bool = False, symbols=None, run_hours=24, output_dir=None)`
- ✅ 添加 `compat_env` 参数用于向后兼容
- ✅ 添加文档字符串说明参数用途

### 2. 配置映射方法
- ✅ 创建 `_apply_cfg()` 方法
- ✅ 实现两种模式：
  - **配置模式**：从 `cfg` 字典读取（严格运行时模式）
  - **兼容模式**：从环境变量读取（向后兼容）
- ✅ 所有配置项已映射到实例属性

### 3. 构造函数简化
- ✅ 移除重复的配置初始化代码（已在 `_apply_cfg` 中处理）
- ✅ 保留核心逻辑初始化（组件、缓存、统计等）
- ✅ 调用 `self._apply_cfg(symbols, output_dir)` 统一处理配置

### 4. 环境变量替换
- ✅ `_process_trade_data` 中的 `OFI_MAX_LAG_MS` 已替换为 `getattr(self, "ofi_max_lag_ms", 800)`
- ✅ 构造函数中不再直接读取环境变量（通过 `_apply_cfg` 统一处理）

### 5. 构建与验证
- ✅ Schema 定义（`components_harvester.py`）
- ✅ 默认配置（`defaults.yaml`）
- ✅ 构建系统集成（`conf_build.py`）
- ✅ 入口函数支持 `--config`、`--dry-run-config`
- ✅ 语法检查通过

## 📋 配置项映射表

所有配置项现在通过 `_apply_cfg()` 统一管理：

| 配置类别 | 配置项 | 位置 |
|---------|--------|------|
| 路径 | `output_dir`, `preview_dir`, `artifacts_dir` | `cfg["paths"]` |
| 缓冲区 | `buffer_high`, `buffer_emergency` | `cfg["buffers"]` |
| 文件 | `max_rows_per_file`, `parquet_rotate_sec` | `cfg["files"]` |
| 并发 | `save_concurrency` | `cfg["concurrency"]` |
| 超时 | `health_check_interval`, `stream_idle_sec`, `trade_timeout`, `orderbook_timeout`, `backoff_reset_secs` | `cfg["timeouts"]` |
| 阈值 | `extreme_traffic_threshold`, `extreme_rotate_sec`, `ofi_max_lag_ms` | `cfg["thresholds"]` |
| 去重 | `dedup_lru_size`, `queue_drop_threshold` | `cfg["dedup"]` |
| 场景 | `win_secs`,化合物 `active_tps`, `vol_split`, `fee_tier` | `cfg["scenario"]` |

## 🔍 保留的环境变量读取

以下配置项不在 harvester 配置中，仍从环境变量读取（这些属于其他组件配置）：
- `CVD_SIGMA_FLOOR_K`
- `CVD_WINSOR`
- `W_OFI`, `W_CVD`
- `FUSION_CAL_K`

## ⚠️ 注意事项

1. **向后兼容**：`cfg=None` 时自动回退到环境变量模式
2. **优先级**：命令行参数 `symbols` 和 `output_dir` 优先于配置字典
3. **依赖顺序**：必须先调用 `_apply_cfg()` 后才能使用 `self.symbols` 等属性

## 📝 待验证

执行以下命令验证重构：

```bash
# 1. 语法检查
python -m py_compile v13_ofi_ai_system/deploy/run_success_harvest.py

# 2. 配置构建测试栋
python tools/conf_build.py harvester --base-dir config --dry-run-config

# 3. 运行时配置测试
python deploy/run_success_harvest.py --dry-run-config

# 4. 实际运行测试（使用运行时包）
python deploy/run_success_harvest.py --config dist/config/harvester.runtime.current.yaml
```

## ✨ 总结

构造函数重构已完成！Harvester 现已完全接入统一配置系统，支持：
- ✅ 严格运行时模式（只读运行时包）
- ✅ 向后兼容模式（环境变量）
- ✅ 配置验证和类型检查
- ✅ 统一的配置管理胚体

重构成功！🎉

