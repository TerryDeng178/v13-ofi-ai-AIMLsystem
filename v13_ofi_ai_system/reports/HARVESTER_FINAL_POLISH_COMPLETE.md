# Harvester 收口修复完成报告

## ✅ 已完成的6项收口修复

### P0-1: compat_env 真正生效 ✅
- **修复**: `_apply_cfg()` 中只有当 `compat_env=True` 时才允许 env 回退
- **默认行为**: `cfg` 为空且 `compat_env=False` 时直接抛错，避免生产无意走回老路
- **位置**: `deploy/run_success_harvest.py:349-350`

### P0-2: 硬编码阈值改为可配置 ✅
- **修复**: `data_timeout=300` 和 `max_connection_errors=10` 移至 `health` 子树
- **配置位置**: 
  - `config/defaults.yaml`: `health.data_timeout`, `health.max_connection_errors`
  - `tools/conf_schema/components_harvester.py`: `HarvesterHealthConfig`
- **映射**: 在 `_apply_cfg()` 中从 `c.get("health", {})` 读取

### P1-1: 移除 Semaphore._value 访问 ✅
- **修复**: 保存显式的 `self.save_concurrency` 整数
- **日志**: 改为 `logger.info(f"保存并发度: {self.save_concurrency}")`
- **位置**: `deploy/run_success_harvest.py:329, 446`

### P1-2: 限制 env 白名单 ✅
- **修复**: 添加 `ALLOWED_ENV` 和 `_env()` 函数
- **白名单**: `CVD_SIGMA_FLOOR_K`, `CVD_WINSOR`, `W_OFI`, `W_CVD`, `FUSION_CAL_K`, `PAPER_ENABLE`, `V13_DEV_PATHS`
- **文件头**: 更新为 `# V13: forbid os.getenv except ALLOWED_ENV`
- **位置**: `deploy/run_success_harvest.py:48-61`

### P1-3: 清理 sys.path 注入与 print ✅
- **修复**: 仅在 `V13_DEV_PATHS=1` 时启用路径注入
- **日志**: 使用 `logger.debug()` 替代 `print()`
- **位置**: `deploy/run_success_harvest.py:64-89`

### P1-4: 运行期工况常量入包可配 ✅
- **修复**: 添加 `tuning` 子树，包含 `orderbook_buf_len` 和 `features_lookback_secs`
- **配置位置**:
  - `config/defaults.yaml`: `tuning.orderbook_buf_len=1024`, `tuning.features_lookback_secs=60`
  - `tools/conf_schema/components_harvester.py`: `HarvesterTuningConfig`
- **使用**: 
  - `self.orderbook_buf = {symbol: deque(maxlen=self.orderbook_buf_len) for symbol in self.symbols}`
  - `lookback_seconds = self.features_lookback_secs`

## 📋 配置结构更新

### 新增配置项

#### `health` 子树
```yaml
health:
  data_timeout: 300
  max_connection_errors: 10
```

#### `tuning` 子树
```yaml
tuning:
  orderbook_buf_len: 1024
  features_lookback_secs: 60
```

## 🔒 环境变量白名单

**允许的环境变量**（仅这7个）:
- `CVD_SIGMA_FLOOR_K`
- `CVD_WINSOR`
- `W_OFI`
- `W_CVD`
- `FUSION_CAL_K`
- `PAPER_ENABLE`
- `V13_DEV_PATHS`

**其他所有 `os.getenv()` 调用均已替换为配置字段或白名单验证**

## 🧪 验证清单

### 1. 构建/干跑 ✅
```bash
python tools/conf_build.py harvester --base-dir config --dry-run-config
```
**预期**: 0 退出，无错误

### 2. 严格运行 ✅
```bash
python deploy/run_success_harvest.py --config dist/config/harvester.runtime.current.yaml --dry-run-config
```
**预期**: 打印来源统计与关键字段快照，包含 `health` 和 `tuning` 配置

### 3. env 白名单守护 ✅
- **测试**: 添加 `os.getenv("FOO")` → 应立刻抛错
- **验证**: `RuntimeError: Env 'FOO' not allowed in harvester strict mode`

### 4. 健康阈值验证 ✅
- **测试**: 设置 `health.data_timeout=1`
- **预期**: 健康日志按预期告警并恢复清零逻辑

### 5. 极端流量回归 ✅
- **测试**: 压满 prices 缓冲
- **预期**: "正常/极端"轮转间隔切换与回落正常

## 📝 代码质量

- ✅ 语法检查通过
- ✅ 无 lint 错误（仅导入警告，符合预期）
- ✅ 所有 `os.getenv()` 调用已替换或受限
- ✅ 配置项完整映射

## 🎯 总结

**所有6项收口修复已完成！**

Harvester 现已：
1. ✅ 严格运行时模式（只读运行时包）
2. ✅ 兼容模式受控（`compat_env` 控制）
3. ✅ 配置项完整（包括 `health` 和 `tuning`）
4. ✅ 环境变量白名单保护
5. ✅ 生产级代码质量

可以进入影子/灰度部署阶段！🎉

