# Harvester 构造函数重构分析

## 问题诊断：为什么重构一直失败？

### 主要原因

1. **文件过大**：`run_success_harvest.py` 有 **2147行**，`__init__` 函数超过 **240行**
2. **依赖关系复杂**：配置项之间有依赖（如 `self.symbols` 被多处使用）
3. **替换字符串不精确**：大段代码替换时，空格、注释等细微差异导致匹配失败
4. **文件状态变化**：多次编辑后，文件内容与预期不一致

### 当前状态

✅ **已完成**：
- `_apply_cfg()` 方法已创建并实现
- 语法错误已修复（`Kit`、`centre`、` priorities`）
- 配置映射逻辑完整（支持 cfg 和环境变量两种模式）

⚠️ **待完成**：
- `__init__` 函数签名还未更新（仍为旧签名）
- `__init__` 中仍有硬编码配置和环境变量读取
- 需要调用 `_apply_cfg()` 并移除重复代码

## 解决方案

### 策略：分步骤、小范围修改

#### 步骤1：修改函数签名和开头（最小改动）
```python
def __init__(self, cfg: dict = None, *, compat_env: bool = False, symbols=None, run_hours=24, output_dir=None):
    self.cfg = cfg or {}
    self._compat_env = compat_env
    self.base_dir = Path(__file__).parent.absolute()
    self.run_hours = run_hours
    self.start_time = datetime.now().timestamp()
    self.end_time = self.start_time + (run_hours * 3600)
    
    # 调用配置应用方法
    self._apply_cfg(symbols, output_dir)
    
    # 继续后续初始化...
```

#### 步骤2：移除重复的配置初始化
删除以下重复代码（已在 `_apply_cfg` 中处理）：
- `self.symbols = symbols or [...]`
- `self.output_dir = ...`
- `self.preview_dir = ...`
- `self.artifacts_dir = ...`
- `self.buffer_high = {...}`
- `self.buffer_emergency = {...}`
- `self.extreme_traffic_threshold = int(os.getenv(...))`
- `self.extreme_rotate_sec = int(os.getenv(...))`
- `self.max_rows_per_file = int(os.getenv(...))`
- `self.save_semaphore = asyncio.Semaphore(...)`
- 以及其他所有从环境变量读取的配置

#### 步骤3：替换方法中的 `os.getenv` 调用
在 `_process_trade_data` 等方法中，将：
```python
max_lag_ms = int(os.getenv('OFI_MAX_LAG_MS', '800'))
```
改为：
```python
max_lag_ms = getattr(self, "ofi_max_lag_ms", 800)
```

## 推荐执行顺序

1. ✅ **已完成**：创建 `_apply_cfg` 方法
2. 🔄 **进行中**：修改 `__init__` 签名并调用 `_apply_cfg`
3. ⏳ **待执行**：删除 `__init__` 中的重复配置初始化
4. ⏳ **待执行**：替换方法中的 `os.getenv` 调用（第4步）

## 验证方法

重构完成后，运行以下命令验证：

```bash
# 1. 语法检查
python -m py_compile deploy/run_success_harvest.py

# 2. 配置构建测试
python tools/conf_build.py harvester --base-dir config --dry-run-config

# 3. 运行时配置测试
python deploy/run_success_harvest.py --dry-run-config
```

## 风险控制

- **向后兼容**：保留 `compat_env` 参数和旧参数（`symbols`, `run_hours`, `output_dir`）
- **渐进迁移**：`_apply_cfg` 支持 cfg 为空时回退到环境变量模式
- **测试覆盖**：确保两种模式都能正常工作

## 下一步行动

建议采用**分段替换**方式：
1. 每次只替换 10-20 行代码
2. 每次替换后进行语法检查
3. 逐步移除重复代码，保留核心逻辑

这样可以避免一次性大范围替换导致的失败。

