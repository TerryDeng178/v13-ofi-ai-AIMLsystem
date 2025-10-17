# Task 1.2.1 开发计划文档

## 📋 任务概述
**任务名称**: 创建OFI计算器基础类  
**任务编号**: Task_1.2.1  
**预计时间**: 30分钟  
**当前状态**: ⏳ 待确认后开始

---

## 🎯 任务目标分析

### 已提供的参考代码
用户提供了一个完整的参考实现包（`Task_1_2_1_OFI_Calculator_Pack/`），包含：
1. ✅ 完整的 `real_ofi_calculator.py` (134行)
2. ✅ 测试文件 `test_real_ofi_calculator.py` (30行)
3. ✅ README 和 CURSOR提示词

### 关键设计要点
根据CURSOR提示词，这个任务有以下严格要求：
1. ✅ **只做纯计算，不做任何 I/O**
2. ✅ **禁止修改** `binance_websocket_client.py` 和 `async_logging.py`
3. ✅ **仅实现 `update_with_snapshot`**，`update_with_l2_delta` 保持 `NotImplementedError`
4. ✅ **不引入第三方库**（numpy/pandas等），只用标准库
5. ✅ **提供 docstring、类型注解**
6. ✅ **必须通过 `py_compile` 和 `pytest`**

---

## 📐 核心算法设计

### 1. OFI计算公式
```
OFI = Σ w_k * (Δb_k - Δa_k)

其中:
- w_k: 第k档的权重（默认: [0.4, 0.25, 0.2, 0.1, 0.05]）
- Δb_k: 第k档买单量变化 = bids[k].qty - prev_bids[k].qty
- Δa_k: 第k档卖单量变化 = asks[k].qty - prev_asks[k].qty
```

### 2. z-OFI计算（标准化）
```
z-ofi = (ofi - mean) / std

其中:
- 使用滚动窗口（默认300）
- warmup期: max(5, z_window//5) 数据点
- warmup期间 z_ofi = None
```

### 3. EMA平滑
```
ema = α*ofi + (1-α)*ema

其中:
- α: ema_alpha（默认0.2）
- 首帧: ema = ofi
```

---

## 📂 文件结构

### 目标文件
```
v13_ofi_ai_system/src/real_ofi_calculator.py  (新建)
```

### 测试文件
```
v13_ofi_ai_system/tests/test_real_ofi_calculator.py  (新建)
```

---

## 🔧 实现方案

### 方案选择：直接复用参考代码
**理由**:
1. ✅ 参考代码完全符合任务要求
2. ✅ 代码质量高（类型注解、docstring、异常处理）
3. ✅ 已经过设计和验证
4. ✅ 使用标准库，无第三方依赖
5. ✅ 符合项目规则（最小补丁、接口不变量）

**执行步骤**:
1. 复制 `Task_1_2_1_OFI_Calculator_Pack/v13_ofi_ai_system/src/real_ofi_calculator.py`  
   到 `v13_ofi_ai_system/src/real_ofi_calculator.py`

2. 复制 `Task_1_2_1_OFI_Calculator_Pack/tests/test_real_ofi_calculator.py`  
   到 `v13_ofi_ai_system/tests/test_real_ofi_calculator.py`

3. 运行语法检查：
   ```bash
   python -m py_compile v13_ofi_ai_system/src/real_ofi_calculator.py
   ```

4. 运行单元测试：
   ```bash
   cd v13_ofi_ai_system
   pytest -q tests/test_real_ofi_calculator.py
   ```

---

## 📊 代码结构说明

### 核心类：RealOFICalculator

#### 配置类
```python
@dataclass
class OFIConfig:
    levels: int = 5           # 订单簿档位数
    weights: Optional[List[float]] = None  # 权重（默认 [0.4, 0.25, 0.2, 0.1, 0.05]）
    z_window: int = 300       # z-score滚动窗口
    ema_alpha: float = 0.2    # EMA平滑系数
```

#### 主要方法
1. **`__init__(symbol, cfg)`**
   - 初始化计算器
   - 设置权重、窗口大小、EMA系数
   - 初始化订单簿缓存

2. **`update_with_snapshot(bids, asks, event_time_ms)`**
   - 核心计算方法
   - 输入：当前订单簿快照
   - 输出：OFI值、z-OFI、EMA、各档位贡献

3. **`reset()`**
   - 重置所有状态

4. **`get_state()`**
   - 获取当前状态（用于调试）

#### 辅助方法
- `_pad_snapshot()`: 填充订单簿到K档
- `_mean_std()`: 计算均值和标准差
- `_is_finite_number()`: 检查数值有效性

---

## ✅ 验证标准

### 1. 语法检查
```bash
python -m py_compile v13_ofi_ai_system/src/real_ofi_calculator.py
```
**期望**: 无错误输出

### 2. 单元测试
```bash
cd v13_ofi_ai_system
pytest -q tests/test_real_ofi_calculator.py
```
**期望**: 所有测试通过
- `test_weights_valid`: 权重合法性
- `test_ofi_direction`: OFI方向正确性
- `test_warmup_behavior`: warmup行为

### 3. 导入测试
```python
from v13_ofi_ai_system.src.real_ofi_calculator import RealOFICalculator, OFIConfig
calc = RealOFICalculator("ETHUSDT")
print(calc.get_state())
```

---

## 📝 任务清单

### 开发任务
- [ ] 复制 `real_ofi_calculator.py` 到目标位置
- [ ] 创建测试目录 `v13_ofi_ai_system/tests/`（如不存在）
- [ ] 创建 `__init__.py` 文件（如需要）
- [ ] 复制测试文件到目标位置
- [ ] 运行语法检查
- [ ] 运行单元测试
- [ ] 更新任务卡 `Task_1.2.1_创建OFI计算器基础类.md`
- [ ] Git提交更改

### 验证任务
- [ ] ✅ 代码无语法错误
- [ ] ✅ 通过 lint 检查
- [ ] ✅ 通过所有测试
- [ ] ✅ 无 mock/占位/跳过
- [ ] ✅ 产出真实验证结果
- [ ] ✅ 更新相关文档

---

## ⚠️ 注意事项

### 禁止修改的文件
- ❌ `v13_ofi_ai_system/src/binance_websocket_client.py`
- ❌ `v13_ofi_ai_system/src/utils/async_logging.py`

### 必须遵守的规则
1. ✅ 只做纯计算，不做 I/O
2. ✅ 只用标准库（不用 numpy/pandas）
3. ✅ 仅实现 `update_with_snapshot`
4. ✅ `update_with_l2_delta` 保持 `NotImplementedError`
5. ✅ 提供完整的类型注解
6. ✅ 权重必须非负且归一化为1
7. ✅ 处理边界情况（无效数据、空数组等）

---

## 📈 预期结果

### 文件创建
```
v13_ofi_ai_system/
├── src/
│   ├── real_ofi_calculator.py (新建, 134行)
│   ├── binance_websocket_client.py (不改)
│   └── utils/
│       └── async_logging.py (不改)
└── tests/
    ├── __init__.py (可能新建)
    └── test_real_ofi_calculator.py (新建, 30行)
```

### 测试输出示例
```python
>>> from v13_ofi_ai_system.src.real_ofi_calculator import RealOFICalculator
>>> calc = RealOFICalculator("ETHUSDT")
>>> calc.get_state()
{
    'symbol': 'ETHUSDT',
    'levels': 5,
    'weights': [0.4, 0.25, 0.2, 0.1, 0.05],
    'bids': [[0.0, 0.0], [0.0, 0.0], ...],
    'asks': [[0.0, 0.0], [0.0, 0.0], ...],
    'bad_points': 0,
    'ema_ofi': None,
    'ofi_hist_len': 0
}
```

### OFI计算示例
```python
bids = [(100.0, 5.0), (99.9, 3.0), (99.8, 2.0), (99.7, 1.5), (99.6, 1.0)]
asks = [(100.1, 4.0), (100.2, 3.5), (100.3, 2.5), (100.4, 2.0), (100.5, 1.5)]

result = calc.update_with_snapshot(bids, asks, event_time_ms=1697527081000)

# result 结构：
{
    'symbol': 'ETHUSDT',
    'event_time_ms': 1697527081000,
    'ofi': 0.0,  # 首次计算，delta=0
    'k_components': [0.0, 0.0, 0.0, 0.0, 0.0],
    'z_ofi': None,  # warmup期
    'ema_ofi': 0.0,
    'meta': {
        'levels': 5,
        'weights': [0.4, 0.25, 0.2, 0.1, 0.05],
        'bad_points': 0,
        'warmup': True
    }
}
```

---

## 🔄 后续集成计划

### Task 1.2.2: 实现OFI核心算法
- 使用 `RealOFICalculator` 类
- 读取 NDJSON 数据文件
- 批量计算 OFI 值
- 输出 OFI 时间序列

### Task 1.2.3: 与WebSocket集成（可选）
- 在 `binance_websocket_client.py` 中添加回调
- 实时计算 OFI
- 保存 OFI 到数据文件

---

## 📚 参考文档

- **任务卡**: `v13_ofi_ai_system/TASKS/Stage1_真实OFI核心/Task_1.2.1_创建OFI计算器基础类.md`
- **参考代码**: `Task_1_2_1_OFI_Calculator_Pack/`
- **CURSOR提示词**: `Task_1_2_1_OFI_Calculator_Pack/CURSOR_PROMPT_Task_1_2_1.md`
- **使用规范**: `v13_ofi_ai_system/src/BINANCE_WEBSOCKET_CLIENT_USAGE.md`

---

## ✅ 确认清单

请确认以下内容后，我将开始执行：

- [ ] 理解OFI计算公式和算法
- [ ] 同意使用提供的参考代码
- [ ] 确认文件路径和目录结构
- [ ] 确认测试方法和验证标准
- [ ] 确认不修改 WebSocket 客户端和日志模块
- [ ] 准备好开始开发

---

**开发计划版本**: v1.0  
**创建时间**: 2025-10-17  
**预计执行时间**: 约15分钟（复制+测试）  
**风险评估**: 低（使用已验证的参考代码）

