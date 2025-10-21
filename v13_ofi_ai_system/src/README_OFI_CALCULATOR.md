# RealOFICalculator 使用说明 (L1 OFI版本)

## 📋 概述

`RealOFICalculator` 是 V13 系统的核心OFI（Order Flow Imbalance）计算组件，基于订单簿快照计算L1价跃迁敏感的OFI指标。

**模块**: `v13_ofi_ai_system/src/real_ofi_calculator.py` *(以项目实际路径为准)*  
**任务**: Task 1.2.1 - 创建OFI计算器基础类 (L1价跃迁敏感版本)  
**创建时间**: 2025-10-17  
**最后更新**: 2025-10-21 (L1 OFI价跃迁敏感版本)

---

## 🎯 核心功能

1. **L1 OFI计算**: 最优价跃迁敏感版本，检测价格跃迁冲击
2. **加权OFI计算**: 5档订单簿深度加权
3. **Z-score标准化**: 滚动窗口标准化（优化版，"上一窗口"基线）
4. **EMA平滑**: 指数移动平均平滑
5. **数据清洗**: 自动处理无效数据

---

## 🚀 快速开始

### 基本使用

```python
from real_ofi_calculator import RealOFICalculator, OFIConfig

# 1. 创建配置（可选，使用默认配置）
config = OFIConfig(
    levels=5,           # 订单簿档位数
    z_window=300,       # Z-score滚动窗口
    ema_alpha=0.2       # EMA平滑系数
)

# 2. 初始化计算器
calc = RealOFICalculator("ETHUSDT", config)

# 3. 准备订单簿数据
bids = [
    [3245.5, 10.5],   # [价格, 数量] 按价格降序
    [3245.4, 8.3],
    [3245.3, 12.1],
    [3245.2, 5.8],
    [3245.1, 9.2]
]

asks = [
    [3245.6, 11.2],   # [价格, 数量] 按价格升序
    [3245.7, 9.5],
    [3245.8, 7.8],
    [3245.9, 13.4],
    [3246.0, 6.9]
]

# 4. 计算OFI（可选传入事件时间戳，单位毫秒）
result = calc.update_with_snapshot(bids, asks, event_time_ms=1697567890123)

# 5. 使用结果
print(f"OFI: {result['ofi']:.4f}")
print(f"Z-score: {result['z_ofi']:.4f}")
print(f"EMA: {result['ema_ofi']:.4f}")
```

---

## 📊 返回值说明

`update_with_snapshot()` 返回一个字典：

```python
{
    "symbol": "ETHUSDT",              # 交易对
    "event_time_ms": 1697567890123,   # 事件时间戳（毫秒，可选）
    "ofi": 0.1234,                    # 原始OFI值
    "k_components": [                 # 各档OFI贡献
        0.05,  # 档位0贡献
        0.03,  # 档位1贡献
        ...
    ],
    "z_ofi": 1.25,                    # Z-score标准化值（warmup期为None）
    "ema_ofi": 0.0987,                # EMA平滑值
    "meta": {
        "levels": 5,                  # 档位数
        "weights": [0.4, 0.25, ...],  # 权重列表
        "bad_points": 0,              # 坏数据点计数
        "warmup": False,              # 是否在warmup期
        "std_zero": False             # 标准差是否为0
    }
}
```

### 字段详解

| 字段 | 类型 | 说明 | 注意事项 |
|------|------|------|----------|
| `ofi` | float | 原始OFI值 | 可正可负，绝对值越大代表不平衡越严重 |
| `z_ofi` | float\|None | Z-score标准化值 | warmup期间为None，标准差为0时为0.0 |
| `ema_ofi` | float | EMA平滑值 | 首次等于当前ofi，之后递推更新 |
| `k_components` | List[float] | 各档贡献 | 用于验证：sum(k_components) ≈ ofi |
| `meta.warmup` | bool | warmup状态 | 历史数据不足时为True |
| `meta.std_zero` | bool | 标准差为0标记 | 用于监控数据质量 |

---

## ⚙️ 配置参数

### OFIConfig 参数

```python
@dataclass
class OFIConfig:
    levels: int = 5                          # 订单簿档位数（1-20）
    weights: Optional[List[float]] = None    # 自定义权重（None=标准权重）
    z_window: int = 300                      # Z-score滚动窗口大小
    ema_alpha: float = 0.2                   # EMA平滑系数（0-1）
```

#### 参数说明

**levels** (档位数)
- 默认: 5
- 范围: ≥1 (实践建议 1-20)
- 说明: 使用订单簿前N档计算OFI
- 推荐: 5档（币安深度快照标准）

**weights** (权重)
- 默认: `None` (使用标准权重 `[0.4, 0.25, 0.2, 0.1, 0.05]`)
- 说明: 自定义各档权重，负值会被截为0再归一化
- 示例: `[0.5, 0.3, 0.2]` 表示只使用3档

**z_window** (Z-score窗口)
- 默认: 300
- 范围: 10-10000
- 说明: 滚动窗口大小，用于计算均值和标准差
- 推荐: 300 (约6分钟 @ 50Hz)

**ema_alpha** (EMA系数)
- 默认: 0.2
- 范围: 0.0-1.0
- 说明: EMA平滑系数，越大对当前值越敏感
- 推荐: 0.1-0.3

---

## 🔧 高级用法

### 1. 自定义权重

```python
# 只使用前3档，权重分别为50%, 30%, 20%
config = OFIConfig(
    levels=3,
    weights=[0.5, 0.3, 0.2]
)
calc = RealOFICalculator("BTCUSDT", config)
```

### 2. 状态管理

```python
# 获取当前状态
state = calc.get_state()
print(f"历史数据点数: {state['ofi_hist_len']}")
print(f"坏数据点数: {state['bad_points']}")

# 重置计算器
calc.reset()
```

### 3. 处理warmup期

```python
result = calc.update_with_snapshot(bids, asks)

if result['meta']['warmup']:
    print("⚠️ 数据预热中，Z-score暂不可用")
    # 可以使用原始OFI或EMA
    print(f"使用EMA: {result['ema_ofi']:.4f}")
else:
    print(f"Z-score: {result['z_ofi']:.4f}")
```

### 4. 监控数据质量

```python
result = calc.update_with_snapshot(bids, asks)

# 检查标准差为0情况
if result['meta']['std_zero']:
    print("⚠️ 标准差为0，数据可能异常")

# 检查坏数据点
if result['meta']['bad_points'] > 0:
    print(f"⚠️ 检测到 {result['meta']['bad_points']} 个坏数据点")
```

---

## 📝 典型使用场景

### 场景1: 实时OFI监控

```python
calc = RealOFICalculator("ETHUSDT", OFIConfig(z_window=300))

while True:
    # 获取最新订单簿快照
    bids, asks = get_orderbook_snapshot()
    
    # 计算OFI
    result = calc.update_with_snapshot(bids, asks)
    
    # 判断信号
    if not result['meta']['warmup']:
        z = result['z_ofi']
        if z > 2.0:
            print("🟢 强买入信号")
        elif z < -2.0:
            print("🔴 强卖出信号")
```

### 场景2: 回测分析

```python
calc = RealOFICalculator("BTCUSDT", OFIConfig(z_window=300))
ofi_series = []

# 遍历历史数据
for snapshot in historical_snapshots:
    bids, asks = snapshot['bids'], snapshot['asks']
    result = calc.update_with_snapshot(bids, asks)
    
    # 收集OFI序列
    ofi_series.append({
        'timestamp': snapshot['timestamp'],
        'ofi': result['ofi'],
        'z_ofi': result['z_ofi'],
        'ema_ofi': result['ema_ofi']
    })

# 分析OFI特征
analyze_ofi_predictive_power(ofi_series)
```

### 场景3: 与WebSocket集成

```python
# 实际应用请参考 run_realtime_ofi.py 脚本
# 以下为集成示意（简化版）

from real_ofi_calculator import RealOFICalculator, OFIConfig

# 创建OFI计算器
calc = RealOFICalculator("ethusdt", OFIConfig())

# 在WebSocket消息处理回调中使用
def process_orderbook_message(msg):
    # 解析消息获取bids/asks
    bids = msg['bids'][:5]  # 前5档买单
    asks = msg['asks'][:5]  # 前5档卖单
    event_time_ms = msg['E']  # 事件时间（毫秒）
    
    # 计算OFI
    result = calc.update_with_snapshot(bids, asks, event_time_ms)
    
    # 实时打印
    if not result['meta']['warmup']:
        print(f"[{event_time_ms}] OFI={result['ofi']:+.4f} "
              f"Z={result['z_ofi']:+.2f}")

# 完整实现请参考:
# - run_realtime_ofi.py: 生产级WebSocket集成
# - README_realtime_ofi.md: 完整使用文档
```

---

## ⚠️ 注意事项

### 数据格式要求

1. **订单簿格式**
   - Bids: `[[price, qty], ...]` 按价格**降序**排列
   - Asks: `[[price, qty], ...]` 按价格**升序**排列
   - 价格: 有限正数
   - 数量: 非负数

2. **无效数据处理**
   - 价格为 `NaN`/`Inf`: 自动设为0.0
   - 数量为负: 自动设为0.0
   - 无效数据计入 `bad_points`

### 性能优化

1. **档位数选择**
   - 5档: 标准配置，性能良好
   - >10档: 计算量增加，收益递减
   - 推荐: 5档

2. **窗口大小**
   - 窗口过小: Z-score不稳定
   - 窗口过大: 响应迟钝
   - 推荐: 300 (约6分钟 @ 50Hz)

3. **内存占用**
   - 逻辑量纲: 8 bytes/值 (float64)
   - Python实际开销更高（deque对象开销）
   - 300窗口量级可忽略不计

### 常见问题

**Q: 为什么前几百个点 `z_ofi` 都是 `None`?**  
A: 这是warmup期，历史数据不足。默认需要 `max(5, z_window//5)` 个数据点。

**Q: `std_zero=True` 是什么意思?**  
A: 标准差≤1e-9，通常表示数据静止不变。此时 `z_ofi` 设为0.0。

**Q: 如何处理连接断开重连?**  
A: 建议调用 `reset()` 清空历史数据，重新warmup。

**Q: `k_components` 有什么用?**  
A: 用于验证计算正确性：`sum(k_components) ≈ ofi` (误差<1e-9)。

---

## 🔗 相关文档

- `run_realtime_ofi.py` - 实时OFI运行脚本
- `analysis.py` - OFI数据分析工具
- `README_realtime_ofi.md` - 实时运行完整文档
- `BINANCE_WEBSOCKET_CLIENT_USAGE.md` - WebSocket客户端文档

---

## 📊 技术细节

### L1 OFI计算公式

```
对于最优档位 (k=0):
    if 价格跃迁:
        bid_impact = 新最优价队列 - 旧最优价队列
        ask_impact = 新最优价队列 - 旧最优价队列
        OFI_0 = w_0 × (bid_impact - ask_impact)
    else:
        OFI_0 = w_0 × (Δbid_0 - Δask_0)  # 标准数量变化

对于其余档位 (k=1 to K-1):
    OFI_k = w_k × (Δbid_k - Δask_k)

总OFI:
    OFI = Σ OFI_k
```

**L1价跃迁冲击逻辑**:
- 价上涨：新最优价队列为正冲击，旧队列为负冲击
- 价下跌：旧最优价队列为负冲击，新队列为正冲击
- 价格不变：使用标准数量变化

### Z-score标准化（优化版）

```
基线 = ofi_hist[:-1]  # "上一窗口"，不包含当前值
mean = mean(基线)
std = std(基线)

if std > 1e-9:
    z_ofi = (ofi - mean) / std
else:
    z_ofi = 0.0
    std_zero = True
```

**优化要点**:
- ✅ 避免当前值稀释基线统计量
- ✅ 标准差为0时显式标记
- ✅ warmup阈值: `max(5, z_window//5)`

### EMA更新

```
if ema_ofi is None:
    ema_ofi = ofi  # 首次初始化
else:
    ema_ofi = alpha × ofi + (1-alpha) × ema_ofi_prev
```

---

## 🎯 最佳实践

1. **参数配置**
   - 生产环境: `OFIConfig(levels=5, z_window=300, ema_alpha=0.2)`
   - 快速响应: `ema_alpha=0.3`, `z_window=150`
   - 稳定信号: `ema_alpha=0.1`, `z_window=600`

2. **数据验证**
   - 定期检查 `bad_points` 计数
   - 监控 `std_zero` 标记
   - 验证 `sum(k_components) ≈ ofi`

3. **状态管理**
   - 连接断开后调用 `reset()`
   - 定期保存 `get_state()` 用于监控
   - 避免频繁reset（会丢失历史数据）

4. **性能监控**
   - 计算延迟应 <0.1ms (单次)
   - 内存占用稳定
   - CPU占用极低

---

**版本**: V13.1.2.1 (L1 OFI版本)  
**最后更新**: 2025-10-21  
**维护者**: V13 OFI+CVD+AI System Team

