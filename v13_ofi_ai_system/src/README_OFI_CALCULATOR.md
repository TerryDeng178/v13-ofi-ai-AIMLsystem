# RealOFICalculator 使用说明 (全局统一基线版本)

## 📋 概述

`RealOFICalculator` 是 V13 系统的核心OFI（Order Flow Imbalance）计算组件，基于订单簿快照计算L1价跃迁敏感的OFI指标，现已集成全局统一基线配置体系。

**模块**: `v13_ofi_ai_system/src/real_ofi_calculator.py` *(以项目实际路径为准)*  
**任务**: Task 1.2.5 - OFI计算测试与全局基线配置  
**创建时间**: 2025-10-17  
**最后更新**: 2025-10-27 (全局统一基线配置版本)

---

## 🎯 核心功能

1. **L1 OFI计算**: 最优价跃迁敏感版本，检测价格跃迁冲击
2. **加权OFI计算**: 5档订单簿深度加权
3. **Z-score标准化**: 滚动窗口标准化（优化版，"上一窗口"基线）
4. **EMA平滑**: 指数移动平均平滑
5. **数据清洗**: 自动处理无效数据
6. **全局基线配置**: 分层配置体系（Global → Profile → Regime → Symbol override）
7. **尾部监控**: 实时监控P(|z|>2)和P(|z|>3)指标
8. **动态参数调整**: 支持运行时参数更新和配置热重载

---

## 🚀 快速开始

### 基本使用

```python
from real_ofi_calculator import RealOFICalculator, OFIConfig
from ofi_config_parser import OFIConfigParser

# 方式1: 使用全局基线配置（推荐）
parser = OFIConfigParser("config/defaults.yaml")
config = parser.get_ofi_config("ETHUSDT", "offline_eval", "active")
calc = RealOFICalculator("ETHUSDT", config)

# 方式2: 手动创建配置（兼容旧版本）
config = OFIConfig(
    levels=5,                    # 订单簿档位数
    z_window=80,                 # Z-score滚动窗口（高流动性-活跃）
    ema_alpha=0.30,             # EMA平滑系数
    z_clip=None,                # Z-score裁剪（None=禁用）
    winsorize_ofi_delta=3.0,    # Winsorize MAD系数
    std_floor=1e-7              # 标准差下限
)

# 3. 初始化计算器
calc = RealOFICalculator("ETHUSDT", config)

# 4. 准备订单簿数据
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

# 5. 计算OFI（可选传入事件时间戳，单位毫秒）
result = calc.update_with_snapshot(bids, asks, event_time_ms=1697567890123)

# 6. 使用结果
print(f"OFI: {result['ofi']:.4f}")
print(f"Z-score: {result['z_ofi']:.4f}")
print(f"EMA: {result['ema_ofi']:.4f}")

# 7. 监控尾部指标（新增）
meta = result['meta']
print(f"P(|z|>2): {meta['p_gt2_percent']:.2f}%")
print(f"P(|z|>3): {meta['p_gt3_percent']:.2f}%")
print(f"总样本数: {meta['total_cnt']}")
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
        "std_zero": False,            # 标准差是否为0
        # 新增尾部监控指标
        "p_gt2_cnt": 45,              # |z|>2的样本数
        "p_gt3_cnt": 8,               # |z|>3的样本数
        "total_cnt": 1000,            # 总样本数
        "p_gt2_percent": 4.5,         # P(|z|>2)百分比
        "p_gt3_percent": 0.8          # P(|z|>3)百分比
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
| `meta.p_gt2_cnt` | int | \|z\|>2样本数 | 累计计数，用于尾部监控 |
| `meta.p_gt3_cnt` | int | \|z\|>3样本数 | 累计计数，用于尾部监控 |
| `meta.total_cnt` | int | 总样本数 | 累计计数，用于计算百分比 |
| `meta.p_gt2_percent` | float | P(\|z\|>2)百分比 | 实时计算，目标范围1-8% |
| `meta.p_gt3_percent` | float | P(\|z\|>3)百分比 | 实时计算，目标≤1.5% |

---

## ⚙️ 配置参数

### 全局基线配置体系

现在推荐使用全局基线配置体系，支持分层配置：

```yaml
# config/defaults.yaml
ofi:
  profiles:
    offline_eval:                    # 离线评估配置
      z_clip: null                   # 禁用Z-score裁剪
      winsor_k_mad: 3.0              # Winsorize MAD系数
      std_floor: 1e-7                # 标准差下限
      regimes:
        high_liquidity:
          active: { z_window: 80,  ema_alpha: 0.30 }   # 高流动性-活跃
          quiet:  { z_window: 120, ema_alpha: 0.25 }   # 高流动性-安静
        low_liquidity:
          active: { z_window: 120, ema_alpha: 0.20 }   # 低流动性-活跃
          quiet:  { z_window: 180, ema_alpha: 0.20 }   # 低流动性-安静
    online_prod:                     # 线上生产配置
      z_clip: 3.0                    # 启用Z-score裁剪
      winsor_k_mad: 3.0              # Winsorize MAD系数
      std_floor: 1e-7                # 标准差下限
      regimes:
        # ... 与offline_eval相同的regime配置
```

### OFIConfig 参数（兼容模式）

```python
@dataclass
class OFIConfig:
    levels: int = 5                          # 订单簿档位数（1-20）
    weights: Optional[List[float]] = None    # 自定义权重（None=标准权重）
    z_window: int = 80                       # Z-score滚动窗口大小（基线值）
    ema_alpha: float = 0.30                  # EMA平滑系数（基线值）
    z_clip: Optional[float] = None          # Z-score裁剪阈值（None=禁用）
    winsorize_ofi_delta: float = 3.0         # Winsorize MAD系数
    std_floor: float = 1e-7                  # 标准差下限
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
- 默认: 80 (高流动性-活跃基线)
- 范围: 10-10000
- 说明: 滚动窗口大小，用于计算均值和标准差
- 基线值: 高流动性活跃80，安静120；低流动性活跃120，安静180

**ema_alpha** (EMA系数)
- 默认: 0.30 (高流动性-活跃基线)
- 范围: 0.0-1.0
- 说明: EMA平滑系数，越大对当前值越敏感
- 基线值: 高流动性活跃0.30，安静0.25；低流动性0.20

**z_clip** (Z-score裁剪)
- 默认: None (禁用裁剪)
- 范围: None 或 >0
- 说明: Z-score裁剪阈值，None表示禁用
- 推荐: 离线评估None，线上生产3.0

**winsorize_ofi_delta** (Winsorize系数)
- 默认: 3.0
- 范围: >0
- 说明: MAD-based Winsorize软裁剪系数
- 推荐: 3.0 (已优化，避免过度裁剪)

**std_floor** (标准差下限)
- 默认: 1e-7
- 范围: >0
- 说明: 标准差下限，避免分母过小
- 推荐: 1e-7 (已优化)

---

## 🔧 高级用法

### 1. 使用全局基线配置

```python
from ofi_config_parser import OFIConfigParser

# 创建配置解析器
parser = OFIConfigParser("config/defaults.yaml")

# 获取不同场景的配置
# 离线评估 - 高流动性 - 活跃
config_offline = parser.get_ofi_config("BTCUSDT", "offline_eval", "active")

# 线上生产 - 低流动性 - 安静
config_online = parser.get_ofi_config("XRPUSDT", "online_prod", "quiet")

# 创建计算器
calc_btc = RealOFICalculator("BTCUSDT", config_offline)
calc_xrp = RealOFICalculator("XRPUSDT", config_online)
```

### 2. 动态参数调整

```python
# 运行时更新参数
updated = calc.update_params({
    'z_window': 100,      # 更新窗口大小
    'ema_alpha': 0.25,    # 更新EMA系数
    'z_clip': 2.5         # 更新裁剪阈值
})

if updated:
    print(f"参数已更新: {updated}")
    # 计算器会自动重建ofi_hist队列和重新计算权重
```

### 3. 监控尾部指标

```python
result = calc.update_with_snapshot(bids, asks)
meta = result['meta']

# 检查尾部指标是否在正常范围
p_gt2 = meta['p_gt2_percent']
p_gt3 = meta['p_gt3_percent']

if 1.0 <= p_gt2 <= 8.0:
    print(f"✅ P(|z|>2)正常: {p_gt2:.2f}%")
else:
    print(f"⚠️ P(|z|>2)异常: {p_gt2:.2f}% (目标: 1-8%)")

if p_gt3 <= 1.5:
    print(f"✅ P(|z|>3)正常: {p_gt3:.2f}%")
else:
    print(f"⚠️ P(|z|>3)异常: {p_gt3:.2f}% (目标: ≤1.5%)")
```

---

## 📝 典型使用场景

### 场景1: 实时OFI监控（全局基线版本）

```python
from ofi_config_parser import OFIConfigParser

# 使用全局基线配置
parser = OFIConfigParser("config/defaults.yaml")
config = parser.get_ofi_config("ETHUSDT", "online_prod", "active")
calc = RealOFICalculator("ETHUSDT", config)

while True:
    # 获取最新订单簿快照
    bids, asks = get_orderbook_snapshot()
    
    # 计算OFI
    result = calc.update_with_snapshot(bids, asks)
    
    # 判断信号
    if not result['meta']['warmup']:
        z = result['z_ofi']
        meta = result['meta']
        
        # 信号判断
        if z > 2.0:
            print("🟢 强买入信号")
        elif z < -2.0:
            print("🔴 强卖出信号")
        
        # 监控尾部指标
        if meta['p_gt2_percent'] > 8.0:
            print(f"⚠️ 尾部过宽: P(|z|>2)={meta['p_gt2_percent']:.2f}%")
```

### 场景2: 回测分析（离线评估模式）

```python
from ofi_config_parser import OFIConfigParser

# 使用离线评估配置（禁用z_clip）
parser = OFIConfigParser("config/defaults.yaml")
config = parser.get_ofi_config("BTCUSDT", "offline_eval", "active")
calc = RealOFICalculator("BTCUSDT", config)

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
        'ema_ofi': result['ema_ofi'],
        'p_gt2_percent': result['meta']['p_gt2_percent'],
        'p_gt3_percent': result['meta']['p_gt3_percent']
    })

# 分析OFI特征和尾部分布
analyze_ofi_predictive_power(ofi_series)
```

### 场景3: 多交易对监控（2×2场景）

```python
from ofi_config_parser import OFIConfigParser

# 创建配置解析器
parser = OFIConfigParser("config/defaults.yaml")

# 定义监控的交易对和场景
symbols = {
    "BTCUSDT": "high",    # 高流动性
    "ETHUSDT": "high",    # 高流动性
    "XRPUSDT": "low",     # 低流动性
    "DOGEUSDT": "low"     # 低流动性
}

# 创建多个计算器实例
calculators = {}
for symbol, liquidity in symbols.items():
    # 根据流动性选择regime
    regime = "active"  # 或根据市场状态动态选择"quiet"
    config = parser.get_ofi_config(symbol, "online_prod", regime)
    calculators[symbol] = RealOFICalculator(symbol, config)

# 批量处理订单簿数据
def process_multiple_symbols(orderbook_data):
    results = {}
    for symbol, data in orderbook_data.items():
        if symbol in calculators:
            calc = calculators[symbol]
            result = calc.update_with_snapshot(data['bids'], data['asks'])
            results[symbol] = result
    return results
```

---

## ⚠️ 注意事项

### 全局基线配置要求

1. **配置文件路径**
   - 确保 `config/defaults.yaml` 存在且格式正确
   - 使用 `OFIConfigParser` 验证配置完整性

2. **Profile和Regime选择**
   - `offline_eval`: 用于测试和评估，z_clip=null
   - `online_prod`: 用于生产环境，z_clip=3.0
   - `active/quiet`: 根据市场状态动态选择

3. **尾部监控指标**
   - P(|z|>2): 目标范围 1-8%
   - P(|z|>3): 目标 ≤1.5%
   - 超出范围时考虑调整参数

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

1. **基线参数选择**
   - 高流动性-活跃: z_window=80, ema_alpha=0.30
   - 高流动性-安静: z_window=120, ema_alpha=0.25
   - 低流动性-活跃: z_window=120, ema_alpha=0.20
   - 低流动性-安静: z_window=180, ema_alpha=0.20

2. **内存占用**
   - 逻辑量纲: 8 bytes/值 (float64)
   - Python实际开销更高（deque对象开销）
   - 基线窗口量级可忽略不计

### 常见问题

**Q: 如何选择Profile和Regime?**  
A: 离线测试用`offline_eval`，生产环境用`online_prod`。根据市场活跃度选择`active`或`quiet`。

**Q: P(|z|>2)超出1-8%范围怎么办?**  
A: 检查z_clip和winsorize设置，考虑调整z_window或ema_alpha参数。

**Q: 为什么前几百个点 `z_ofi` 都是 `None`?**  
A: 这是warmup期，历史数据不足。基线配置下需要约16-36个数据点。

**Q: `std_zero=True` 是什么意思?**  
A: 标准差≤1e-7，通常表示数据静止不变。此时 `z_ofi` 设为0.0。

**Q: 如何处理连接断开重连?**  
A: 建议调用 `reset()` 清空历史数据，重新warmup。

**Q: `k_components` 有什么用?**  
A: 用于验证计算正确性：`sum(k_components) ≈ ofi` (误差<1e-9)。

**Q: 如何监控尾部指标?**  
A: 使用`meta['p_gt2_percent']`和`meta['p_gt3_percent']`实时监控。

---

## 🔗 相关文档

- `config/defaults.yaml` - 全局基线配置文件
- `src/ofi_config_parser.py` - 配置解析器
- `examples/ofi_monitoring_system.py` - 监控告警系统
- `examples/gray_validation.py` - 灰度验证框架
- `examples/test_layered_config.py` - 分层配置测试
- `examples/OFI_GLOBAL_BASELINE_IMPLEMENTATION_REPORT.md` - 实施报告
- `examples/analysis.py` - OFI数据分析工具

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

1. **全局基线配置**
   - 优先使用 `OFIConfigParser` 获取配置
   - 离线评估用 `offline_eval` profile
   - 生产环境用 `online_prod` profile
   - 根据市场状态选择 `active/quiet` regime

2. **尾部监控**
   - 实时监控 `p_gt2_percent` 和 `p_gt3_percent`
   - P(|z|>2) 目标范围: 1-8%
   - P(|z|>3) 目标上限: ≤1.5%
   - 超出范围时考虑参数调整

3. **数据验证**
   - 定期检查 `bad_points` 计数
   - 监控 `std_zero` 标记
   - 验证 `sum(k_components) ≈ ofi`

4. **状态管理**
   - 连接断开后调用 `reset()`
   - 定期保存 `get_state()` 用于监控
   - 避免频繁reset（会丢失历史数据）

5. **性能监控**
   - 计算延迟应 <0.1ms (单次)
   - 内存占用稳定
   - CPU占用极低

---

**版本**: V13.1.2.5 (全局统一基线版本)  
**最后更新**: 2025-10-27  
**维护者**: V13 OFI+CVD+AI System Team  
**Git标签**: v1.0-global-baseline

