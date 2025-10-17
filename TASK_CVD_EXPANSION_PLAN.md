# 📊 CVD功能扩展任务规划（方案A）

## 🎯 **方案概述**

**方案名称**: 方案A - 先完成纯OFI，再逐步加入CVD  
**设计理念**: 职责分离、模块化、易测试、易维证  
**当前位置**: Task 1.2.1 (创建OFI计算器基础类) - 待执行

---

## 📋 **现有任务结构（阶段1.2）**

### **当前 Task 1.2.X 任务**
```
Task 1.2.1: 创建OFI计算器基础类 ⏳ (当前任务)
Task 1.2.2: 实现OFI核心算法
Task 1.2.3: 实现OFI Z-score标准化
Task 1.2.4: 集成WebSocket和OFI计算
Task 1.2.5: OFI计算测试
```

---

## 🆕 **新增CVD任务（插入方案）**

### **方案A1: 在Task 1.2.X中插入CVD任务**

```
Task 1.2.1: 创建OFI计算器基础类 ⏳ (当前)
Task 1.2.2: 实现OFI核心算法
Task 1.2.3: 实现OFI Z-score标准化
Task 1.2.4: 集成WebSocket和OFI计算
Task 1.2.5: OFI计算测试

🆕 Task 1.2.6: 创建CVD计算器基础类 (新增)
🆕 Task 1.2.7: 实现CVD核心算法 (新增)
🆕 Task 1.2.8: 实现CVD标准化 (新增)
🆕 Task 1.2.9: 集成Trade流和CVD计算 (新增)
🆕 Task 1.2.10: CVD计算测试 (新增)
🆕 Task 1.2.11: OFI+CVD融合指标 (新增)
🆕 Task 1.2.12: OFI-CVD背离检测 (新增)
```

**优点**:
- ✅ 逻辑连贯（先OFI，再CVD，再融合）
- ✅ 职责清晰（每个任务独立）
- ✅ 易于测试和验证

**缺点**:
- ⚠️ 任务数量增多（从5个增到12个）
- ⚠️ 阶段1.2时间延长（从5天 → 8-10天）

---

## 📝 **详细新增任务定义**

### 🆕 **Task 1.2.6: 创建CVD计算器基础类**

#### **任务目标**
创建独立的CVD计算器类，处理成交数据

#### **任务清单**
- [ ] 创建文件 `v13_ofi_ai_system/src/real_cvd_calculator.py`
- [ ] 实现 `RealCVDCalculator` 类基础结构
- [ ] 定义CVD配置参数（窗口大小、重置周期等）
- [ ] 初始化历史CVD缓存

#### **技术规格**
```python
@dataclass
class CVDConfig:
    reset_period: Optional[int] = None  # CVD重置周期（秒），None=不重置
    z_window: int = 300                 # z-score滚动窗口
    ema_alpha: float = 0.2              # EMA平滑系数

class RealCVDCalculator:
    def __init__(self, symbol: str, cfg: CVDConfig = None):
        self.symbol = symbol
        self.cfg = cfg or CVDConfig()
        self.cumulative_delta = 0.0
        self.cvd_history = []
        ...
```

#### **验证标准**
- ✅ 文件创建成功
- ✅ 类结构正确
- ✅ 通过 `python -m py_compile`
- ✅ 无第三方依赖（只用标准库）

#### **预计时间**: 30分钟

#### **Allowed files**
- `src/real_cvd_calculator.py` (新建)

---

### 🆕 **Task 1.2.7: 实现CVD核心算法**

#### **任务目标**
实现CVD的核心计算逻辑

#### **任务清单**
- [ ] 实现 `update_with_trade()` 方法
- [ ] 判断成交方向（买方主动 vs 卖方主动）
- [ ] 累积成交量差
- [ ] 处理重置逻辑（可选）

#### **技术规格**
```python
def update_with_trade(
    self, 
    price: float, 
    qty: float, 
    is_buyer_maker: bool,  # Binance字段：True=卖方主动
    event_time_ms: int
) -> Dict[str, Any]:
    """
    更新CVD值
    
    参数:
        price: 成交价格
        qty: 成交数量
        is_buyer_maker: True表示卖方主动（买方挂单），False表示买方主动
        event_time_ms: 事件时间戳（毫秒）
    
    返回:
        {
            'symbol': 'ETHUSDT',
            'event_time_ms': 1697527081000,
            'cvd': 12345.67,           # 累积成交量差
            'cvd_delta': +10.5,        # 本次变化
            'direction': 'buy',        # 'buy' or 'sell'
            'z_cvd': 1.23,            # z-score (warmup后)
            'ema_cvd': 12000.0,       # EMA平滑值
            'meta': {...}
        }
    """
    # 判断方向
    if is_buyer_maker:
        # 买方挂单，卖方吃单 → 卖出压力
        delta = -qty
        direction = 'sell'
    else:
        # 卖方挂单，买方吃单 → 买入压力
        delta = +qty
        direction = 'buy'
    
    # 累积
    self.cumulative_delta += delta
    
    # 计算z-score、EMA等
    ...
    
    return result
```

#### **CVD计算公式**
```
CVD_t = CVD_{t-1} + Δ_t

其中:
Δ_t = {
    +qty,  如果是买方主动成交（买入压力）
    -qty,  如果是卖方主动成交（卖出压力）
}
```

#### **验证标准**
- ✅ CVD值正确累积
- ✅ 方向判断准确
- ✅ 处理边界情况（无效数据、异常值）
- ✅ 通过单元测试

#### **预计时间**: 2小时

#### **Allowed files**
- `src/real_cvd_calculator.py` (修改)
- `tests/test_real_cvd_calculator.py` (新建)

---

### 🆕 **Task 1.2.8: 实现CVD标准化**

#### **任务目标**
实现CVD的z-score标准化和EMA平滑

#### **任务清单**
- [ ] 实现 `get_cvd_zscore()` 方法
- [ ] 计算CVD均值和标准差
- [ ] 滚动窗口大小: 300个数据点
- [ ] 计算z-score
- [ ] 实现EMA平滑

#### **验证标准**
- ✅ z-score分布接近标准正态分布
- ✅ 强信号（|Z|>2）频率 5-10%
- ✅ EMA平滑有效
- ✅ 计算稳定

#### **预计时间**: 1小时

---

### 🆕 **Task 1.2.9: 集成Trade流和CVD计算**

#### **任务目标**
集成Binance WebSocket Trade流，实时计算CVD

#### **任务清单**
- [ ] 创建 `v13_ofi_ai_system/src/binance_trade_stream.py`
- [ ] 实现 `BinanceTradeStream` 类（类似于 `BinanceOrderBookStream`）
- [ ] 连接Binance WebSocket Trade流
- [ ] 解析成交数据
- [ ] 实时计算CVD

#### **技术规格**

**Binance Trade Stream URL**:
```python
wss://fstream.binancefuture.com/stream?streams={symbol}@aggTrade
```

**Trade数据格式**:
```json
{
  "e": "aggTrade",           // 事件类型
  "E": 1697527081000,        // 事件时间
  "s": "ETHUSDT",            // 交易对
  "a": 123456,               // 聚合成交ID
  "p": "1800.50",            // 成交价格
  "q": "10.5",               // 成交数量
  "f": 100,                  // 第一个成交ID
  "l": 105,                  // 最后一个成交ID
  "T": 1697527080900,        // 成交时间
  "m": true                  // 买方是否为挂单方（true=卖方主动）
}
```

#### **验证标准**
- ✅ 成功连接Trade流
- ✅ 数据解析正确
- ✅ CVD实时计算正常
- ✅ 无内存泄漏
- ✅ 符合 `BINANCE_WEBSOCKET_CLIENT_USAGE.md` 规范

#### **预计时间**: 2小时

#### **Allowed files**
- `src/binance_trade_stream.py` (新建)

---

### 🆕 **Task 1.2.10: CVD计算测试**

#### **任务目标**
运行CVD实时计算测试，验证稳定性和准确性

#### **任务清单**
- [ ] 创建测试脚本 `examples/run_realtime_cvd.py`
- [ ] 运行CVD实时计算30-60分钟
- [ ] 收集CVD数据
- [ ] 分析CVD分布和统计特性
- [ ] 验证CVD合理性

#### **验证标准**
- ✅ 连续运行 ≥30分钟
- ✅ 数据完整性 >95%
- ✅ CVD值范围合理
- ✅ z-score分布正态
- ✅ 无异常值或断点

#### **预计时间**: 2小时（30分钟测试 + 1.5小时分析）

---

### 🆕 **Task 1.2.11: OFI+CVD融合指标**

#### **任务目标**
创建OFI和CVD的融合指标，综合订单簿压力和成交压力

#### **任务清单**
- [ ] 创建文件 `v13_ofi_ai_system/src/ofi_cvd_fusion.py`
- [ ] 实现 `OFI_CVD_Fusion` 类
- [ ] 定义融合策略（加权平均、信号叠加等）
- [ ] 实现综合信号生成

#### **技术规格**

**融合策略选项**:

1. **加权平均**:
   ```
   Fusion = w_ofi * z_ofi + w_cvd * z_cvd
   
   其中: w_ofi + w_cvd = 1
   默认: w_ofi = 0.6, w_cvd = 0.4
   ```

2. **信号一致性**:
   ```
   强买入信号: z_ofi > 2 AND z_cvd > 1
   强卖出信号: z_ofi < -2 AND z_cvd < -1
   ```

3. **动态权重**:
   ```
   如果 |z_ofi| > |z_cvd|:
       权重更倾向OFI
   否则:
       权重更倾向CVD
   ```

#### **验证标准**
- ✅ 融合逻辑清晰
- ✅ 信号生成准确
- ✅ 参数可配置
- ✅ 通过单元测试

#### **预计时间**: 2小时

#### **Allowed files**
- `src/ofi_cvd_fusion.py` (新建)
- `tests/test_ofi_cvd_fusion.py` (新建)

---

### 🆕 **Task 1.2.12: OFI-CVD背离检测**

#### **任务目标**
实现OFI和CVD的背离检测，捕捉市场不一致信号

#### **任务清单**
- [ ] 在 `ofi_cvd_fusion.py` 中添加背离检测方法
- [ ] 定义背离条件
- [ ] 实现背离信号生成
- [ ] 测试背离检测效果

#### **技术规格**

**背离类型**:

1. **正向背离（潜在反转上涨）**:
   ```
   条件:
   - 价格创新低
   - OFI上升（买单压力增加）
   - CVD上升（买入成交增加）
   
   含义: 价格下跌但买盘在积累，可能反弹
   ```

2. **负向背离（潜在反转下跌）**:
   ```
   条件:
   - 价格创新高
   - OFI下降（卖单压力增加）
   - CVD下降（卖出成交增加）
   
   含义: 价格上涨但卖盘在积累，可能回调
   ```

3. **OFI-CVD不一致**:
   ```
   条件:
   - z_ofi > 2 BUT z_cvd < -1 (订单簿看涨，但成交看跌)
   - z_ofi < -2 BUT z_cvd > 1 (订单簿看跌，但成交看涨)
   
   含义: 市场信号不一致，谨慎交易
   ```

#### **实现示例**
```python
def detect_divergence(
    self, 
    price: float, 
    z_ofi: float, 
    z_cvd: float,
    lookback: int = 20
) -> Dict[str, Any]:
    """
    检测OFI-CVD背离
    
    返回:
        {
            'has_divergence': True/False,
            'divergence_type': 'bullish' / 'bearish' / 'inconsistent' / None,
            'strength': 0.0-1.0,  # 背离强度
            'action': 'buy' / 'sell' / 'wait',
            'confidence': 0.0-1.0
        }
    """
    ...
```

#### **验证标准**
- ✅ 背离检测逻辑正确
- ✅ 信号生成准确
- ✅ 回测验证有效性（准确率>55%）
- ✅ 通过单元测试

#### **预计时间**: 3小时

---

## 📊 **更新后的完整任务列表（阶段1.2）**

### **1.2 真实OFI+CVD计算（5-8天）**

| 任务编号 | 任务名称 | 状态 | 预计时间 | 类型 |
|---------|---------|------|---------|------|
| Task 1.2.1 | 创建OFI计算器基础类 | ⏳ 待开始 | 30分钟 | OFI |
| Task 1.2.2 | 实现OFI核心算法 | ⏳ 待开始 | 2小时 | OFI |
| Task 1.2.3 | 实现OFI Z-score标准化 | ⏳ 待开始 | 1小时 | OFI |
| Task 1.2.4 | 集成WebSocket和OFI计算 | ⏳ 待开始 | 1小时 | OFI |
| Task 1.2.5 | OFI计算测试 | ⏳ 待开始 | 2-4小时 | OFI |
| **🆕 Task 1.2.6** | **创建CVD计算器基础类** | **⏳ 待开始** | **30分钟** | **CVD** |
| **🆕 Task 1.2.7** | **实现CVD核心算法** | **⏳ 待开始** | **2小时** | **CVD** |
| **🆕 Task 1.2.8** | **实现CVD标准化** | **⏳ 待开始** | **1小时** | **CVD** |
| **🆕 Task 1.2.9** | **集成Trade流和CVD计算** | **⏳ 待开始** | **2小时** | **CVD** |
| **🆕 Task 1.2.10** | **CVD计算测试** | **⏳ 待开始** | **2小时** | **CVD** |
| **🆕 Task 1.2.11** | **OFI+CVD融合指标** | **⏳ 待开始** | **2小时** | **融合** |
| **🆕 Task 1.2.12** | **OFI-CVD背离检测** | **⏳ 待开始** | **3小时** | **融合** |

**总时间**: 原5天 → 新8-10天

---

## 🎯 **后续阶段调整建议**

### **阶段1.3: OFI+CVD信号验证**

由于加入了CVD，需要同步调整验证任务：

#### 原任务
- Task 1.3.1: 收集历史OFI数据
- Task 1.3.2: 创建OFI信号分析工具
- Task 1.3.3: 分析OFI预测能力
- Task 1.3.4: 生成OFI验证报告
- Task 1.3.5: 阶段1总结和决策

#### 调整为
- Task 1.3.1: 收集历史**OFI+CVD**数据
- Task 1.3.2: 创建**OFI+CVD**信号分析工具
- Task 1.3.3: 分析**OFI、CVD、融合指标**预测能力
- 🆕 **Task 1.3.3.5**: 分析**背离信号**预测能力
- Task 1.3.4: 生成**OFI+CVD**验证报告
- Task 1.3.5: 阶段1总结和决策

---

### **阶段2: 简单真实交易**

策略需要同时使用OFI和CVD：

#### 调整
- Task 2.2.1: 创建简单**OFI+CVD**策略类
- Task 2.2.2: 实现开仓逻辑（使用**融合信号**）
- Task 2.2.3: 实现平仓逻辑（考虑**背离信号**）

---

### **阶段3: 逐步加入AI**

AI特征需要包含CVD：

#### 调整
- Task 3.1.2: 特征工程
  - OFI相关特征（原有）
  - **🆕 CVD相关特征（新增）**
  - **🆕 OFI-CVD融合特征（新增）**
  - **🆕 背离特征（新增）**
  - 价格相关特征
  - 订单簿相关特征

---

## 📋 **新增文件清单**

### **新建文件**
```
v13_ofi_ai_system/
├── src/
│   ├── real_cvd_calculator.py (新建, 约120行)
│   ├── binance_trade_stream.py (新建, 约150行)
│   └── ofi_cvd_fusion.py (新建, 约200行)
├── tests/
│   ├── test_real_cvd_calculator.py (新建, 约40行)
│   └── test_ofi_cvd_fusion.py (新建, 约30行)
└── examples/
    └── run_realtime_cvd.py (新建, 约50行)
```

**总新增代码量**: 约590行

---

## 🔄 **依赖关系**

### **任务依赖图**
```
Task 1.2.1 (OFI基础类)
    ↓
Task 1.2.2 (OFI算法)
    ↓
Task 1.2.3 (OFI标准化)
    ↓
Task 1.2.4 (OFI集成)
    ↓
Task 1.2.5 (OFI测试)
    ↓
Task 1.2.6 (CVD基础类) ←──┐
    ↓                      │
Task 1.2.7 (CVD算法)       │ 可并行
    ↓                      │
Task 1.2.8 (CVD标准化)     │
    ↓                      │
Task 1.2.9 (CVD集成) ←─────┘
    ↓
Task 1.2.10 (CVD测试)
    ↓
Task 1.2.11 (融合指标) ←── 需要 Task 1.2.5 + Task 1.2.10
    ↓
Task 1.2.12 (背离检测)
```

---

## ⚠️ **注意事项**

### **1. 数据源不同**
- **OFI**: 订单簿数据（`@depth`）
- **CVD**: 成交数据（`@aggTrade`）
- **解决**: 需要同时维护两个WebSocket连接

### **2. 数据频率不同**
- **OFI**: 100ms更新（每秒10次）
- **CVD**: 成交实时推送（不定频率，取决于交易活跃度）
- **解决**: 需要时间对齐机制

### **3. 存储和回放**
- **OFI**: 已有NDJSON+Parquet存储
- **CVD**: 需要新增Trade数据存储
- **解决**: 复用 `async_logging.py` 和存储架构

### **4. 计算复杂度**
- **OFI**: 每次需计算5档差异
- **CVD**: 每次只需累加
- **融合**: 需要同步OFI和CVD的z-score
- **解决**: 优化计算，避免阻塞

---

## 🎯 **执行建议**

### **立即执行**
1. ✅ **先完成 Task 1.2.1 - 1.2.5（纯OFI）**
   - 验证OFI计算正确性
   - 建立基准信号质量

### **用户确认后执行**
2. ⏳ **创建7个新任务文件**（Task_1.2.6 - Task_1.2.12）
3. ⏳ **更新任务索引和任务卡**
4. ⏳ **逐步执行CVD任务**

### **阶段1.2完成后**
5. ⏳ **调整阶段1.3任务（加入CVD验证）**
6. ⏳ **调整阶段2任务（使用融合策略）**
7. ⏳ **调整阶段3任务（加入CVD特征）**

---

## 📊 **时间预估**

| 阶段 | 原预估 | 新预估 | 增加 |
|------|--------|--------|------|
| 阶段1.2 | 5天 | 8-10天 | +3-5天 |
| 阶段1.3 | 3天 | 4-5天 | +1-2天 |
| 阶段2 | 2-3天 | 3-4天 | +1天 |
| 阶段3 | 5-7天 | 6-8天 | +1天 |
| **总计** | **15-18天** | **21-27天** | **+6-9天** |

---

## ✅ **下一步操作**

### **等待用户确认**:
1. [ ] 是否同意在阶段1.2加入CVD任务？
2. [ ] 是否同意新增7个CVD相关任务？
3. [ ] 是否同意预计时间增加6-9天？
4. [ ] 是否需要调整任务优先级或细节？

### **确认后我将**:
1. [ ] 创建7个新任务文件（Task_1.2.6 - Task_1.2.12）
2. [ ] 更新 `TASK_INDEX.md`
3. [ ] 更新 `📋V13_TASK_CARD.md`
4. [ ] 提交Git更改
5. [ ] 继续执行 Task 1.2.1（OFI计算器基础类）

---

**文档版本**: v1.0  
**创建时间**: 2025-10-17  
**适用项目**: V13 OFI+AI System  
**当前任务**: Task 1.2.1 (OFI计算器基础类)  
**等待用户确认**: ✅ 是

