# CVD系统文件关联说明

## 📊 文档概述

**文档目的**: 完整说明CVD（Cumulative Volume Delta）系统的所有关联文件、组件、配置和使用方式  
**适用人员**: 开发者、分析师、运维人员  
**更新时间**: 2025-10-19

---

## 🗂️ 文件结构总览

```
v13_ofi_ai_system/
├── src/                          # 核心源代码
│   ├── real_cvd_calculator.py    # CVD核心算法实现
│   └── __init__.py
│
├── examples/                      # 示例和运行脚本
│   ├── run_realtime_cvd.py      # 实时CVD数据采集主程序
│   ├── analysis_cvd.py           # CVD数据分析脚本
│   └── test_delta_z.py           # Delta-Z模式测试
│
├── config/                        # 配置文件
│   ├── step_1_6_analysis.env    # Step 1.6分析模式配置（基线）
│   ├── step_1_6_fixed_gold.env  # Step 1.6修复版配置
│   └── step_1_6_clean_gold.env  # Step 1.6干净金测配置
│
├── data/                          # 数据输出目录
│   ├── cvd_step_1_6_microtune_ethusdt/    # Step 1.6测试数据
│   ├── cvd_clean_gold_test/               # 干净金测数据
│   ├── cvd_fixed_gold_test/               # 修复版测试数据
│   └── cvd_analysis_verify_v*/            # 验证测试数据
│
├── docs/                          # 文档
│   ├── reports/                  # 测试报告
│   │   ├── STEP_1_6_CLEAN_GOLD_TEST_REPORT.md
│   │   ├── ENGINEERING_FIXES_REPORT.md
│   │   ├── CODE_AUDIT_REPORT.md
│   │   └── ANALYSIS_MODE_VERIFICATION_FINAL.md
│   ├── CVDSYSTEM_ARCHITECTURE.md          # 系统架构
│   └── CVD_SYSTEM_FILES_GUIDE.md          # 本文档
│
└── TASKS/                         # 任务卡片
    ├── Task_1.2.10_CVD计算测试.md
    ├── Task_1.2.10.1_CVD问题修复（特别任务）.md
    └── Task_1.2.10.1_P1.2微调优化.md
```

---

## 🔧 核心组件详解

### 1. 核心算法 (`src/real_cvd_calculator.py`)

**功能**: CVD核心计算逻辑

**主要类**:
- `CVDConfig`: 配置数据类
- `RealCVDCalculator`: CVD计算器主类

**关键方法**:
```python
update_with_trade()      # 更新CVD计算
get_state()              # 获取当前状态
reset()                  # 重置状态
_z_delta()               # Delta-Z计算（Step 1.6基线）
_peek_delta_z()          # 只读Delta-Z估计
_robust_mad_sigma()      # 稳健MAD估计
```

**配置参数**:
```python
z_mode: str = "delta"                    # Z-score模式（delta/level）
half_life_trades: int = 300              # EWMA半衰期
winsor_limit: float = 8.0                # Winsorize截断限制
freeze_min: int = 80                     # 暖启动最小笔数
stale_threshold_ms: int = 5000           # 空窗阈值（毫秒）
scale_mode: str = "hybrid"               # 尺度计算模式（ewma/hybrid）
ewma_fast_hl: int = 80                   # 快速EWMA半衰期
scale_fast_weight: float = 0.35          # 快速EWMA权重
scale_slow_weight: float = 0.65          # 慢速EWMA权重
mad_window_trades: int = 300             # MAD窗口大小
mad_scale_factor: float = 1.4826         # MAD缩放因子
mad_multiplier: float = 1.45             # MAD乘数（地板高度）
post_stale_freeze: int = 2               # 空窗后冻结笔数
```

**关键特性**:
- ✅ Delta-Z标准化（Step 1.6基线）
- ✅ 混合尺度地板（双EWMA + MAD）
- ✅ 事件时间分段冻结（>5s→2笔，4-5s→1笔）
- ✅ Winsorize截断（±8σ）
- ✅ 权重自动归一化

---

### 2. 实时采集 (`examples/run_realtime_cvd.py`)

**功能**: 实时连接Binance WebSocket，采集aggTrade数据并计算CVD

**主要组件**:
- `CVDRecord`: 数据记录结构
- `MonitoringMetrics`: 监控指标
- `WatermarkBuffer`: 2秒水位线重排缓冲
- `ws_consume()`: WebSocket消费协程
- `processor()`: 数据处理协程

**运行方式**:
```bash
# 基本运行（使用环境变量配置）
cd v13_ofi_ai_system/examples
python run_realtime_cvd.py --symbol ETHUSDT --duration 2100 --output-dir ../data/test

# 使用配置文件
source ../config/step_1_6_analysis.env
python run_realtime_cvd.py --symbol ETHUSDT --duration 2100
```

**环境变量配置**:
```bash
# ============================================
# Step 1.6基线配置（默认值）
# ============================================

# 基础配置
CVD_Z_MODE=delta             # Z-score模式（delta/level）
HALF_LIFE_TRADES=300         # EWMA半衰期
WINSOR_LIMIT=8.0             # Winsorize截断限制
FREEZE_MIN=80                # 暖启动最小笔数
STALE_THRESHOLD_MS=5000      # 空窗阈值（毫秒）

# 混合尺度地板
SCALE_MODE=hybrid            # 尺度计算模式（ewma/hybrid）
EWMA_FAST_HL=80              # 快速EWMA半衰期
SCALE_FAST_WEIGHT=0.35       # 快速EWMA权重（和为1.0 ✓）
SCALE_SLOW_WEIGHT=0.65       # 慢速EWMA权重
MAD_WINDOW_TRADES=300        # MAD窗口大小
MAD_SCALE_FACTOR=1.4826      # MAD缩放因子
MAD_MULTIPLIER=1.45          # MAD乘数（地板高度）

# 水位线配置
WATERMARK_MS=2000            # 水位线延迟（分析模式：2s）

# 空窗后冻结配置（显式声明）
POST_STALE_FREEZE=2          # >5s 空窗后首2笔不产Z
HARD_FREEZE_MS=5000          # 硬冻结阈值（>5s）
SOFT_FREEZE_MS=4000          # 软冻结阈值（4-5s，首1笔不产Z）

# 队列策略（分析模式：阻塞不丢）★ 重要
DROP_OLD=false               # 分析模式一律阻塞不丢，确保数据完整性
                             # 实时模式灰度稳定后可改为true

# 日志和性能配置
PRINT_EVERY=1000             # 每1000条打印一次（减少日志开销）
METRICS_FLUSH_INTERVAL_MS=10000      # 每10秒刷新指标
WATERMARK_FLUSH_INTERVAL_MS=200      # 每200ms强制flush水位线
```

**⚠️ 关键配置说明**:

1. **DROP_OLD策略** (强烈建议显式设置):
   - **分析模式**: `DROP_OLD=false` (阻塞不丢，确保数据完整)
   - **实时模式**: 初期`DROP_OLD=false`，灰度稳定后可改为`true`

2. **权重和为1.0检查**:
   - `SCALE_FAST_WEIGHT + SCALE_SLOW_WEIGHT = 1.0` ✓
   - 代码会自动归一化，但建议配置文件保持和为1.0

3. **冻结参数显式声明**:
   - `POST_STALE_FREEZE=2`: >5s空窗后首2笔不产Z
   - `HARD_FREEZE_MS=5000`: 硬冻结阈值
   - `SOFT_FREEZE_MS=4000`: 软冻结阈值（4-5s）
```

**输出产物**:
- `cvd_{symbol}_{timestamp}.parquet`: CVD数据（Parquet格式）
- `report_{symbol}_{timestamp}.json`: 运行报告

**关键特性**:
- ✅ 分析模式：队列阻塞不丢（DROP_OLD=false）
- ✅ 2秒水位线重排（基于event_time_ms）
- ✅ 定时flush（每200ms）
- ✅ ID健康监控（去重/去倒序）
- ✅ Step 1.6默认参数

---

### 3. 数据分析 (`examples/analysis_cvd.py`)

**功能**: 分析CVD测试数据，生成报告和图表

**运行方式**:
```bash
cd v13_ofi_ai_system/examples
python analysis_cvd.py --input ../data/cvd_test/cvd_ethusdt_*.parquet --output-dir ../docs/reports
```

**分析指标**:

#### 数据质量（8/8质量门控）
1. **时长**: ≥30分钟（分析模式）
2. **ID健康**: agg_dup_rate==0、backward_rate≤0.5%
3. **连续性**: p99_interarrival≤5s、gaps_over_10s==0
4. **数据质量**: parse_errors==0、queue_dropped_rate≤0.5%
5. **延迟**: P95仅展示（分析模式不阻断）
6. **Z质量**: median(|Z|)≤1.0、P(|Z|>2)≤8%、P(|Z|>3)≤2%
7. **一致性**: 逐笔守恒=0、首尾守恒≈0（相对容差）
8. **稳定性**: reconnect_count≤3次/小时

**生成图表**:
- `hist_z.png`: Z-score分布直方图
- `z_timeseries.png`: Z-score时间序列
- `event_id_diff.png`: aggTradeId差值分布
- `interarrival_hist.png`: 到达间隔直方图
- `latency_box.png`: 延迟箱线图

**输出报告**:
- `CVD_TEST_REPORT.md`: Markdown格式报告
- `analysis_results.json`: JSON格式结果

**关键特性**:
- ✅ 分析模式口径（p99_interarrival、gaps_over_10s）
- ✅ Z质量判定（P(|Z|>2)、P(|Z|>3)、median|Z|）
- ✅ 首尾守恒相对容差
- ✅ 延迟P95仅展示不阻断

---

## 📁 配置文件详解

### 配置文件快速选择指南

根据使用场景，选择合适的配置文件：

| 配置文件 | 用途 | DROP_OLD | WATERMARK_MS | 适用场景 |
|---------|------|----------|--------------|---------|
| **profiles/analysis.env** | 分析模式（生产） | **false** ✓ | 2000ms | 离线分析、研究、低风险灰度 |
| **profiles/realtime.env** | 实时模式（生产） | false → true | 500ms | 实时触发、高频交易、灰度放量 |
| **step_1_6_analysis.env** | Step 1.6基线 | **false** ✓ | 2000ms | 测试验证、参数基线 |
| **step_1_6_fixed_gold.env** | 修复版金测 | **false** ✓ | 2000ms | 长时间测试（35-40分钟） |

**⚠️ 关键原则**:
- ✅ **分析模式**: 一律`DROP_OLD=false`（阻塞不丢）
- ✅ **实时模式**: 初期`DROP_OLD=false`，灰度稳定后可改为`true`
- ✅ **权重和为1.0**: `SCALE_FAST_WEIGHT + SCALE_SLOW_WEIGHT = 1.0`
- ✅ **冻结参数**: 显式声明`POST_STALE_FREEZE`、`HARD_FREEZE_MS`、`SOFT_FREEZE_MS`

---

### Step 1.6基线配置 (`config/step_1_6_analysis.env`)

**用途**: Step 1.6参数微调的最优配置（生产基线）

**完整配置内容**:
```bash
# 基础配置
CVD_Z_MODE=delta
HALF_LIFE_TRADES=300
WINSOR_LIMIT=8.0
STALE_THRESHOLD_MS=5000
FREEZE_MIN=80

# 混合尺度地板
SCALE_MODE=hybrid
EWMA_FAST_HL=80
SCALE_FAST_WEIGHT=0.35       # 快速权重（和为1.0 ✓）
SCALE_SLOW_WEIGHT=0.65       # 慢速权重
MAD_WINDOW_TRADES=300
MAD_SCALE_FACTOR=1.4826
MAD_MULTIPLIER=1.45

# 水位线配置
WATERMARK_MS=2000

# 空窗后冻结配置（显式声明）
POST_STALE_FREEZE=2          # >5s 空窗后首2笔不产Z
HARD_FREEZE_MS=5000          # 硬冻结阈值（>5s）
SOFT_FREEZE_MS=4000          # 软冻结阈值（4-5s，首1笔不产Z）

# 队列策略（分析模式：阻塞不丢）★ 关键
DROP_OLD=false               # 分析模式一律阻塞不丢，确保数据完整性

# 日志和性能配置
PRINT_EVERY=1000             # 每1000条打印一次（减少日志开销）
```

**⚠️ 关键参数说明**:

1. **DROP_OLD=false** (★ 必须设置)
   - 分析模式**一律阻塞不丢**，确保数据完整性
   - 队列满时阻塞等待，不丢弃任何消息
   - 用于离线分析、研究评估、低风险策略灰度

2. **权重检查** (自动归一化)
   - `SCALE_FAST_WEIGHT + SCALE_SLOW_WEIGHT = 1.0` ✓
   - 代码会自动归一化，但配置文件保持和为1.0更直观

3. **冻结参数** (显式声明)
   - `POST_STALE_FREEZE=2`: >5s空窗后首2笔不产Z
   - `HARD_FREEZE_MS=5000`: 硬冻结阈值（>5s）
   - `SOFT_FREEZE_MS=4000`: 软冻结阈值（4-5s，首1笔不产Z）
   - 代码已内置默认值，但显式声明更清晰

**适用场景**:
- ✅ 离线分析（首选）
- ✅ 研究评估
- ✅ 低风险策略灰度
- ✅ 数据质量验证

**性能指标** (基于20分钟测试):
- P(|Z|>2) = 5.73% ✅ (目标≤8%)
- P(|Z|>3) = 4.65% (目标≤2%，可优化)
- median(|Z|) = 0.0013 ✅
- 队列丢弃率 = 0% ✅
- 数据质量、ID健康、一致性：全绿 ✅

---

### 生产配置文件 (`config/profiles/`)

CVD系统提供两套生产配置，位于`config/profiles/`目录：

#### 分析模式配置 (`profiles/analysis.env`)

**用途**: 离线分析、研究评估、低风险策略灰度

**关键配置**:
```bash
# 水位线配置（分析模式 - 稳健优先）
WATERMARK_MS=2000            # 2秒水位线，确保数据完整性

# 队列策略（★ 分析模式核心）
DROP_OLD=false               # 一律阻塞不丢，确保数据完整性

# 日志和性能配置
PRINT_EVERY=1000             # 每1000条打印一次
METRICS_FLUSH_INTERVAL_MS=10000      # 每10秒刷新指标
WATERMARK_FLUSH_INTERVAL_MS=200      # 每200ms强制flush水位线
```

**适用场景**:
- ✅ 离线数据分析
- ✅ 策略研究和回测
- ✅ 低风险策略灰度验证
- ✅ 质量门控验证

**特点**:
- 数据完整性优先（DROP_OLD=false）
- 2秒水位线（稳健重排）
- 适合长时间运行（小时级别）

---

#### 实时模式配置 (`profiles/realtime.env`)

**用途**: 实时触发、高频交易、灰度放量

**关键配置**:
```bash
# 水位线配置（实时模式 - 低延迟）
WATERMARK_MS=500             # 500ms水位线，降低延迟

# 队列策略（★ 实时模式核心）
DROP_OLD=false               # 初期保持阻塞不丢，灰度稳定后可改为true
# 灰度稳定后可选：DROP_OLD=true（丢弃旧消息，保证低延迟）

# 日志和性能配置
WATERMARK_FLUSH_INTERVAL_MS=100      # 每100ms强制flush（更频繁）
```

**适用场景**:
- ✅ 实时信号触发
- ✅ 高频交易策略
- ✅ 生产环境灰度
- ✅ 低延迟要求场景

**特点**:
- 低延迟优先（500ms水位线）
- 初期阻塞不丢（DROP_OLD=false），灰度稳定后可切换
- 更频繁的flush（100ms）

**⚠️ 灰度切换建议**:
```bash
# 第一阶段：灰度验证（1-2周）
DROP_OLD=false               # 保持阻塞不丢，确保数据质量

# 第二阶段：灰度稳定后（验证通过）
DROP_OLD=true                # 切换为丢弃旧消息，优化延迟
```

---

### 测试配置文件

#### 修复版配置 (`config/step_1_6_fixed_gold.env`)

**用途**: 工程层面修复后的配置，用于长时间金测

**新增配置**:
```bash
# 队列和写盘优化
PRINT_EVERY=1000      # 每1000条打印一次
DROP_OLD=false        # 分析模式：阻塞不丢
METRICS_FLUSH_INTERVAL_MS=10000      # 每10秒刷新指标
WATERMARK_FLUSH_INTERVAL_MS=200      # 每200ms强制flush
```

**修复项**:
- ✅ 队列策略：分析模式阻塞不丢
- ✅ 日志优化：每1000条打印
- ✅ 批量写盘：减少I/O阻塞
- ✅ 定时flush：每200ms
- ✅ 指标刷新：每10秒

---

## 📊 数据流程

### 1. 数据采集流程

```
Binance WebSocket
      ↓
  WS消费协程
      ↓
   队列（50k）─→ DROP_OLD控制
      ↓
数据处理协程
      ↓
WatermarkBuffer（2s重排）
      ↓         ↓
   定时flush   新事件触发
      ↓         ↓
CVD计算器（Delta-Z）
      ↓
  记录保存
      ↓
Parquet导出
```

### 2. Z-score计算流程

```
aggTrade数据
      ↓
计算Delta（±qty）
      ↓
更新EWMA（慢速、快速）
      ↓
更新MAD缓冲区
      ↓
暖启动检查（≥80笔）
      ↓
计算混合尺度
  - ewma_mix = 0.35*fast + 0.65*slow
  - sigma_floor = 1.4826 * MAD * 1.45
  - scale = max(ewma_mix, sigma_floor)
      ↓
事件时间冻结检查
  - E间隔>5s → 首2笔冻结
  - 4-5s → 首1笔冻结
      ↓
计算Delta-Z
  - Z = Delta / scale
      ↓
Winsorize截断（±8σ）
      ↓
输出Z-score
```

---

## 🧪 测试流程

### 快速验证测试（5-10分钟）

**目的**: 验证配置正确、基础功能正常

```bash
cd v13_ofi_ai_system/examples

# 加载Step 1.6配置
source ../config/step_1_6_analysis.env

# 运行5分钟测试
python run_realtime_cvd.py \
  --symbol ETHUSDT \
  --duration 300 \
  --output-dir ../data/quick_test

# 检查结果
python analysis_cvd.py \
  --input ../data/quick_test/cvd_*.parquet \
  --output-dir ../docs/reports/quick_test
```

**预期结果**:
- ✅ 队列丢弃率 = 0%
- ✅ 解析错误 = 0
- ✅ 配置正确加载
- ✅ Z-score正常计算

---

### 干净金测（35-40分钟）

**目的**: 验证长时间稳定性、Z质量达标

```bash
cd v13_ofi_ai_system/examples

# 加载修复版配置
source ../config/step_1_6_fixed_gold.env

# 运行40分钟测试
python run_realtime_cvd.py \
  --symbol ETHUSDT \
  --duration 2400 \
  --output-dir ../data/gold_test

# 生成完整报告
python analysis_cvd.py \
  --input ../data/gold_test/cvd_*.parquet \
  --output-dir ../docs/reports/gold_test
```

**验收标准（8/8质量门控）**:
1. ✅ 时长 ≥ 35分钟
2. ✅ ID健康: agg_dup_rate==0、backward_rate≤0.5%
3. ✅ 连续性: p99_interarrival≤5s、gaps_over_10s==0
4. ✅ 数据质量: queue_dropped_rate≤0.5%
5. ✅ 延迟: P95仅展示
6. ✅ Z质量: median(|Z|)≤1.0、P(|Z|>2)≤8%、P(|Z|>3)≤2%
7. ✅ 一致性: 逐笔守恒=0、首尾守恒≈0
8. ✅ 稳定性: reconnect≤3次/小时

---

## 📋 常见问题排查

### Q1: 队列丢弃率过高（>0.5%）

**症状**: `queue_dropped_rate` 高于0.5%，数据丢失

**排查步骤**:
1. **检查DROP_OLD配置**（★ 最重要）
   ```bash
   # 检查当前配置
   echo $DROP_OLD
   
   # 应该输出：false（分析模式）
   ```

2. **检查配置文件加载**
   ```bash
   # 确认配置文件正确加载
   source config/profiles/analysis.env
   echo $DROP_OLD  # 应输出：false
   ```

3. **检查队列大小**
   - 默认队列大小：50000
   - 查看代码：`queue = asyncio.Queue(maxsize=50000)`

4. **检查系统资源**
   ```bash
   # 检查CPU和内存使用率
   python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"
   ```

5. **检查日志频率**
   - 过高的日志频率会导致I/O阻塞
   - 建议：`PRINT_EVERY=1000`

**解决方案**:

**方案1: 确保分析模式配置** (首选)
```bash
# 1. 使用profiles/analysis.env
source config/profiles/analysis.env

# 2. 验证关键参数
echo "DROP_OLD=$DROP_OLD"           # 应为：false
echo "PRINT_EVERY=$PRINT_EVERY"     # 应为：1000
echo "WATERMARK_MS=$WATERMARK_MS"   # 应为：2000

# 3. 运行测试
python run_realtime_cvd.py --symbol ETHUSDT --duration 600
```

**方案2: 显式设置环境变量**
```bash
# 强制设置DROP_OLD=false
export DROP_OLD=false
export PRINT_EVERY=1000
export METRICS_FLUSH_INTERVAL_MS=10000

python run_realtime_cvd.py --symbol ETHUSDT --duration 600
```

**方案3: 检查系统资源瓶颈**
```bash
# 如果资源不足，降低采集频率或增加队列大小
# 修改run_realtime_cvd.py中的队列大小：
# queue = asyncio.Queue(maxsize=100000)  # 从50000增加到100000
```

**⚠️ 关键提醒**:
- **分析模式必须`DROP_OLD=false`**，这是硬性要求
- **实时模式初期也应`DROP_OLD=false`**，灰度稳定后再切换
- 队列丢弃率>0.5%会导致数据完整性问题，必须解决

---

### Q2: Z-score质量不达标

**症状**: P(|Z|>2) > 8% 或 P(|Z|>3) > 2%

**排查**:
1. 检查配置是否正确加载（查看启动日志"Effective config"）
2. 检查`z_mode`是否为`delta`
3. 检查混合尺度参数（weights、multiplier）
4. 检查数据量是否充足（>1000笔）

**解决**:
```bash
# 验证配置
grep "Z_MODE" logs/startup.log
grep "SCALE_FAST_WEIGHT" logs/startup.log

# 如仍不达标，尝试S7-A微调
export MAD_MULTIPLIER=1.47
```

---

### Q3: 首尾守恒误差过大

**症状**: `conservation_error` 超过容差

**排查**:
1. 检查是否使用相对容差（v1.12+已修复）
2. 检查数据完整性（无丢失）
3. 检查精度问题（浮点累计误差）

**解决**:
- 已在`analysis_cvd.py`中修复为相对容差
- 容差公式: `max(1e-6, 1e-8 * abs(cvd_last - cvd_first))`

---

### Q4: 连续性检查失败

**症状**: p99_interarrival > 5000ms 或 gaps_over_10s > 0

**排查**:
1. 检查网络稳定性
2. 检查重连次数（应≤3次/小时）
3. 检查水位线flush是否正常

**解决**:
```bash
# 检查重连次数
grep "reconnect_count" data/*/report_*.json

# 确保定时flush启用
grep "force_flush_timeout" src/run_realtime_cvd.py
```

---

## 🚀 快速开始指南

### 1分钟快速测试

```bash
# 1. 进入目录
cd v13_ofi_ai_system/examples

# 2. 运行5分钟测试（使用默认Step 1.6配置）
python run_realtime_cvd.py --symbol ETHUSDT --duration 300 --output-dir ../data/quick_test

# 3. 查看结果
python analysis_cvd.py --input ../data/quick_test/cvd_*.parquet
```

---

### 生产环境部署

```bash
# 1. 固化配置
cp config/step_1_6_analysis.env config/profiles/analysis.env

# 2. 启动采集（后台运行）
nohup python run_realtime_cvd.py \
  --symbol ETHUSDT \
  --duration 86400 \
  --output-dir ../data/production \
  > logs/cvd_$(date +%Y%m%d).log 2>&1 &

# 3. 监控
tail -f logs/cvd_$(date +%Y%m%d).log

# 4. 定期分析
python analysis_cvd.py \
  --input ../data/production/cvd_*.parquet \
  --output-dir ../docs/reports/daily
```

---

## 📚 相关文档

### 核心文档
- [CVD系统架构](CVDSYSTEM_ARCHITECTURE.md)
- [Step 1.6干净金测报告](reports/STEP_1_6_CLEAN_GOLD_TEST_REPORT.md)
- [工程修复报告](reports/ENGINEERING_FIXES_REPORT.md)
- [代码审查报告](reports/CODE_AUDIT_REPORT.md)

### 任务卡片
- [Task 1.2.10: CVD计算测试](../TASKS/Stage1_真实OFI+CVD核心/Task_1.2.10_CVD计算测试.md)
- [Task 1.2.10.1: CVD问题修复](../TASKS/Stage1_真实OFI+CVD核心/Task_1.2.10.1_CVD问题修复（特别任务）.md)
- [Task 1.2.10.1: P1.2微调优化](../TASKS/Stage1_真实OFI+CVD核心/Task_1.2.10.1_P1.2微调优化.md)

---

## 📞 技术支持

### 遇到问题？

1. 查看本文档的"常见问题排查"部分
2. 检查`docs/reports/`下的相关报告
3. 查看代码注释和docstring
4. 运行单元测试（如有）

### 重要提醒

- ✅ 使用Step 1.6基线配置作为默认
- ✅ 分析模式必须设置`DROP_OLD=false`
- ✅ 长时间测试前检查系统资源
- ✅ 定期备份数据和配置
- ✅ 监控队列丢弃率和重连次数

---

*文档版本: v1.0*  
*最后更新: 2025-10-19*  
*维护者: V13 OFI+CVD+AI System*
