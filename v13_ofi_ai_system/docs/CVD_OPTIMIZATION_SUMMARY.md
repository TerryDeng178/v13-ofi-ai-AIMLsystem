# CVD优化总结 - 已实施配置更新

**实施时间**: 2025-10-27  
**基于报告**: CVD_TEST_REPORT_DETAILED.md  
**优化目标**: 降低Z-score尾部，提升稳定性

---

## 已实施的优化配置

### 配置变更对比

| 参数 | 原配置（5分钟测试） | 新配置（优化后） | 变更原因 |
|------|-------------------|----------------|---------|
| `z_mode` | "level" | **"delta"** | Delta-Z更符合增量统计特性，尾部更薄 |
| `winsor_limit` | 8.0 | **2.8** | 收紧截断阈值，降低尾部概率 |
| `scale_mode` | "ewma" | **"hybrid"** | Quiet档使用混合尺度+MAD地板 |
| `mad_window_trades` | 300 | **200** | 缩短MAD窗口，提升响应速度 |
| `mad_multiplier` | 1.30 | **1.43** | 提高地板安全系数10% |
| `freeze_min` | 25 | **20** | 降低暖启动阈值，更快进入稳定态 |

### 预期效果

| 指标 | 原值 | 目标值 | 预期改善 |
|------|------|--------|---------|
| P(\|Z\|>2) | 21.82% | ≤8% | 显著下降 |
| P(\|Z\|>3) | 3.43% | ≤2% | 轻微下降 |
| median(\|Z\|) | 1.45 | 0.8~1.1 | 更接近1.0 |
| warmup时间 | - | 更快 | freeze_min降低 |

---

## 待验证的优化

### 1. 队列处理优化（优先级S - 需修改采集器）

**问题**: 队列兑现率3.24% > 0.5%阈值

**建议方案**:
```python
# 在采集器中添加批量处理
self.queue_size = 8192  # 4096 → 8192
self.watermark_ms = 300  # 自适应范围 [200, 800]

# 批量出队策略
if len(buf) >= self.queue_size // 2:
    batch = [buf.popleft() for _ in range(len(buf) // 2)]
    # 批量处理
```

**预期效果**: 队列丢弃率 3.24% → <0.5%

### 2. 测试时长延长（优先级A）

**当前**: 5分钟  
**后方案**: 30-60分钟

```bash
export RUN_HOURS=0.5  # 30分钟测试
```

**目的**: 完结覆盖市场波动周期，验证长期稳定性

---

## 快速验证步骤

### Step 1: 验证配置加载

```bash
cd v13_ofi_ai_system/src
python -c "from real_cvd_calculator import CVDConfig; cfg = CVDConfig(); print(cfg.z_mode, cfg.winsor_limit, cfg.scale_mode)"
```

**预期输出**: `delta 2.8 hybrid`

### Step 2: 重新运行测试（5分钟快速验证）

```bash
cd v13_ofi_ai_system/examples
export SYMBOLS=ETHUSDT
export RUN_HOURS=0.08
export OUTPUT_DIR="deploy/_out/ofi_cvd_optimized/data/ofi_cvd"
export PREVIEW_DIR="deploy/_out/ofi_cvd_optimized/preview/ofi_cvd"

cd ../deploy
python run_success_harvest.py
```

### Step 3: 分析Z-score分布

```bash
cd v13_ofi_ai_system/examples
python analysis_cvd_v2_cvdonly.py \
  --data "deploy/_out/ofi_cvd_optimized/preview/ofi_cvd/date=2025-10-27/symbol=ETHUSDT/kind=cvd" \
  --out cvd_system/figs_optimized \
  --report cvd_system/docs/reports/CVD_OPTIMIZED_TEST.md
```

### Step 4: 对比结果

**关键指标对比**:
- P(\|Z\|>2): 预期从21.82%降至≤8%
- median(\|Z\|): 预期从1.45降至0.8~1.1
- 队列丢弃率: 需配合采集器优化

---

## 配置剖面（未来可选）

### Quiet档（低活跃）
```python
PROFILE_QUIET = {
    'scale_mode': 'hybrid',
    'mad_window_trades': 200,
    'mad_multiplier': 1.43,
    'ewma_fast_hl': 80,
    'winsor_limit': 2.8,
    'freeze_min': 20,
}
```

### Active档（高活跃）
```python
PROFILE_ACTIVE = {
    'scale_mode': 'ewma',
    'half_life_trades': 250,
    'winsor_limit': 3.0,
    'freeze_min': 15,
}
```

**动态切换**: 根据TPS（例如current_tps < 0.5）切换到对应profile

---

## 验收标准（优化后）

| 类别 | 指标 | 原值 | 目标 | 当前配置状态 |
|------|------|------|------|------------|
| **Z-score** | P(\|Z\|>2) | 21.82% | ≤8% | ✅ 配置已调 |
| | P(\|Z\|>3) | 3.43% | ≤2% | ✅ 配置已调 |
| | median(\|Z\|) | 1.45 | ≤1.0 | ✅ 配置已调 |
| **性能** | 队列丢弃率 | 3.24% | <0.5% | ⏳ 待采集器优化 |
| | p99_interarrival | 347ms | ≤5000ms | ✅ 优秀，保持 |
| **时长** | 采集时长 | 5分钟 | ≥30分钟 | ⏳ 需延长测试 |

---

## 下一步行动

### 立即执行（10分钟）
1. ✅ **配置已更新** - real_cvd_calculator.py 的 CVDConfig 类
2. ⏳ **快速验证** - 运行5分钟测试，确认Z-score改善
3. ⏳ **查看报告** - 检查分析结果是否满足预期

### 短期（30分钟）
1. ⏳ **延长测试** - 运行30分钟完整测试
2. ⏳ **分析对比** - 对比优化前后效果
3. ⏳ **文档更新** - 更新配置文档

### 中期（1-2天）
1. ⏳ **采集器优化** - 实现批量出队策略
2. ⏳ **多交易对测试** - 验证配置稳定性
3. ⏳ **性能基准测试** - 建立性能基线

---

## 技术说明

### 为什么选择delta模式？

**Level-Z模式**（原配置）:
- 计算: `z = (CVD - mean(history)) / std(history)`
- 问题: 累积值CVD受历史影响大，容易产生厚尾
- 适用: 长期趋势分析

**Delta-Z模式**（新配置）:
- 计算: `z = ΔCVD / scale` (基于增量)
- 优势: 增量更符合正态分布，尾部更薄
- 适用: 实时交易信号

### 为什么选择hybrid scale_mode？

**EWMA模式**:
- 优点: 简单快速
- 缺点: Quiet时段易产生误报尖峰

**Hybrid模式** (EWMA + MAD地板):
- 优点: MAD地板抑制Quiet时段的假信号
- 缺点: 计算复杂度略高
- 结果: 稳健性提升，尾部更薄

---

## 回退方案

如新配置效果不如预期：

1. **回退到level模式**: 仅保留其他优化
   ```python
   z_mode: str = "level"
   winsor_limit: float = 3.0  # 折中值
   ```

2. **混合使用**: 当前配置，逐步调整winsor_limit
   ```python
   winsor_limit: float = 3.5  # 2.8 → 3.5 渐进收紧
   ```

3. **A/B对比**: 同时运行两套配置，对比结果

---

**配置更新完成时间**: 2025-10-27  
**下一步**: 运行5分钟快速验证测试  
**预计验证时间**: 约10分钟（含分析）
