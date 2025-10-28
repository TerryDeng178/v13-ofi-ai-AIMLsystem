# CVD计算器优化实施指南

**基于**: CVD_TEST_REPORT_DETAILED.md  
**目标**: 降低队列丢弃率，优化Z-score分布，提升生产稳定性  
**优先级**: S（立即实施）+ A（本轮完成）

---

## 问题诊断总结

### 当前状态（5分钟测试）
- ✅ 核心功能正常（8,466笔无错误）
- ❌ 队列丢弃率：3.24% > 0.5%
- ❌ Z-score尾部过厚：P(|Z|>2) = 21.82% > 8%
- ❌ Z中位数偏高：median(|Z|) = 1.45 > 1.0
- ⚠️ 测试时长不足：5分钟 < 30分钟

### 根本原因
1. **Z计算使用旧口径**: `z_mode=level` + `winsor_limit=8.0` 放大了尾部
2. **队列处理效率低**: 顺序出队 + 固定watermark
3. **配置不一致**: WebSocket订阅与实际symbol不匹配

---

## 优先级S：立即实施（不改采集器代码）

### 1.1 切换Z计算口径到Delta-Z

**修改位置**: `CVDConfig` 类默认值

```python
# real_cvd_calculator.py
@dataclass
class CVDConfig:
    z_mode: str = "delta"        # level → delta
    winsor_limit: float = 2.8    # 8.0 → 2.8
    freeze_min: int = 20         # 25 → 20
    half_life_trades: int = 300  # 保持不变
    
    # Quiet档配置（disk_cvd）
    scale_mode: str = "hybrid"
    mad_window_trades: int = 200   # 300 → 200
    mad_multiplier: float = 1.43   # 1.30 → 1.43
    ewma_fast_hl: int = 80         # 保持不变
    scale_fast_weight: float = 0.30
    scale_slow_weight: float = 0.70
```

**预期效果**:
- P(|Z|>2): 21.82% → ≤8%
- median(|Z|): 1.45 → 0.8~1.1
- 更符合增量统计特性

### 1.2 队列优化（采集器侧）

**修改位置**: `run_success_harvest.py` 的配置段落

```python
# 在 __init__ 方法中添加
self.watermark_ms = int(os.getenv('WATERMARK_MS', '300'))  # 自适应阈值
self.queue_size = int(os.getenv('QUEUE_SIZE', '8192'))     # 4096 → 8192
self.diag_sample_n = int(os.getenv('DIAG_SAMPLE_N', '50')) # 降低日志频率
```

**处理优化**（仅修改处理逻辑）:
```python
# 批量出队 + 水位检查
def _batch_process_queue(self, symbol):
    buf = self.data_buffers[kind][symbol]
    if len(buf) < self.queue_size // 2:
        return []  # 低于水位线，不处理
    
    # 批量取出队列的一半
    batch = []
    while len(buf) > 0 and len(batch) < len(buf) // 2:
        batch.append(buf.popleft())
    
    return batch
```

**预期效果**:
- 队列丢弃率: 3.24% → <0.5%
- 延迟尾部降低
- 吞吐量提升

### 1.3 配置一致性修复

**修改位置**: 测试脚本环境变量

```bash
# 确保WebSocket订阅与实际symbol一致
export SYMBOLS=ETHUSDT  # 已正确
# 生成URL时为: ethusdt@aggTrade (而不是btcusdt)
```

**验证**: 检查采集日志中的实际订阅流

---

## 优先级A：本轮完成

### 2.1 分场景Z参数（Quiet/Active）

**创建配置**: `CVD_SCALE_MODE_PROFILES`

```python
# 在 real_cvd_calculator.py 中添加
PROFILES = {
    'quiet': {
        'scale_mode': 'hybrid',
        'mad_window_trades': 200,
        'mad_multiplier': 1.43,
        'ewma_fast_hl': 80,
        'winsor_limit': 2.8,
        'freeze_min': 20,
    },
    'active': {
        'scale_mode': 'ewma',
        'mad_window_trades': 300,
        'mad_multiplier': 1. Mahmoud's half_life_trades': 250,
        'winsor_limit': 3.0,
        'freeze_min': 15,
    }
}
```

**动态切换**（建议在以后集成）:
```python
# 根据TPS切换profile
if current_tps < 0.5:  # Quiet
    profile = PROFILES['quiet']
else:  # Active
    profile = PROFILES['active']

self.cfg.scale_mode = profile['scale_mode']
self.cfg.mad_multiplier = profile['mad_multiplier']
```

### 2.2 测试时长延长

**修改测试配置**:
```bash
export RUN_HOURS=0.5  # 30分钟（基础完整测试）
# 或
export RUN_HOURS=1.0  # 60分钟（覆盖完整session）
```

---

## 实施步骤（按序执行）

### Step 1: 应用CVD配置优化

修改 `real_cvd_calculator.py`:
1. `z_mode`: "level" → "delta"
2. `winsor_limit`: 8.0 → 2.8
3. `scale_mode`: "ewma" → "hybrid"（先固定Quiet档）
4. `mad_multiplier`: 1.30 → 1.43

### Step 2: 优化队列处理（可选）

如采集器性能成为瓶颈，可调整：
- `queue_size`: 4096 → 8192
- `watermark_ms`: 300 → 自适应
- 实现批量出队

### Step 3: 重新测试（30分钟）

```bash
cd v13_ofi_ai_system/examples
python test_cvd_shadow_collection.py
# 或手动设置环境变量并运行采集器
```

### Step 4: 验证结果

对照新验收标准：
- ✅ 队列丢弃率 < 0.5%
- ✅ P(|Z|>2) ≤ 8%
- ✅ median(|Z|) ≤ 1.0
- ✅ 时长 ≥ 30分钟

---

## 建议的配置基线（可直接套用）

### 生产配置文件 `cvd_config.yaml`

```yaml
components:
  cvd:
    # 基础Z计算
    z_mode: delta
    winsor_limit: 2.8
    freeze_min: 20
    z_window: 150
    
    # 半衰期
    half_life_trades: 300
    ema_alpha: 0.2
    
    # Quiet档（默认）
    scale_mode: hybrid
    mad_window_trades: 200
    mad_multiplier: 1.43
    ewma_fast_hl: 80
    scale_fast_weight: 0.30
    scale_slow_weight: 0.70
    
    # Active档（待动态切换）
    active:
      scale_mode: ewma
      half_life_trades: 250
      winsor_limit: 3.0
    
    # 冻结配置
    soft_freeze_ms: 4000
    hard_freeze_ms: 5000
    post_stale_freeze: 2

stream:
  watermark_ms: 300        # 自适应范围 [200, 800]
  queue_size: 8192
  diag_sample_n: 50

fusion:
  w_ofi: 0.6
  w_cvd: 0.4
  fuse_buy: 1.2
  fuse_sell: -1.2
  min_consistency: 0.2
```

---

## 验收标准（优化后）

| 指标 | 当前 | 目标 | 方法 |
|------|------|------|------|
| 队列丢弃率 | 3.24% | <0.5% | 队列优化 + 批量处理 |
| median(\|Z\|) | 1.45 | ≤1.0 | delta模式 + winsor=2.8 |
| P(\|Z\|>2) | 21.82% | ≤8% | 同上 |
| P(\|Z\|>3) | 3.43% | ≤2% | 同上 |
| 采集时长 | 5分钟 | ≥30分钟 | 延长测试 |
| 连续性 | 优秀 | 保持 | 无需改 |

---

## 回退方案

如delta模式不如预期：

1. **保留level模式**：仅调整winsor_limit: 8.0 → 3.0
2. **渐进式启用**：先测1分钟，确认Z分布改善后再延长时间
3. **A/B对比**：同时跑level和delta，对比结果

---

**实施责任人**: 开发团队  
**预计时间**: 30分钟（配置） + 30分钟测试  
**验证方式**: 重新运行 `test_cvd_shadow_collection.py` 并分析报告
