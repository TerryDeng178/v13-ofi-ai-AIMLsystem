# 统一配置集成计划

## 📊 当前状态总结

### ✅ 已完成集成的组件
1. **OFI计算器** - `real_ofi_calculator.py`
2. **CVD计算器** - `real_cvd_calculator.py`
3. **融合指标** - `ofi_cvd_fusion.py`
4. **WebSocket客户端** - `binance_websocket_client.py`
5. **Grafana配置** - `grafana_config.py`
6. **背离检测指标** - `divergence_metrics.py` (部分完成)
7. **融合指标导出** - `fusion_prometheus_exporter.py` (部分完成)

### ❌ 待集成的组件
1. **背离检测核心** - `ofi_cvd_divergence.py`
2. **策略模式管理器** - `strategy_mode_manager.py`
3. **交易流处理** - `binance_trade_stream.py`
4. **融合指标收集器** - `fusion_metrics.py`

---

## 🎯 集成计划

### 阶段1：背离检测核心配置集成 (优先级：🔴 高)

#### 目标
将背离检测模块的所有硬编码参数迁移到统一配置系统

#### 硬编码参数清单
```python
# ofi_cvd_divergence.py - DivergenceConfig
swing_L: int = 12                # 枢轴检测窗口长度
ema_k: int = 5                   # EMA平滑参数
z_hi: float = 1.5                # 高强度阈值
z_mid: float = 0.7               # 中等强度阈值
min_separation: int = 6          # 最小枢轴间距
cooldown_secs: float = 1.0       # 冷却时间
warmup_min: int = 100            # 暖启动最小样本数
max_lag: float = 0.300           # 最大滞后时间
use_fusion: bool = True          # 是否使用融合指标
```

#### 实施步骤
1. **扩展system.yaml** - 添加 `divergence_detection` 配置节
   ```yaml
   divergence_detection:
     # 枢轴检测参数
     swing_L: 12
     ema_k: 5
     
     # 强度阈值
     z_hi: 1.5
     z_mid: 0.7
     
     # 去噪参数
     min_separation: 6
     cooldown_secs: 1.0
     warmup_min: 100
     max_lag: 0.300
     
     # 融合参数
     use_fusion: true
   ```

2. **创建配置加载器** - `src/divergence_config_loader.py`
   ```python
   class DivergenceConfigLoader:
       def __init__(self, config_loader):
           self.config_loader = config_loader
       
       def load_config(self) -> DivergenceConfig:
           # 从统一配置加载参数
           pass
   ```

3. **修改OFI_CVD_Divergence类** - 支持从配置加载器初始化
   ```python
   def __init__(self, cfg: DivergenceConfig = None, config_loader=None):
       if config_loader:
           self.cfg = self._load_from_config_loader(config_loader)
       else:
           self.cfg = cfg or DivergenceConfig()
   ```

4. **创建测试脚本** - `test_divergence_config.py`
5. **更新文档** - 添加配置使用示例

**预计时间**: 2-3小时  
**优先级**: 🔴 高 (影响交易决策)

---

### 阶段2：策略模式管理器配置集成 (优先级：🟡 中)

#### 目标
将策略模式管理器的时间表和市场阈值配置迁移到统一系统

#### 硬编码参数清单
```python
# strategy_mode_manager.py
# 时间表配置
# 市场活跃度阈值
# 迟滞逻辑参数
# 模式切换规则
```

#### 实施步骤
1. **扩展system.yaml** - 添加 `strategy_mode` 配置节
   ```yaml
   strategy_mode:
     # 默认模式
     default_mode: "active"
     
     # 时间表配置
     schedule:
       active_hours:
         - start: "09:00"
           end: "16:00"
           timezone: "Asia/Hong_Kong"
       quiet_hours:
         - start: "00:00"
           end: "09:00"
         - start: "16:00"
           end: "24:00"
     
     # 市场活跃度阈值
     thresholds:
       trades_per_min: 100
       quote_updates_per_sec: 10
       spread_bps: 5
       volatility_bps: 10
     
     # 迟滞参数
     hysteresis:
       min_duration_secs: 300
       cooldown_secs: 60
   ```

2. **创建配置加载器** - `src/strategy_mode_config_loader.py`
3. **修改StrategyModeManager类** - 集成配置加载器
4. **创建测试脚本** - `test_strategy_mode_config.py`
5. **更新文档**

**预计时间**: 2-3小时  
**优先级**: 🟡 中 (影响系统运行模式)

---

### 阶段3：交易流处理配置集成 (优先级：🟢 低)

#### 目标
将交易流处理的WebSocket和队列参数迁移到统一配置

#### 硬编码参数清单
```python
# binance_trade_stream.py
HEARTBEAT_TIMEOUT = 30  # 已修复
BACKOFF_MAX = 15        # 已修复
QUEUE_SIZE = 2048       # 待修复
PRINT_EVERY = 100       # 待修复
```

#### 实施步骤
1. **扩展system.yaml** - 添加 `trade_stream` 配置节
   ```yaml
   trade_stream:
     queue_size: 2048
     print_every: 100
     heartbeat_timeout: 30
     backoff_max: 15
     ping_interval: 20
     close_timeout: 10
   ```

2. **修改binance_trade_stream.py** - 从环境变量或配置文件读取
3. **创建测试脚本**
4. **更新文档**

**预计时间**: 1-2小时  
**优先级**: 🟢 低 (已有环境变量支持)

---

### 阶段4：融合指标收集器配置集成 (优先级：🟡 中)

#### 目标
将融合指标收集器的参数迁移到统一配置

#### 硬编码参数清单
```python
# fusion_metrics.py - FusionMetricsCollector
# 指标收集间隔
# 统计窗口大小
# 更新频率
```

#### 实施步骤
1. **检查现有配置** - `system.yaml` 中的 `fusion_metrics` 部分
2. **修改FusionMetricsCollector类** - 支持从配置加载
3. **创建测试脚本**
4. **更新文档**

**预计时间**: 1-2小时  
**优先级**: 🟡 中 (已有部分配置)

---

## 📅 实施时间表

### Week 1: 高优先级组件
- Day 1-2: 背离检测核心配置集成
- Day 3: 策略模式管理器配置集成（第1部分）

### Week 2: 中低优先级组件
- Day 1: 策略模式管理器配置集成（第2部分）
- Day 2: 融合指标收集器配置集成
- Day 3: 交易流处理配置集成

### Week 3: 测试和文档
- Day 1-2: 全面测试和集成验证
- Day 3: 文档更新和最终审查

---

## 🎯 验收标准

### 技术验收
- [ ] 所有硬编码参数迁移到 `system.yaml`
- [ ] 每个组件支持环境变量覆盖
- [ ] 配置热更新功能正常
- [ ] 所有测试用例通过
- [ ] 无端口冲突
- [ ] 性能无明显下降

### 文档验收
- [ ] 每个组件的配置使用示例
- [ ] 环境变量覆盖说明
- [ ] 故障排查指南
- [ ] 迁移指南（从硬编码到配置）

### 用户验收
- [ ] 配置文件易于理解和修改
- [ ] 环境切换简单（development/testing/production）
- [ ] 调试和日志清晰
- [ ] 向后兼容性保持

---

## 🔧 技术实施指南

### 1. 配置加载器模板
```python
class ComponentConfigLoader:
    """组件配置加载器模板"""
    
    def __init__(self, config_loader):
        self.config_loader = config_loader
    
    def load_config(self) -> ComponentConfig:
        try:
            config_raw = self.config_loader.get('component_name', {})
            
            # 提取参数
            param1 = config_raw.get('param1', default_value)
            param2 = config_raw.get('param2', default_value)
            
            return ComponentConfig(
                param1=param1,
                param2=param2
            )
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return ComponentConfig()
```

### 2. 组件类修改模板
```python
class Component:
    def __init__(self, cfg: ComponentConfig = None, config_loader=None):
        if config_loader:
            self.cfg = self._load_from_config_loader(config_loader)
        else:
            self.cfg = cfg or ComponentConfig()
    
    def _load_from_config_loader(self, config_loader) -> ComponentConfig:
        loader = ComponentConfigLoader(config_loader)
        return loader.load_config()
```

### 3. 测试脚本模板
```python
def test_config_loading():
    """测试配置加载"""
    config_loader = ConfigLoader()
    component = Component(config_loader=config_loader)
    
    assert component.cfg is not None
    assert component.cfg.param1 == expected_value
    print("✅ 配置加载测试通过")

def test_env_override():
    """测试环境变量覆盖"""
    os.environ['V13__COMPONENT__PARAM1'] = 'override_value'
    
    config_loader = ConfigLoader()
    component = Component(config_loader=config_loader)
    
    assert component.cfg.param1 == 'override_value'
    print("✅ 环境变量覆盖测试通过")
```

---

## 📝 配置文件组织

### system.yaml 结构建议
```yaml
# ============================================================================
# 系统配置
# ============================================================================
system:
  version: "v13.0"
  environment: "development"

# ============================================================================
# 核心组件配置
# ============================================================================
components:
  ofi: { ... }
  cvd: { ... }
  websocket: { ... }
  fusion_metrics: { ... }
  divergence_detection: { ... }      # 新增
  strategy_mode: { ... }             # 新增
  trade_stream: { ... }              # 新增

# ============================================================================
# 监控配置
# ============================================================================
monitoring:
  prometheus: { port: 8003 }
  divergence_metrics: { port: 8004 }
  fusion_metrics: { port: 8005 }
  strategy_metrics: { port: 8006 }
  grafana: { ... }

# ============================================================================
# 性能配置
# ============================================================================
performance: { ... }

# ============================================================================
# 日志配置
# ============================================================================
logging: { ... }
```

---

## 🚀 快速开始

### 立即开始阶段1
```bash
# 1. 创建配置节
vim config/system.yaml  # 添加 divergence_detection 配置

# 2. 创建配置加载器
python src/divergence_config_loader.py

# 3. 修改核心类
vim src/ofi_cvd_divergence.py  # 添加 config_loader 支持

# 4. 测试
python test_divergence_config.py
```

---

**计划创建时间**: 2025-10-20 06:45  
**计划版本**: v1.0  
**下一次更新**: 完成阶段1后

