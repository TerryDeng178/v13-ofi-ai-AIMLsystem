# Task 1.2.14: 集成CVD模块参数热更新

## 📋 任务信息

- **任务编号**: Task_1.2.14
- **任务名称**: 集成CVD模块参数热更新（基于Task 0.7动态模式切换）
- **所属阶段**: 阶段1 - 真实OFI+CVD核心
- **优先级**: 高
- **预计时间**: 2-3小时
- **实际时间**: ___（完成后填写）___
- **任务状态**: ⏳ 待开始
- **前置任务**: 
  - ✅ Task_0.7（动态模式切换与差异化配置）- 框架已完成
  - ✅ Task_1.2.10（CVD计算测试）- CVD模块已稳定

---

## 🎯 任务目标

将 Task 0.7 完成的动态模式切换框架与实际的 CVD 模块集成，实现 CVD 参数的热更新，使系统能够根据市场活跃度自动调整 CVD 计算参数。

### 核心目标

1. **实现CVD参数热更新接口** - 在 `RealCVDCalculator` 中添加 `update_params()` 方法
2. **集成到策略模式管理器** - 在 `StrategyModeManager.apply_params()` 中调用CVD模块
3. **验证参数生效** - 确保参数切换后 CVD 计算使用新参数
4. **测试热更新性能** - 验证更新耗时 ≤ 100ms（P99）
5. **验证数据连续性** - 确保参数切换不影响 CVD 序列连续性

---

## 📝 任务清单

### 阶段1: CVD模块增强（1小时）

- [ ] 1.1 在 `RealCVDCalculator` 类中添加 `update_params()` 方法
  - [ ] 支持热更新：`window_ticks`, `ema_span`, `denoise_sigma`
  - [ ] 使用线程锁保护关键状态
  - [ ] 返回更新成功/失败状态
  - [ ] 记录参数变化日志

- [ ] 1.2 添加参数验证逻辑
  - [ ] 验证参数合法性（范围检查）
  - [ ] 验证参数组合合理性
  - [ ] 失败时保持原参数不变

- [ ] 1.3 添加状态保存与恢复
  - [ ] 保存当前计算状态（EMA、MAD等）
  - [ ] 参数切换时智能调整状态
  - [ ] 避免状态突变导致的异常值

### 阶段2: 集成到策略模式管理器（30分钟）

- [ ] 2.1 修改 `StrategyModeManager.apply_params()`
  - [ ] 添加 CVD 模块实例引用
  - [ ] 实现 `apply_to_cvd()` 方法
  - [ ] 处理更新成功/失败场景
  - [ ] 记录更新耗时到 Prometheus

- [ ] 2.2 实现回滚机制
  - [ ] CVD更新失败时回滚到旧参数
  - [ ] 记录失败模块为 'cvd'
  - [ ] 触发告警

### 阶段3: 测试与验证（30分钟-1小时）

- [ ] 3.1 单元测试
  - [ ] 测试 `update_params()` 基本功能
  - [ ] 测试参数验证逻辑
  - [ ] 测试线程安全性
  - [ ] 测试失败回滚

- [ ] 3.2 集成测试
  - [ ] 在真实数据流中触发参数切换
  - [ ] 验证CVD序列连续性
  - [ ] 验证新参数生效
  - [ ] 验证更新耗时 ≤ 100ms

- [ ] 3.3 性能测试
  - [ ] 测量P50/P95/P99更新耗时
  - [ ] 确保无数据丢失
  - [ ] 确保无计算异常

---

## 📦 Allowed Files

### 允许修改的文件
- `src/real_cvd_calculator.py` - 添加 `update_params()` 方法
- `src/utils/strategy_mode_manager.py` - 实现 CVD 集成
- `examples/run_realtime_cvd.py` - 传递 CVD 实例给管理器

### 允许创建的文件
- `tests/test_cvd_hot_update.py` - CVD 热更新单元测试
- `examples/test_cvd_mode_switching.py` - 端到端测试脚本

### 不允许修改的文件
- `config/system.yaml` - 配置结构已固定
- 其他核心算法文件

---

## 📚 依赖项

### 前置任务
- ✅ Task_0.7（动态模式切换框架）
- ✅ Task_1.2.10（CVD计算稳定）

### 技术依赖
- Python threading 模块
- 已有的 PrometheusMetrics
- 已有的 StrategyModeManager

---

## ✅ 验证标准

### 功能验证
- [ ] **V1**: `update_params()` 方法能成功更新 CVD 参数
- [ ] **V2**: 参数验证能正确拒绝非法参数
- [ ] **V3**: 线程安全 - 并发调用不导致状态错乱
- [ ] **V4**: 失败回滚 - 更新失败时保持原参数

### 性能验证
- [ ] **V5**: 参数更新耗时 P99 ≤ 100ms
- [ ] **V6**: 更新期间无数据丢失
- [ ] **V7**: 更新期间无计算异常（NaN/Inf）

### 连续性验证
- [ ] **V8**: CVD 序列在参数切换前后保持连续
- [ ] **V9**: Z-score 序列无异常跳变（除预期的尺度变化）

### 集成验证
- [ ] **V10**: 策略模式管理器能正确调用 CVD 更新
- [ ] **V11**: Prometheus 指标正确记录更新耗时
- [ ] **V12**: 更新失败时触发 `strategy_params_update_failures_total` 计数

**通过标准**: 12/12 验证全部通过

---

## 🧪 测试结果

### 单元测试

**执行时间**: ___  
**测试环境**: 开发环境

| 测试用例 | 状态 | 说明 |
|---------|------|------|
| test_update_params_success | ___ | 基本功能测试 |
| test_update_params_validation | ___ | 参数验证测试 |
| test_update_params_thread_safety | ___ | 线程安全测试 |
| test_update_params_rollback | ___ | 失败回滚测试 |

### 集成测试

**执行时间**: ___  
**测试环境**: 真实币安数据流

| 测试场景 | 状态 | 说明 |
|---------|------|------|
| active → quiet 切换 | ___ | 验证参数从活跃切到安静模式 |
| quiet → active 切换 | ___ | 验证参数从安静切到活跃模式 |
| CVD 序列连续性 | ___ | 切换前后CVD值连续 |
| Z-score 合理性 | ___ | 切换后Z-score正常 |

### 性能测试

**执行时间**: ___  
**样本数量**: 100次切换

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 更新耗时 P50 | ≤ 50ms | ___ ms | ___ |
| 更新耗时 P95 | ≤ 80ms | ___ ms | ___ |
| 更新耗时 P99 | ≤ 100ms | ___ ms | ___ |
| 数据丢失率 | 0% | ___ % | ___ |
| 计算异常率 | 0% | ___ % | ___ |

---

## 📊 DoD检查清单

### 代码质量
- [ ] 代码通过 pylint/flake8 检查
- [ ] 代码有完整的 docstring
- [ ] 关键逻辑有行内注释
- [ ] 无 TODO/FIXME 标记

### 测试覆盖
- [ ] 单元测试覆盖率 ≥ 80%
- [ ] 集成测试通过
- [ ] 性能测试通过
- [ ] 边界条件测试通过

### 文档完善
- [ ] 代码 docstring 完整
- [ ] 更新 `CVD_SYSTEM_FILES_GUIDE.md`
- [ ] 添加参数热更新使用示例
- [ ] 更新 CHANGELOG

### 集成验证
- [ ] 与策略模式管理器集成成功
- [ ] Prometheus 指标正常记录
- [ ] 日志输出符合预期
- [ ] 无遗留问题

---

## 📝 执行记录

### 开始时间
___（开始时填写）___

### 完成时间
___（完成时填写）___

### 实际耗时
___（完成时填写）___

### 遇到的问题
___（记录遇到的问题）___

### 解决方案
___（记录解决方案）___

### 经验教训
___（记录经验教训）___

---

## 🔗 相关链接

### 前置任务
- [Task_0.7_动态模式切换与差异化配置](../Stage0_准备工作/✅Task_0.7_动态模式切换与差异化配置.md)
- [Task_1.2.10_CVD计算测试](./✅Task_1.2.10_CVD计算测试.md)

### 后续任务
- [Task_1.2.15_集成OFI/Risk/Performance模块](./Task_1.2.15_集成OFI与Risk模块参数热更新.md)
- [Task_1.2.16_真实环境24小时测试](./Task_1.2.16_真实环境24小时测试.md)

### 参考文档
- [Task_0.7_FINAL_COMPLETION_REPORT.md](../Stage0_准备工作/Task_0.7_FINAL_COMPLETION_REPORT.md)
- [SYSTEM_CONFIG_GUIDE.md](../../docs/SYSTEM_CONFIG_GUIDE.md)

---

## ⚠️ 注意事项

### 技术注意事项

1. **线程安全**
   - CVD 计算通常在主线程
   - 参数更新可能来自监控线程
   - 必须使用锁保护共享状态

2. **状态一致性**
   - EMA 状态需要智能调整（不能简单重置）
   - MAD 窗口需要保留或插值
   - 避免参数切换导致的异常值

3. **性能敏感**
   - CVD 是高频计算（每笔交易）
   - 参数更新必须快速（≤ 100ms）
   - 避免在更新期间阻塞主计算

4. **回滚策略**
   - 保存旧参数副本
   - 更新失败立即回滚
   - 记录详细失败原因

### 业务注意事项

1. **数据连续性优先**
   - 参数切换不应导致CVD序列断裂
   - Z-score 可以有尺度变化，但不应有突变

2. **监控充分**
   - 记录每次参数切换
   - 记录切换前后的CVD/Z-score样本
   - 便于后续分析和优化

3. **渐进式集成**
   - 先完成CVD，再集成OFI
   - 每个模块独立验证
   - 避免一次性集成多个模块

---

## 💡 实现建议

### CVD 模块增强示例

```python
class RealCVDCalculator:
    def __init__(self, config):
        self.config = config
        self.params_lock = threading.RLock()
        # ... 现有代码 ...
    
    def update_params(self, new_params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        热更新CVD参数
        
        Args:
            new_params: 新参数字典，例如：
                {
                    'window_ticks': 2000,
                    'ema_span': 60,
                    'denoise_sigma': 2.5
                }
        
        Returns:
            (是否成功, 错误信息)
        """
        with self.params_lock:
            try:
                # 1. 验证参数
                if not self._validate_params(new_params):
                    return False, "Invalid parameters"
                
                # 2. 保存旧参数（用于回滚）
                old_params = self._get_current_params()
                
                # 3. 应用新参数
                self._apply_params(new_params)
                
                # 4. 调整内部状态（如需要）
                self._adjust_state(old_params, new_params)
                
                logger.info(f"CVD params updated: {old_params} -> {new_params}")
                return True, ""
                
            except Exception as e:
                logger.error(f"Failed to update CVD params: {e}")
                return False, str(e)
    
    def _validate_params(self, params: Dict[str, Any]) -> bool:
        """验证参数合法性"""
        if 'window_ticks' in params:
            if not (100 <= params['window_ticks'] <= 20000):
                return False
        
        if 'ema_span' in params:
            if not (10 <= params['ema_span'] <= 500):
                return False
        
        if 'denoise_sigma' in params:
            if not (0.5 <= params['denoise_sigma'] <= 5.0):
                return False
        
        return True
```

### 策略模式管理器集成示例

```python
class StrategyModeManager:
    def __init__(self, config):
        # ... 现有代码 ...
        self.cvd_instance = None  # 将在外部设置
    
    def set_cvd_instance(self, cvd):
        """设置CVD实例引用"""
        self.cvd_instance = cvd
    
    def apply_params(self, mode: StrategyMode) -> Tuple[bool, List[str]]:
        """应用参数（含CVD集成）"""
        # ... 现有代码 ...
        
        # 应用CVD参数
        if self.cvd_instance and 'cvd' in new_params:
            success, error = self.cvd_instance.update_params(new_params['cvd'])
            if not success:
                logger.error(f"CVD update failed: {error}")
                _metrics.inc_counter('strategy_params_update_failures_total', {'module': 'cvd'})
                failed_modules.append('cvd')
        
        # ... 现有代码 ...
```

---

**任务创建时间**: 2025-10-19 06:20  
**任务创建者**: AI Assistant  
**任务版本**: V1.0  
**任务状态**: ⏳ 待开始

