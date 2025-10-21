# Task 1.2.15: 集成OFI/Risk/Performance模块参数热更新

## 📋 任务信息

- **任务编号**: Task_1.2.15
- **任务名称**: 集成OFI/Risk/Performance模块参数热更新
- **所属阶段**: 阶段1 - 真实OFI+CVD核心
- **优先级**: 中
- **预计时间**: 3-4小时
- **实际时间**: ___（完成后填写）___
- **任务状态**: ⏳ 待开始
- **前置任务**: 
  - ✅ Task_0.7（动态模式切换框架）
  - ⏳ Task_1.2.14（集成CVD模块参数热更新）

---

## 🎯 任务目标

在 CVD 模块集成完成的基础上，继续集成 OFI、Risk 和 Performance 模块的参数热更新，实现完整的动态模式切换系统。

### 核心目标

1. **OFI模块热更新** - bucket_ms, depth_levels, watermark_ms等
2. **Risk模块热更新** - position_limit, order_rate_limit等
3. **Performance模块热更新** - print_every, flush_interval等
4. **完整性验证** - 所有模块协同工作
5. **端到端测试** - 完整切换流程验证

---

## 📝 任务清单

### 阶段1: OFI模块集成（1.5小时）

- [ ] 1.1 在 OFI 计算器中添加 `update_params()` 方法
- [ ] 1.2 集成到策略模式管理器
- [ ] 1.3 单元测试
- [ ] 1.4 集成测试

### 阶段2: Risk模块集成（1小时）

- [ ] 2.1 创建 Risk 参数管理器
- [ ] 2.2 集成到策略模式管理器
- [ ] 2.3 单元测试

### 阶段3: Performance模块集成（30分钟）

- [ ] 3.1 实现 Performance 参数热更新
- [ ] 3.2 集成到策略模式管理器
- [ ] 3.3 验证测试

### 阶段4: 完整性测试（1小时）

- [ ] 4.1 所有模块协同测试
- [ ] 4.2 端到端切换测试
- [ ] 4.3 性能测试
- [ ] 4.4 回滚测试

---

## ✅ 验证标准

- [ ] **V1**: OFI参数热更新成功（3个参数）
- [ ] **V2**: Risk参数热更新成功（2个参数）
- [ ] **V3**: Performance参数热更新成功（2个参数）
- [ ] **V4**: 所有模块参数同步切换
- [ ] **V5**: 更新耗时P99 ≤ 100ms
- [ ] **V6**: 单模块失败不影响其他模块
- [ ] **V7**: 失败回滚机制正常工作

**通过标准**: 7/7 验证全部通过

---

## 📦 Allowed Files

### 允许修改的文件
- `src/real_ofi_calculator.py` - 添加 `update_params()`
- `src/utils/risk_manager.py` - 创建或修改
- `src/utils/performance_manager.py` - 创建或修改
- `src/utils/strategy_mode_manager.py` - 完善集成

### 允许创建的文件
- `tests/test_ofi_hot_update.py`
- `tests/test_risk_hot_update.py`
- `tests/test_full_integration.py`

---

---

## 🔗 相关链接

### 前置任务
- [Task_0.7_动态模式切换与差异化配置](../Stage0_准备工作/✅Task_0.7_动态模式切换与差异化配置.md)
- [Task_1.2.14_集成CVD模块参数热更新](./Task_1.2.14_集成CVD模块参数热更新.md)

### 后续任务
- [Task_1.2.16_真实环境24小时测试](./Task_1.2.16_真实环境24小时测试.md)

---

**任务创建时间**: 2025-10-19 06:30  
**任务状态**: ⏳ 待开始

_详细内容参照 Task_1.2.14 的格式，此处简化_

