# Task 1.2.6: 创建CVD计算器基础类

## 📋 任务信息

- **任务编号**: Task_1.2.6
- **任务名称**: 创建CVD计算器基础类
- **所属阶段**: 阶段1.2 - 真实OFI+CVD计算
- **优先级**: 高
- **预计时间**: 30分钟
- **实际时间**: ___（完成后填写）___
- **任务状态**: ⏳ 待开始

---

## 🎯 任务目标

创建独立的CVD（Cumulative Volume Delta）计算器类，处理成交数据并计算累积成交量差。

---

## 📝 任务清单

- [ ] 创建文件 `v13_ofi_ai_system/src/real_cvd_calculator.py`
- [ ] 实现 `RealCVDCalculator` 类基础结构
- [ ] 定义CVD配置参数（窗口大小、重置周期等）
- [ ] 初始化历史CVD缓存

---

## 🔧 技术规格

### CVD配置类
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CVDConfig:
    reset_period: Optional[int] = None  # CVD重置周期（秒），None=不重置
    z_window: int = 300                 # z-score滚动窗口
    ema_alpha: float = 0.2              # EMA平滑系数
```

### RealCVDCalculator类结构
```python
class RealCVDCalculator:
    def __init__(self, symbol: str, cfg: CVDConfig = None):
        self.symbol = symbol
        self.cfg = cfg or CVDConfig()
        self.cumulative_delta = 0.0
        self.cvd_history = deque(maxlen=self.cfg.z_window)
        self.ema_cvd = None
        ...
```

---

## ✅ 验证标准

- [ ] 文件创建成功
- [ ] 类结构正确
- [ ] 参数定义合理
- [ ] 通过 `python -m py_compile v13_ofi_ai_system/src/real_cvd_calculator.py`
- [ ] 无第三方依赖（只用标准库）

---

## 📊 测试结果

___（测试完成后填写）___

---

## 🔗 相关文件

### Allowed files
- `src/real_cvd_calculator.py` (新建)

### 依赖
- 无（只使用Python标准库）

---

## 📚 参考资料

- CVD（累积成交量差）定义：`CVD = Σ(买方主动成交量 - 卖方主动成交量)`
- 数据来源：Binance WebSocket成交流（`@aggTrade`）

---

## ⚠️ 注意事项

1. ✅ 只使用Python标准库（collections, dataclass, typing等）
2. ✅ 不引入numpy/pandas等第三方库
3. ✅ 类设计要清晰，职责单一
4. ✅ 配置参数要合理，易于调整

---

## 📋 DoD检查清单

- [ ] **代码无语法错误** - 能正常运行
- [ ] **通过py_compile检查** - 代码风格符合规范
- [ ] **无Mock/占位/跳过** - 所有功能真实实现
- [ ] **产出真实验证结果** - 有真实数据、日志、截图
- [ ] **更新相关文档** - 任务文件、README等
- [ ] **提交Git** - 代码已提交，提交信息清晰

---

## 📝 执行记录

### 遇到的问题
___（记录遇到的问题）___

### 解决方案
___（记录解决方案）___

### 经验教训
___（记录经验教训）___

---

## 📈 质量评分

- **代码质量**: ___/10
- **文档完整性**: ___/10
- **测试覆盖率**: ___/10
- **总体评分**: ___/10

---

## 🔄 任务状态更新

- **开始时间**: ___
- **完成时间**: ___
- **实际耗时**: ___
- **是否可以继续下一个任务**: ⬜ 是 / ⬜ 否

---

**创建时间**: 2025-10-17  
**最后更新**: 2025-10-17  
**创建人**: AI Assistant  
**审核人**: ___

