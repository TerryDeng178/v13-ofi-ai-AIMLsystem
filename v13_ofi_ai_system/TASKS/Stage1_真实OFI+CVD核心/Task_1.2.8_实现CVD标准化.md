# Task 1.2.8: 实现CVD标准化

## 📋 任务信息

- **任务编号**: Task_1.2.8
- **任务名称**: 实现CVD标准化
- **所属阶段**: 阶段1.2 - 真实OFI+CVD计算
- **优先级**: 高
- **预计时间**: 1小时
- **实际时间**: ___（完成后填写）___
- **任务状态**: ⏳ 待开始

---

## 🎯 任务目标

实现CVD的z-score标准化和EMA平滑，使CVD值可与OFI进行比较和融合。

---

## 📝 任务清单

- [ ] 实现 `get_cvd_zscore()` 方法
- [ ] 计算CVD均值和标准差
- [ ] 滚动窗口大小: 300个数据点
- [ ] 计算z-score
- [ ] 实现EMA平滑

---

## 🔧 技术规格

### Z-score计算
```python
def get_cvd_zscore(self):
    """计算CVD Z-score"""
    if len(self.cvd_history) < 30:  # warmup期
        return 0.0
    
    cvd_array = list(self.cvd_history)
    mean = sum(cvd_array) / len(cvd_array)
    variance = sum((x - mean) ** 2 for x in cvd_array) / len(cvd_array)
    std = variance ** 0.5
    
    if std < 1e-6:
        return 0.0
    
    current_cvd = self.cvd_history[-1]
    z_score = (current_cvd - mean) / std
    
    return z_score
```

### EMA平滑
```python
# 在update_with_trade中
if self.ema_cvd is None:
    self.ema_cvd = self.cumulative_delta
else:
    alpha = self.cfg.ema_alpha
    self.ema_cvd = alpha * self.cumulative_delta + (1 - alpha) * self.ema_cvd
```

---

## ✅ 验证标准

- [ ] z-score分布接近标准正态分布
- [ ] 强信号（|Z|>2）频率 5-10%
- [ ] EMA平滑有效
- [ ] 计算稳定，无异常值

---

## 📊 测试结果

___（测试完成后填写）___

---

## 🔗 相关文件

### Allowed files
- `src/real_cvd_calculator.py` (修改)
- `tests/test_real_cvd_calculator.py` (修改)

---

## ⚠️ 注意事项

1. ✅ warmup期间返回0或None
2. ✅ 防止除零错误（std < 1e-6）
3. ✅ 滚动窗口要高效（使用deque）
4. ✅ EMA参数可调整

---

## 📋 DoD检查清单

- [ ] **代码无语法错误**
- [ ] **通过单元测试**
- [ ] **无Mock/占位/跳过**
- [ ] **产出真实验证结果**
- [ ] **更新相关文档**
- [ ] **提交Git**

---

## 📝 执行记录

### 遇到的问题
___（记录遇到的问题）___

### 解决方案
___（记录解决方案）___

---

## 📈 质量评分

- **算法准确性**: ___/10
- **性能效率**: ___/10
- **总体评分**: ___/10

---

## 🔄 任务状态更新

- **开始时间**: ___
- **完成时间**: ___
- **是否可以继续**: ⬜ 是 / ⬜ 否

---

**创建时间**: 2025-10-17  
**最后更新**: 2025-10-17

