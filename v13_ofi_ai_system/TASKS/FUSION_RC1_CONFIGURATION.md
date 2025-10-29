# FUSION RC1 配置冻结报告

## 版本信息
- **版本号**: v13-fusion-rc1-20251028
- **冻结日期**: 2025-10-28
- **状态**: RC (Release Candidate)
- **目标**: 24小时观察后正式发布

## 配置位置

### 1. 源代码配置（已更新）
**文件**: `src/ofi_cvd_fusion.py`  
**类**: `OFICVDFusionConfig`

### 2. 系统配置（已更新）
**文件**: `config/system.yaml`  
**段**: `fusion_metrics`

## 完整配置

### 基础信号阈值（包B基线）
```yaml
thresholds:
  fuse_buy: 0.95
  fuse_sell: -0.95
  fuse_strong_buy: 1.70
  fuse_strong_sell: -1.70
```

### 一致性阈值（包B配置）
```yaml
consistency:
  min_consistency: 0.12
  strong_min_consistency: 0.45
```

### 数据处理配置
```yaml
data_processing:
  z_clip: 4.0
  max_lag: 0.25
  warmup_samples: 10
```

### 去噪/节流参数
```yaml
denoising:
  hysteresis_exit: 0.6
  cooldown_secs: 0.30
  min_consecutive: 1
  min_duration: 1
```

### 高级机制（三大突破）
```yaml
advanced_mechanisms:
  # 机制1: 方向翻转即重臂
  rearm_on_flip: true
  flip_rearm_margin: 0.05
  
  # 机制2: 自适应冷却
  adaptive_cooldown:
    enabled: true
    k: 0.6
    min_secs: 0.12
  
  # 机制3: 微型突发合并
  burst_coalesce_ms: 120.0
```

## 预期性能指标

### 已验证指标（合成数据）
- ✅ 非中性率: 8.00% (S1), 7.99% (S2)
- ✅ 对冲识别: 0.00% (S3)
- ✅ P99延迟: 0.010ms
- ✅ 冷却命中率: ~92% (优化前 ~97%)

### 生产环境目标
- 🎯 非中性率: 7-9%
- 🎯 冷却命中率: ≤92%
- 🎯 对冲场景: ≤1%
- 🎯 信号质量: Hit@5s ≥50%, IC@5s ≥0.1

## 部署计划

### 阶段1: 纸交易验证（当前）
- ✅ 配置已冻结为 RC1
- ✅ 部署到纸交易环境
- ⏳ 观察24小时
- ⏳ 验证信号质量指标

### 阶段2: 正式发布（24小时后）
- ⏳ 更新版本号至 v13-fusion-v1.0
- ⏳ 同步配置到 defaults.yaml
- ⏳ 正式上线生产环境
- ⏳ 持续监控和优化

## 配置迁移指南

### 从旧配置迁移到 RC1

#### 关键变更
1. **阈值大幅降低**: fuse_buy 从 1.2 → 0.95
2. **一致性要求降低**: min_consistency 从 0.30 → 0.12
3. **冷却时间缩短**: cooldown_secs 从 1.2 → 0.30
4. **新增三大高级机制**: 缺省全部启用

#### 风险提示
- ⚠️ 信号频率将显著增加（1.65% → 8%）
- ⚠️ 初期需要密切监控信号质量
- ⚠️ 建议先在纸交易环境充分验证

## 回滚方案

如果24小时观察发现问题，可快速回滚：

### 方案1: 完全回滚
```yaml
# 恢复到旧配置
fusion_metrics:
  thresholds:
    fuse_buy: 1.2
    fuse_sell: -1.2
    fuse_strong_buy: 2.2
    fuse_strong_sell: -2.2
  consistency:
    min_consistency: 0.30
    strong_min_consistency: 0.60
  denoising:
    hysteresis_exit: 1.0
    cooldown_secs: 1.2
    min_consecutive: 2
  advanced_mechanisms:
    rearm_on_flip: false
    adaptive_cooldown:
      enabled: false
    burst_coalesce_ms: 0
```

### 方案2: 部分回滚（仅关闭高级机制）
```yaml
advanced_mechanisms:
  rearm_on_flip: false
  adaptive_cooldown:
    enabled: false
  burst_coalesce_ms: 0
```

## 变更历史

### v13-fusion rehearsals → v13-fusion-rc1-20251028

**主要变更**:
- 实施包B参数优化
- 实现三大高级机制
- 非中性率提升至8%
- 验证测试全部通过

**发布原因**:
- 达成5-9%非中性率目标
- 对冲场景保持稳定
- 性能指标优秀
- 准备进入24小时验证

## 联系方式

如遇问题，联系 V13 开发团队。

---

**版本**: v13-fusion-rc1-20251028  
**日期**: 2025-10-28  
**状态**: RC → 待验证

