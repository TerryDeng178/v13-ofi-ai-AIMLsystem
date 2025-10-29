# Config 收口 & 防回归合并：Fail Gate/指纹/热更抗抖/旧键清理

**生成时间**: 2025-10-30 04:31:35

## 概述

本次合并包含以下改进：
- Fail Gate：冲突键检测机制
- 指纹一致性：日志与 metrics 双重验证
- 热更新抗抖：连续 reload 稳定性验证
- 旧键清理：配置路径统一管理

## 配置验证 (validate_config)

- **模式**: strict
- **状态**: ✅ 通过
- **冲突数**: 0
- **错误数**: 0
- **未知键数**: 0
- **状态说明**: 严格模式，无冲突无错误，配置完全通过 ✅

## 运行时探针 (runtime_probe)

- **冒烟测试**: ✅ 通过
  - 时长: 60.0s
  - 错误数: 0
- **热更新测试**: ✅ 通过
  - reload 次数: 5
  - p50 时延: 44.02ms
  - p95 时延: 44.06ms
  - p99 时延: 44.07ms

## 纸上交易金丝雀测试 (paper_canary)

- **状态**: ✅ 通过
- **运行时长**: 1.0 分钟 (CI 短版，生产建议运行 60 分钟)
- **撮合错误**: 0
- **p99 时延**: 0.41ms (✅)
- **信号触发率** (活跃时段):
  - OFI: 0.0417/s
  - CVD: 0.0208/s
  - Fusion: 0.0625/s
  - Divergence: 0.0833/s

## 指纹一致性 (fingerprint_consistency)

- **状态**: ✅ 一致
- **日志指纹**: 215e148dae86d23b
- **Metrics 指纹**: 215e148dae86d23b

## 总体状态

**整体状态**: ✅ 全部通过

## 报告文件

详细报告位于 `reports/` 目录：
- `validate_config_summary.json`
- `runtime_probe_report.json`
- `paper_canary_report.json`
- `fingerprint_consistency.json`
