# PR 描述模板：Config 收口 & 防回归合并

## 📋 合并内容

本次合并包含以下改进：
- ✅ Fail Gate：冲突键检测机制
- ✅ 指纹一致性：日志与 metrics 双重验证
- ✅ 热更新抗抖：连续 reload 稳定性验证
- ✅ 旧键清理：配置路径统一管理

## ✅ 验证结果

### 配置验证
- ✅ strict 模式通过
- ✅ `overall_pass=true`
- ✅ `unknown_count=0`
- ✅ `conflicts_count=0`
- ✅ `errors_count=0`
- ✅ `allow_legacy_keys=false`（严格模式）

### 运行时探针
- ✅ 冒烟测试：60.0s，12 次检查，0 错误
- ✅ 热更新测试：5× reload，0 错误，0 半配置状态
- ✅ `reload_latency_p99_ms ≈ 44ms`（远低于 200ms 阈值）

### 纸上交易金丝雀（CI 短版）
- ✅ 运行时长：1.0 分钟（CI 短版，生产建议 60 分钟）
- ✅ 撮合错误：0
- ✅ `p99 时延：0.41ms` < 500ms
- ✅ 四类信号触发率 > 0：
  - OFI: 0.0417/s
  - CVD: 0.0208/s
  - Fusion: 0.0625/s
  - Divergence: 0.0833/s

### 指纹一致性
- ✅ 日志指纹 = Metrics 指纹：`215e148dae86d23b`
- ✅ 无漂移

## 📎 附件

详细报告请查看 PR 附件或 `reports/` 目录：
- `reports/validate_config_summary.json`
- `reports/runtime_probe_report.json`
- `reports/paper_canary_report.json`
- `reports/fingerprint_consistency.json`
- `reports/RELEASE_NOTES.md`

## 🔍 合并后验证计划

1. **主线上跑 60 分钟金丝雀**（影子模式，不下单）
   - 目标：`error_rate=0`、`latency.p99<500ms`、四类信号在活跃时段均>0、指纹无漂移
   - 执行：`tools/paper_canary_production.ps1` (Windows) 或 `tools/paper_canary_production.sh` (Linux/Mac)

2. **监控护栏**（合并后一周重点盯）
   - 指纹一致性（日志=指标）
   - 旧键/冲突计数=0
   - 热更新 p99 < 200ms（当前 ~44ms，有充足余量）

## 🔄 合并方式

- [ ] Squash and merge
- [ ] 目标分支：`main` 或 `release`
- [ ] 合并后打标签（例如：`v13.2.0`）

