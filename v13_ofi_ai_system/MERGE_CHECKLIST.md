# 合并检查清单 & 执行指南

## ✅ 合并前检查（已完成）

### 配置验证
- [x] `validate_config --strict` 通过
- [x] `overall_pass=true`
- [x] `unknown_count=0`
- [x] `conflicts_count=0`
- [x] `errors_count=0`
- [x] `allow_legacy_keys=false`（严格模式）

### 运行时探针
- [x] 冒烟测试：60.0s，0 错误
- [x] 热更新测试：5× reload，0 错误
- [x] `reload_latency_p99_ms ≈ 44ms` < 200ms 阈值
- [x] 无半配置状态

### 纸上交易金丝雀（CI 短版）
- [x] 运行时长：1.0 分钟（CI 短版）
- [x ) 撮合错误：0
- [x] `p99 时延：0.41ms` < 500ms
- [x] 四类信号触发率 > 0（OFI/CVD/Fusion/Divergence）
- [x] 指纹一致性：通过

### 指纹一致性
- [x] 日志指纹 = Metrics 指纹：`215e148dae86d23b`
- [x] 无漂移

### 报告完整性
- [x] `validate_config_summary.json` - PASS
- [x] `runtime_probe_report.json` - PASS（60s + 5×）
- [x] `paper_canary_report.json` - PASS
- [x] `fingerprint_consistency.json` - PASS
- [x] `RELEASE_NOTES.md` - 已生成且准确

## 🚀 立即执行：合并操作

### 1. 创建 PR
```bash
# 切换到功能分支
git checkout <feature_branch>

# 添加所有变更
git add .

# 提交（使用规范格式）
git commit -m "Config 收口 & 防回归合并：Fail Gate/指纹/热更抗抖/旧键清理"

# 推送分支
git push origin <feature_branch>
```

### 2. PR 描述模板
```markdown
## 合并内容

本次合并包含以下改进：
- Fail Gate：冲突键检测机制
- 指纹一致性：日志与 metrics 双重验证
- 热更新抗抖：连续 reload 稳定性验证
- 旧键清理：配置路径统一管理

## 验证结果

### 配置验证
- ✅ strict 模式通过
- ✅ 无未知键、无冲突、无错误

### 运行时探针
- ✅ 60s 冒烟测试：0 错误
- ✅ 5× 热更新：p99≈44ms（远低于 200ms 阈值）

### 纸上交易金丝雀（CI 短版）
- ✅ 1 分钟测试：撮合错误=0，p99=0.41ms，四类信号均有触发

团队指纹一致性
- ✅ 日志指纹 = Metrics 指纹：215e148dae86d23b

## 附件

详细报告请查看 PR 附件：
- `reports/validate_config_summary.json`
- `reports/runtime_probe_report.json`
- `reports/paper_canary_report.json`
- `reports/fingerprint_consistency.json`
- `reports/RELEASE_NOTES.md`
```

### 3. Squash Merge
- [ ] 选择 "Squash and merge"
- [ ] 目标分支：`main` 或 `release`
- [ ] 合并消息：使用上述 PR 描述模板

### 4. 打标签
```bash
# 切换到主分支
git checkout main
git pull origin main

# 打标签（例如：v13.2.0）
git tag -a v13.2.0 -m "Config 收口 & 防回归：Fail Gate/指纹/热更抗抖"
git push origin v13.2.0
```

## 🧪 合并后：主线上跑 60 分钟金丝雀

### Windows
```powershell
cd v13_ofi_ai_system
.\tools\paper_canary_production.ps1
```

### Linux/Mac
```bash
cd v13_ofi_ai_system
chmod +x tools/paper_canary_production.sh
./tools/paper_canary_production.sh
```

### 判定标准
- [ ] `error_rate=0`
- [ ] `latency.p99 < 500ms`
- [ ] 四类信号在活跃时段均 > 0
- [ ] 指纹无漂移（初始指纹 = 最终指纹）

## 📊 监控护栏（合并后一周重点盯）

### 每日检查项

#### 1. 指纹一致性
- [ ] 日志指纹 = Metrics 指纹
- [ ] 无漂移
- [ ] **异常即回滚气压阀
```

- [ ] 双层数据校验（日志 vs 指标）

### 6. 热更新性能监控
- [ ] p99 < 200ms（当前 ~44ms，有充足余量）
- [ ] 持续监控，关注是否随负载上升
- [ ] 若 > 200ms → 考虑节流/批量合并 reload

#### 2. 纸上撮合错误率
- [ ] 维持 0
- [ ] 若上升 → 立即定位匹配引擎

#### 3. 信号产出率
- [ ] OFI 触发率 > 0（活跃时段）
- [ ] CVD 触发率 > 0（活跃时段）
- [ ] Fusion 触发率 > 0（活跃时段）
- [ ] Divergence 触发率 > 0（活跃时段）

#### 4. 配置校验
- [ ] 生产环境 `allow_legacy_keys=false` 或未设置
- [ ] 将校验结果写入发布页

#### 5. 热更新分位
- [ ] 关注 p95/p99 是否随负载上升
- [ ] 建议阈值：p99 < 200ms（当前 ~44ms，有充足余量）
- [ ] 若 > 200ms → 考虑节流/批量合并 reload

## 🔧 CI 配置建议

### 当前配置
- ✅ smoke=60s
- ✅ stress_reload=5
- ✅ `reload_latency_p99_ms < 200ms` 断言（已添加到 CI）

### 金丝雀时长策略
- **CI 环境**：1 分钟（快速反馈）
- **主线/生产前**：60 分钟（完整验证）
- **夜间任务**：10 分钟长版抽检（可选）

## 📝 回滚预案

如果合并后发现问题：

1. **立即回滚到标签版本**
```bash
git checkout v13.2.0  # 或上一个稳定版本
git push origin main --force  # 谨慎操作，需团队确认
```

2. **记录问题**
- 问题描述
- 触发条件
- 影响范围
- 回滚时间

3. **修复后重新验证**
- 重新运行所有门禁检查
- 确保问题已解决
- 创建修复 PR

## ✨ 成功标准

本次合并成功的标志：
- ✅ 所有门禁检查通过
- ✅ 60 分钟生产金丝雀通过
- ✅ 合并后一周内无配置相关故障
- ✅ 指纹一致性保持稳定
- ✅ 热更新性能保持在阈值内

---

**生成时间**: 2025-10-30
**状态**: ✅ 准备就绪，可以合并

