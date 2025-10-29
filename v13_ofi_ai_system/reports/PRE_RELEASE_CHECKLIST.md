# 配置系统上线前检查清单

**版本**: 1.0.0+dee5fb37  
**准备时间**: 2025-01-XX

## 必做微修（上线前完成）

### ✅ 1. 修复 logging_setup 导入

**位置**: `paper_trading_simulator.py::__main__`

**修复内容**:
- 添加安全降级：`ModuleNotFoundError` 时回退到标准库 `logging`
- 支持 `LOG_LEVEL` 环境变量控制日志级别
- 添加降级日志提示

**状态**: ✅ 已完成

### 2. 主分支"未消费键=失败"演练

**命令**:
```bash
CI_BRANCH=main python tools/conf_build.py all --base-dir config
```

**预期**: 若存在未消费键则构建失败（feature分支保留为warning）

**验证脚本**: `tools/conf_doctor.py::test_unconsumed_keys_gate()`

### 3. 库式注入健康检查

**命令**:
```bash
python core/core_algo.py --dry-run-config
```

**预期**: 
- 确认只用 `components` 子树注入
- 打印来源统计
- 日志显示 `[库式注入] 使用components子树并举始化组件`

### 4. 场景快照指纹负例测试

**步骤**:
1. 备份场景文件: `cp config/scenarios/strategy_params_fusion.yaml config/scenarios/strategy_params_fusion.yaml.bak`
2. 修改1字节: `sed -i 's/mode: auto/mode: autX/' config/scenarios/strategy_params_fusion.yaml` (Linux) 或手动编辑1个字符
3. 启动Strategy dry-run: `python tools/conf_build.py strategy --base-dir config --dry-run-config`

**预期**: 拒绝启动并报指纹不一致

---

## 建议的上线步骤（灰度 1→N）

### Step A: 打标 & 冻结

**操作**:
1. 确认构建产物: `dist/config/*.runtime.1.0.0.dee5fb37.yaml`
2. 验证 `current` 软链/复制指向正确版本
3. 冻结产物目录（只读或归档）

**检查点**:
- [ ] 所有6个组件运行时第已生成
- [ ] Git SHA为8位十六进制
- [ ] `current` 链接正确

### Step B: 影子一班次（24h）

**操作**:
- 用严格运行包跑24h影子（只读包+指纹校验）
- 观察"构建来源统计"与融合权重/阈值是否与包内一致

**监控指标**:
- [ ] 配置加载成功率 100%
- [ ] 场景快照指纹校验通过
- [ ] 库式注入日志正常
- [ ] 融合权重/阈值与包内一致

### Step C: 小流量灰度

**操作**:
- 单实例打开交易回路
- 监测异常自动回滚到上一个 `current`

**回滚条件**:
- [ ] 配置加载失败
- [ ] 场景快照指纹不一致
- [ ] 库式注入异常
- [ ] 融合权重/阈值偏差

### Step D: 清理兼容

**操作**:
- 确认无回退后，关闭兼容入口（旧路径加载/旁路配置）
- 将其标记为下一个小版本删除项

**删除项**:
- [ ] `CoreAlgorithm._init_components()` 旧路径分支
- [ ] `--compat-global-config` 命令行参数
- [ ] `V13_COMPAT_GLOBAL_CONFIG` 环境变量

---

## 额外加分项（非必须）

### 1. 产物规范守护

**状态**: ✅ 已实现

- 文件名正则校验: `^[a-z_]+\.runtime\.\d+\.\d+\.\d+\.[0-9a-f]{8}\.ya?ml$`
- Git SHA 8位十六进制校验
- 命名不合规直接 fail

### 2. 跨平台一致性

**状态**: ✅ 已实现

- 路径展示统一为 POSIX 分隔符（`print_path.replace('\\', '/')`）
- 内部使用 `pathlib.Path` 自动适配

### 3. 一键体检脚本

**状态**: ✅ 已创建 `tools/conf_doctor.py`

**功能**:
- Dry-run验证
- 文件名格式验证
- Git SHA格式验证
- 运行时包结构验证
- 场景快照指纹验证
- 未消费键阻断验证

**使用**:
```bash
python tools/conf_doctor.py
```

---

## 验证命令快速参考

```bash
# 1. 完整体检
python tools/conf_doctor.py

# 2. Dry-run验证
python tools/conf_build.py all --base-dir config --dry-run-config

# 3. 主分支未消费键阻断
CI_BRANCH=main python tools/conf_build.py all --base-dir config

# 4. 库式注入健康检查
python core/core_algo.py --dry-run-config

# 5. 场景快照指纹负例测试
# (修改场景文件后)
python tools/conf_build.py strategy --base-dir config --dry-run-config
```

---

## 总体评价

**状态**: 🎉 **配置系统链路已经闭环，可以稳妥进入影子→灰度→全量的发布流程！**

**完成度**:
- ✅ P0/P1/P2 修复全部完成
- ✅ 验收测试通过
- ✅ 缩进问题修复
- ✅ logging_setup安全降级
- ✅ 一键体检脚本

**下一步**: 按照 Step A → Step B → Step C → Step D 逐步推进灰度发布。

