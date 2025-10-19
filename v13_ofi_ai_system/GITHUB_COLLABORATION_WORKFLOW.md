# GitHub AI协作工作流指南

## 🎯 协作目标

本指南旨在帮助AI助手和Cursor在GitHub上高效协作开发V13 OFI+CVD交易策略系统。

## 📋 协作前准备

### 1. 环境设置
```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/v13-ofi-ai-system.git
cd v13-ofi-ai-system

# 安装依赖
pip install -r requirements.txt

# 配置环境
copy env.example .env
# 编辑.env文件设置密码
```

### 2. 验证环境
```bash
# 检查系统状态
python verify_monitoring.py

# 运行测试
python -m unittest tests.test_strategy_mode_manager -v

# 检查配置
python -m py_compile src/utils/config_loader.py
```

## 🔄 标准协作流程

### 阶段1：任务选择与规划

1. **查看任务状态**
   - 阅读 `TASKS/README.md` 了解项目全貌
   - 查看 `TASKS/Stage*/` 目录中的任务卡片
   - 选择状态为 `⏳待开始` 或 `🔄进行中` 的任务

2. **理解任务要求**
   - 仔细阅读任务卡片内容
   - 查看相关依赖任务
   - 理解技术要求和验收标准

3. **创建开发分支**
   ```bash
   git checkout -b feature/task-xxx-description
   ```

### 阶段2：开发实施

1. **代码开发**
   - 遵循 `.cursorrules` 中的开发规范
   - 使用配置系统管理参数
   - 添加必要的监控指标
   - 编写单元测试

2. **配置更新**
   - 修改 `config/system.yaml` 如需新增配置
   - 更新环境特定配置文件
   - 确保环境变量覆盖支持

3. **监控集成**
   - 添加Prometheus指标
   - 更新Grafana仪表盘（如需要）
   - 添加告警规则（如需要）

### 阶段3：测试验证

1. **单元测试**
   ```bash
   python -m unittest tests.test_* -v
   ```

2. **配置验证**
   ```bash
   python -m py_compile src/utils/*.py
   ```

3. **系统验证**
   ```bash
   python verify_monitoring.py
   ```

4. **集成测试**
   - 启动监控系统：`start_full_monitoring.bat`
   - 验证Grafana仪表盘数据
   - 测试告警功能

### 阶段4：文档更新

1. **任务状态更新**
   - 更新任务卡片状态
   - 添加完成说明和结果
   - 记录遇到的问题和解决方案

2. **代码文档**
   - 更新函数和类的文档字符串
   - 添加必要的注释
   - 更新README（如需要）

3. **配置文档**
   - 更新配置说明
   - 添加新参数的使用示例
   - 更新环境变量说明

### 阶段5：提交与合并

1. **代码提交**
   ```bash
   git add .
   git commit -m "feat: 完成任务描述"
   git push origin feature/task-xxx-description
   ```

2. **创建Pull Request**
   - 使用PR模板填写详细信息
   - 关联相关Issue
   - 添加测试结果截图

3. **代码审查**
   - 等待其他AI或维护者审查
   - 根据反馈进行修改
   - 确保所有检查通过

4. **合并代码**
   - 审查通过后合并到主分支
   - 删除功能分支
   - 更新任务状态为 `✅已完成`

## 🤖 AI协作最佳实践

### 1. 沟通规范

**任务分配**
- 在Issue中明确任务范围和要求
- 使用标签分类任务类型（bug/feature/enhancement）
- 设置优先级和截止时间

**进度报告**
- 定期更新任务状态
- 在Issue中记录重要进展
- 及时报告遇到的问题

**代码审查**
- 提供详细的PR描述
- 解释设计决策和实现思路
- 添加测试结果和性能数据

### 2. 代码质量

**编码规范**
- 遵循PEP 8和项目代码风格
- 使用类型提示提高代码可读性
- 保持函数简洁，单一职责

**测试覆盖**
- 核心功能必须有单元测试
- 测试用例要覆盖边界条件
- 集成测试验证端到端功能

**文档完整**
- 代码注释要清晰准确
- README和配置文档要及时更新
- 任务卡片要记录完整的开发过程

### 3. 配置管理

**配置一致性**
- 所有参数通过配置系统管理
- 环境变量命名遵循 `V13__section__key` 格式
- 配置变更要同步更新文档

**环境隔离**
- 开发、测试、生产环境配置分离
- 敏感信息使用环境变量
- 配置验证要覆盖所有环境

### 4. 监控集成

**指标设计**
- 指标命名要清晰有意义
- 标签设计要支持灵活查询
- 指标类型要符合Prometheus规范

**仪表盘更新**
- 新增指标要同步更新Grafana
- 面板配置要支持多环境
- 告警规则要覆盖关键场景

## 🚨 常见问题与解决方案

### 1. 环境问题

**问题**: 监控系统启动失败
**解决**: 
```bash
# 检查Docker状态
docker --version
docker compose --version

# 检查端口占用
netstat -ano | findstr :3000
netstat -ano | findstr :9090

# 重启服务
docker compose down
docker compose up -d
```

**问题**: Python依赖安装失败
**解决**:
```bash
# 升级pip
python -m pip install --upgrade pip

# 清理缓存
pip cache purge

# 重新安装
pip install -r requirements.txt --force-reinstall
```

### 2. 配置问题

**问题**: 配置加载失败
**解决**:
- 检查YAML语法是否正确
- 验证环境变量格式
- 查看配置加载器日志

**问题**: 环境变量不生效
**解决**:
- 确认变量名格式：`V13__section__key`
- 检查变量值类型是否匹配
- 验证配置路径是否存在

### 3. 监控问题

**问题**: Grafana显示"No Data"
**解决**:
```bash
# 检查Prometheus连接
curl http://localhost:9090/api/v1/targets

# 检查指标服务器
curl http://localhost:8000/metrics

# 重启监控服务
start_full_monitoring.bat
```

**问题**: 告警不触发
**解决**:
- 检查告警规则语法
- 验证指标名称和标签
- 查看Alertmanager配置

## 📞 协作支持

### 获取帮助
1. **查看文档**: `docs/` 目录下的详细文档
2. **任务说明**: `TASKS/` 目录下的任务卡片
3. **代码注释**: 代码中的详细注释和文档字符串
4. **Issue讨论**: 在GitHub Issue中提问和讨论

### 贡献指南
1. **Fork项目**: 创建自己的分支进行开发
2. **提交PR**: 使用标准PR模板提交代码
3. **代码审查**: 参与其他PR的审查
4. **文档改进**: 帮助完善项目文档

### 维护者职责
1. **任务管理**: 定期更新任务状态和优先级
2. **代码审查**: 及时审查和合并PR
3. **问题处理**: 快速响应和解决Issue
4. **文档维护**: 保持文档与代码同步

---

**最后更新**: 2025-10-19  
**版本**: V1.3  
**维护者**: V13开发团队
