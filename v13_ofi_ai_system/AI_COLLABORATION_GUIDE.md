# V13 OFI+CVD AI协作指南

## 🤖 项目概述

这是一个**高频交易策略系统**，专注于OFI（Order Flow Imbalance）和CVD（Cumulative Volume Delta）信号分析，支持动态模式切换和完整的监控体系。

## 🎯 核心功能

### 1. 交易信号分析
- **OFI计算**: 订单流不平衡分析
- **CVD计算**: 累积成交量差值计算
- **Z-score标准化**: 统计信号标准化
- **动态模式切换**: 根据市场条件自动调整策略参数

### 2. 完整监控系统
- **Prometheus**: 指标收集和存储
- **Grafana**: 可视化仪表盘（3个核心面板）
- **Alertmanager**: 告警通知系统
- **Loki + Promtail**: 日志聚合和分析

### 3. 配置管理
- **统一配置系统**: `system.yaml` + 环境特定配置
- **动态参数热更新**: 运行时参数调整
- **多环境支持**: development/testing/production

## 📁 项目结构

```
v13_ofi_ai_system/
├── config/                    # 配置管理
│   ├── system.yaml           # 主配置文件
│   ├── environments/         # 环境特定配置
│   └── alerting_rules_strategy.yaml
├── src/                      # 核心代码
│   ├── utils/
│   │   ├── config_loader.py  # 配置加载器
│   │   └── strategy_mode_manager.py  # 模式管理器
│   └── ...
├── grafana/                  # 监控配置
│   ├── dashboards/          # Grafana仪表盘
│   ├── alerting_rules/      # 告警规则
│   ├── provisioning/        # 自动配置
│   └── *.yml               # 各服务配置
├── TASKS/                   # 任务管理
│   ├── Stage0_准备工作/     # 基础任务
│   └── Stage1_真实OFI+CVD核心/  # 核心开发任务
├── docs/                    # 文档
├── tests/                   # 测试
└── docker-compose.yml       # 完整监控栈
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone <repository-url>
cd v13_ofi_ai_system

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
copy env.example .env
# 编辑.env文件设置密码
```

### 2. 启动监控系统
```bash
# 一键启动完整监控栈
start_full_monitoring.bat

# 或手动启动
docker compose up -d
```

### 3. 启动指标服务器
```bash
# 启动模拟指标服务器
cd grafana
python simple_metrics_server.py 8000
```

### 4. 访问服务
- **Grafana**: http://localhost:3000 (admin/从.env读取密码)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093

## 🔧 开发指南

### 配置系统
- **主配置**: `config/system.yaml` - 系统级配置
- **环境配置**: `config/environments/*.yaml` - 环境特定覆盖
- **环境变量**: 支持`V13__section__key=value`格式覆盖

### 动态模式切换
- **模式类型**: auto/active/quiet
- **触发条件**: 时间调度 + 市场活动
- **参数热更新**: 支持运行时参数调整
- **监控指标**: 13个Prometheus指标

### 任务管理
- **任务卡片**: `TASKS/`目录下的Markdown文件
- **状态标记**: ✅已完成、🔄进行中、⏳待开始
- **依赖关系**: 任务间有明确的依赖链

## 📊 监控仪表盘

### 1. Strategy Mode Overview
- 当前模式状态
- 切换历史和原因分布
- 市场触发因子监控
- 价差和波动率分析

### 2. Strategy Performance
- 参数更新性能（P50/P95/P99）
- 更新失败统计
- 性能趋势分析
- 直方图分布

### 3. Strategy Alerts
- 告警状态监控
- 心跳健康检查
- 告警趋势分析
- 日志历史查看

## 🧪 测试验证

### 单元测试
```bash
python -m unittest tests.test_strategy_mode_manager -v
```

### 系统验证
```bash
python verify_monitoring.py
```

### 配置验证
```bash
python -m py_compile src/utils/config_loader.py
python -m py_compile src/utils/strategy_mode_manager.py
```

## 🔄 AI协作工作流

### 1. 任务分配
- 查看`TASKS/`目录了解当前任务状态
- 选择未完成的任务进行开发
- 更新任务状态和进度

### 2. 代码开发
- 遵循现有代码风格和架构
- 添加适当的测试和文档
- 确保配置系统兼容性

### 3. 监控集成
- 新增功能需要添加相应的监控指标
- 更新Grafana仪表盘配置
- 添加必要的告警规则

### 4. 测试验证
- 运行单元测试确保功能正确
- 使用验证脚本检查系统状态
- 在真实环境中进行集成测试

## 📝 重要文件说明

### 核心配置文件
- `config/system.yaml`: 主配置文件，包含所有系统参数
- `src/utils/config_loader.py`: 配置加载器，支持环境变量覆盖
- `src/utils/strategy_mode_manager.py`: 动态模式切换核心逻辑

### 监控配置
- `docker-compose.yml`: 完整监控栈配置
- `grafana/dashboards/`: 3个修复版仪表盘
- `grafana/alerting_rules/`: 告警规则配置

### 任务管理
- `TASKS/README.md`: 任务总览
- `TASKS/TASK_INDEX.md`: 任务索引
- `TASKS/Stage*/`: 各阶段任务卡片

## 🚨 注意事项

### 开发规范
1. **配置优先**: 所有参数都应通过配置系统管理
2. **监控集成**: 新功能必须添加相应的监控指标
3. **测试覆盖**: 核心功能必须有单元测试
4. **文档更新**: 修改功能后更新相关文档

### 环境要求
- Python 3.11+
- Docker & Docker Compose
- Windows 10/11 (当前环境)

### 安全考虑
- 生产环境密码不要硬编码
- 使用环境变量管理敏感信息
- 定期更新依赖包版本

## 🤝 协作建议

### 对于新加入的AI
1. 先阅读`PROJECT_CORE_DOCUMENTATION_INDEX.md`了解项目全貌
2. 查看`TASKS/`目录了解当前开发状态
3. 运行`verify_monitoring.py`确保环境正常
4. 选择合适的小任务开始贡献

### 对于维护者
1. 定期更新任务状态
2. 保持文档与代码同步
3. 监控系统健康状态
4. 及时处理告警和问题

## 📞 支持

如有问题，请查看：
- `docs/`目录下的详细文档
- `TASKS/`目录下的任务说明
- 代码中的注释和文档字符串

---

**最后更新**: 2025-10-19  
**版本**: V1.3  
**状态**: 生产就绪
