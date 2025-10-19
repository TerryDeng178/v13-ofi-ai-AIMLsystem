# V13 OFI+CVD 高频交易策略系统

[![CI](https://github.com/your-username/v13-ofi-cvd-framework/workflows/CI/badge.svg)](https://github.com/your-username/v13-ofi-cvd-framework/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-compose-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个专注于**OFI（Order Flow Imbalance）**和**CVD（Cumulative Volume Delta）**信号分析的高频交易策略系统，支持动态模式切换和完整的监控体系。

## ✨ 核心特性

### 🎯 交易信号分析
- **OFI计算**: 订单流不平衡分析，捕捉市场微观结构变化
- **CVD计算**: 累积成交量差值计算，识别买卖压力
- **Z-score标准化**: 统计信号标准化，提高信号质量
- **动态模式切换**: 根据市场条件自动调整策略参数

### 📊 完整监控系统
- **Prometheus**: 指标收集和存储
- **Grafana**: 3个专业仪表盘（Overview/Performance/Alerts）
- **Alertmanager**: 智能告警通知系统
- **Loki + Promtail**: 日志聚合和分析

### ⚙️ 智能配置管理
- **统一配置系统**: `system.yaml` + 环境特定配置
- **动态参数热更新**: 运行时参数调整，无需重启
- **多环境支持**: development/testing/production
- **环境变量覆盖**: 支持`V13__section__key=value`格式

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone https://github.com/your-username/v13-ofi-cvd-framework.git
cd v13-ofi-cvd-framework

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
copy env.example .env
# 编辑.env文件设置密码
```

### 2. 启动完整监控系统
```bash
# 一键启动（推荐）
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
│   ├── dashboards/          # 3个专业仪表盘
│   ├── alerting_rules/      # 告警规则
│   ├── provisioning/        # 自动配置
│   └── *.yml               # 各服务配置
├── TASKS/                   # 任务管理
│   ├── Stage0_准备工作/     # 基础任务
│   └── Stage1_真实OFI+CVD核心/  # 核心开发任务
├── docs/                    # 详细文档
├── tests/                   # 测试套件
└── docker-compose.yml       # 完整监控栈
```

## 📊 监控仪表盘

### 1. Strategy Mode Overview
- 当前模式状态和切换历史
- 市场触发因子监控
- 价差和波动率分析
- 切换原因分布统计

### 2. Strategy Performance  
- 参数更新性能（P50/P95/P99）
- 更新失败统计和趋势
- 性能直方图分布
- 模块级失败分析

### 3. Strategy Alerts
- 告警状态实时监控
- 心跳健康检查
- 告警趋势分析
- 日志历史查看

## 🤖 AI协作支持

本项目专门为AI协作设计，包含：

- **AI协作指南**: `AI_COLLABORATION_GUIDE.md` - 详细的AI协作说明
- **任务管理**: `TASKS/`目录 - 结构化的任务卡片系统
- **自动化CI/CD**: GitHub Actions工作流
- **Issue/PR模板**: 标准化的协作流程
- **完整文档**: 从快速开始到深度开发

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

## 📈 性能特性

- **低延迟**: 优化的数据处理管道
- **高可用**: 容器化部署，支持水平扩展
- **实时监控**: 13个关键Prometheus指标
- **智能告警**: 基于规则的自动告警系统
- **持久化**: 数据持久化存储，容器重启不丢失

## 🔧 开发指南

### 配置系统
所有参数通过`config/system.yaml`管理，支持：
- 环境变量覆盖：`V13__section__key=value`
- 环境特定配置：`config/environments/*.yaml`
- 动态热更新：运行时参数调整

### 动态模式切换
- **模式类型**: auto/active/quiet
- **触发条件**: 时间调度 + 市场活动
- **参数热更新**: 支持运行时参数调整
- **监控指标**: 13个Prometheus指标

## 📚 文档资源

- [AI协作指南](AI_COLLABORATION_GUIDE.md) - 详细的AI协作说明
- [项目文档索引](PROJECT_CORE_DOCUMENTATION_INDEX.md) - 完整文档导航
- [任务管理](TASKS/README.md) - 任务状态和进度
- [GitHub上传指南](GITHUB_UPLOAD_GUIDE.md) - 部署到GitHub的详细步骤

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交变更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 支持

如有问题，请查看：
- [文档目录](docs/) - 详细技术文档
- [任务管理](TASKS/) - 当前开发状态
- [Issue模板](.github/ISSUE_TEMPLATE/) - 报告问题或请求功能

---

**版本**: V1.3  
**状态**: 生产就绪  
**最后更新**: 2025-10-19