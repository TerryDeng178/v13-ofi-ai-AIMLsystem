# 配置文件说明

本目录包含 V13 OFI+CVD+AI 交易系统的所有配置文件。

## 📁 目录结构

```
config/
├── system.yaml                    # 系统主配置文件
├── environments/                  # 环境特定配置
│   ├── development.yaml          # 开发环境
│   ├── testing.yaml              # 测试环境
│   └── production.yaml           # 生产环境
├── profiles/                      # 组件配置文件（现有）
│   ├── analysis.env              # CVD分析模式
│   └── realtime.env              # CVD实时模式
├── step_1_6_*.env                # Step 1.6 基线配置（现有）
└── README.md                      # 本文件
```

## 🎯 配置系统概述

### 配置层次

系统采用**三层配置架构**：

```
1. system.yaml (基础配置)
         ↓
2. environments/{ENV}.yaml (环境覆盖)
         ↓
3. 环境变量 (运行时覆盖)
```

**优先级**: 环境变量 > 环境配置 > 系统配置

### 配置类型

| 配置文件 | 用途 | 使用时机 |
|---------|------|---------|
| `system.yaml` | 系统默认配置 | 所有环境的基础 |
| `development.yaml` | 开发环境 | 本地开发、调试 |
| `testing.yaml` | 测试环境 | 集成测试、验证 |
| `production.yaml` | 生产环境 | 实盘交易 |
| `profiles/*.env` | 组件配置 | CVD/OFI 特定参数 |

## 🚀 使用方法

### 方法1: 使用新配置系统（推荐）

```python
from src.utils.config_loader import load_config, get_config

# 加载配置
config = load_config()

# 获取配置值
queue_size = config['performance']['queue']['max_size']

# 或使用便捷方法
queue_size = get_config('performance.queue.max_size', default=50000)
```

### 方法2: 使用环境变量指定环境

```bash
# Linux/Mac
export ENV=production
python examples/run_realtime_cvd.py

# Windows PowerShell
$env:ENV="production"
python examples/run_realtime_cvd.py
```

### 方法3: 继续使用现有 .env 文件（向后兼容）

```bash
# 分析模式
python examples/run_realtime_cvd.py --symbol ETHUSDT --duration 2400

# 这会自动加载 config/profiles/analysis.env
```

## ⚙️ 配置文件详解

### system.yaml

系统主配置文件，包含：

- **system**: 系统元信息
- **data_source**: 数据源配置（WebSocket等）
- **components**: 组件开关（CVD/OFI/AI/Trading）
- **paths**: 路径配置
- **performance**: 性能参数（队列、批处理、刷新）
- **logging**: 日志配置
- **monitoring**: 监控配置
- **features**: 特性开关

### environments/*.yaml

环境特定配置，覆盖 `system.yaml` 中的值：

| 环境 | 特点 | 适用场景 |
|-----|------|---------|
| **development** | 详细日志、小队列、高频刷新 | 本地开发调试 |
| **testing** | 标准日志、中队列、标准刷新 | 集成测试验证 |
| **production** | 警告日志、大队列、低频刷新 | 实盘交易运行 |

## 🔧 环境变量覆盖

所有配置都可以通过环境变量覆盖：

### 格式

```
SECTION_SUBSECTION_KEY=value
```

### 示例

```bash
# 覆盖队列大小
export PERFORMANCE_QUEUE_MAX_SIZE=100000

# 覆盖日志级别
export LOGGING_LEVEL=DEBUG

# 覆盖系统环境
export ENV=production
```

## 📋 配置参数速查

### 性能参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `performance.queue.max_size` | 50000 | 最大队列大小 |
| `performance.queue.full_behavior` | block | 队列满时行为 |
| `performance.flush.watermark_interval_ms` | 200 | Watermark刷新间隔 |
| `performance.flush.metrics_interval_ms` | 10000 | 指标刷新间隔 |
| `performance.logging.print_every_trades` | 1000 | 打印频率 |

### 日志参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `logging.level` | INFO | 日志级别 |
| `logging.file.enabled` | true | 启用文件日志 |
| `logging.file.max_size_mb` | 100 | 单个日志文件大小 |
| `logging.console.enabled` | true | 启用控制台日志 |

### 组件开关

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `components.cvd.enabled` | true | 启用CVD组件 |
| `components.ofi.enabled` | true | 启用OFI组件 |
| `components.ai.enabled` | false | 启用AI组件 |
| `components.trading.enabled` | false | 启用交易组件 |

## 🛡️ 配置验证

配置加载器会自动验证：

1. ✅ 必需配置项存在
2. ✅ 路径有效性
3. ✅ 类型正确性
4. ✅ YAML格式正确

如果验证失败，会抛出 `ValueError` 并提示具体错误。

## 🔄 配置迁移

### 从 .env 迁移到 system.yaml

**不需要立即迁移！** 系统完全向后兼容。

如果将来需要迁移：

1. 保留现有 `.env` 文件
2. 将特定参数添加到 `system.yaml` 或环境配置
3. 在脚本中添加 `--use-system-config` 参数（可选）

## 📚 更多信息

- 详细配置指南: `docs/SYSTEM_CONFIG_GUIDE.md`
- CVD配置说明: `docs/CVD_SYSTEM_FILES_GUIDE.md`
- 配置参数对比: `docs/CONFIG_PARAMETERS_GUIDE.md`

## ⚠️ 注意事项

1. **敏感信息**: API密钥、密码等应通过环境变量设置，不要写入配置文件
2. **路径**: 相对路径会自动转换为相对于项目根目录的绝对路径
3. **环境**: 默认环境为 `development`，生产环境请显式设置 `ENV=production`
4. **兼容性**: 所有现有 `.env` 文件继续有效，不影响当前功能

## 🎨 配置示例

### 示例1: 开发环境运行

```bash
# 使用开发环境配置
ENV=development python examples/run_realtime_cvd.py --symbol ETHUSDT
```

### 示例2: 覆盖特定参数

```bash
# 使用生产环境，但覆盖日志级别
ENV=production LOGGING_LEVEL=INFO python examples/run_realtime_cvd.py
```

### 示例3: 编程方式访问配置

```python
from src.utils.config_loader import load_config

# 加载配置
config = load_config()

# 访问配置
print(f"System: {config['system']['name']}")
print(f"Environment: {config['system']['environment']}")
print(f"Queue size: {config['performance']['queue']['max_size']}")
```

---

**版本**: v13.0  
**最后更新**: 2025-10-19  
**维护者**: V13 Team

