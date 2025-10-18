# 系统配置指南

**V13 OFI+CVD+AI Trading System - 统一配置系统使用指南**

---

## 📋 目录

- [系统概述](#系统概述)
- [快速开始](#快速开始)
- [配置架构](#配置架构)
- [配置文件详解](#配置文件详解)
- [使用方法](#使用方法)
- [环境变量覆盖](#环境变量覆盖)
- [配置验证](#配置验证)
- [最佳实践](#最佳实践)
- [故障排除](#故障排除)
- [迁移指南](#迁移指南)

---

## 系统概述

### 设计理念

V13 统一配置系统采用**分层、可覆盖、环境感知**的设计理念：

1. **分层**: 系统配置 → 环境配置 → 环境变量
2. **可覆盖**: 上层配置可以覆盖下层配置
3. **环境感知**: 根据运行环境自动加载对应配置
4. **向后兼容**: 完全兼容现有 `.env` 配置文件

### 核心特性

- ✅ **统一管理**: 所有配置集中在 `config/` 目录
- ✅ **环境隔离**: 开发/测试/生产环境独立配置
- ✅ **动态覆盖**: 支持环境变量运行时覆盖
- ✅ **类型安全**: 自动类型转换和验证
- ✅ **路径解析**: 相对路径自动转换为绝对路径
- ✅ **零侵入**: 不需要修改现有代码

---

## 快速开始

### 5分钟上手

#### 1. 加载配置（Python）

```python
from src.utils.config_loader import load_config, get_config

# 方式1: 加载完整配置
config = load_config()
queue_size = config['performance']['queue']['max_size']

# 方式2: 直接获取特定配置（推荐）
queue_size = get_config('performance.queue.max_size', default=50000)
log_level = get_config('logging.level', default='INFO')
```

#### 2. 指定运行环境

```bash
# 开发环境（默认）
python examples/run_realtime_cvd.py

# 测试环境
ENV=testing python examples/run_realtime_cvd.py

# 生产环境
ENV=production python examples/run_realtime_cvd.py
```

#### 3. 覆盖特定参数

```bash
# 临时增加队列大小
PERFORMANCE_QUEUE_MAX_SIZE=100000 python examples/run_realtime_cvd.py

# 临时改变日志级别
LOGGING_LEVEL=DEBUG python examples/run_realtime_cvd.py
```

---

## 配置架构

### 配置层次结构

```
┌─────────────────────────────────────────┐
│      环境变量 (最高优先级)                │
│  PERFORMANCE_QUEUE_MAX_SIZE=100000      │
└─────────────────────────────────────────┘
                    ↓ 覆盖
┌─────────────────────────────────────────┐
│   environments/{ENV}.yaml                │
│   (环境特定配置)                         │
└─────────────────────────────────────────┘
                    ↓ 覆盖
┌─────────────────────────────────────────┐
│   system.yaml                            │
│   (系统默认配置)                         │
└─────────────────────────────────────────┘
```

### 配置文件关系

```
config/
├── system.yaml              # 基础配置（所有环境共享）
│
├── environments/            # 环境特定配置（覆盖基础）
│   ├── development.yaml    # 开发环境覆盖
│   ├── testing.yaml        # 测试环境覆盖
│   └── production.yaml     # 生产环境覆盖
│
└── profiles/                # 组件特定配置（现有系统）
    ├── analysis.env        # CVD分析模式
    └── realtime.env        # CVD实时模式
```

### 配置优先级

**从高到低**:

1. 🥇 **环境变量** - 运行时覆盖
2. 🥈 **环境配置** - `environments/{ENV}.yaml`
3. 🥉 **系统配置** - `system.yaml`
4. 4️⃣ **默认值** - 代码中的默认值

---

## 配置文件详解

### system.yaml

**系统主配置文件** - 所有环境的基础配置

#### 主要配置节

```yaml
system:           # 系统元信息
data_source:      # 数据源配置
components:       # 组件开关
paths:            # 路径配置
performance:      # 性能参数
logging:          # 日志配置
monitoring:       # 监控配置
database:         # 数据库配置
testing:          # 测试配置
features:         # 特性开关
notifications:    # 通知配置
security:         # 安全配置
```

#### 关键配置示例

```yaml
# 性能配置
performance:
  queue:
    max_size: 50000                    # 队列大小
    full_behavior: "block"             # 队列满时行为
  
  flush:
    watermark_interval_ms: 200         # Watermark刷新间隔
    metrics_interval_ms: 10000         # 指标刷新间隔
  
  logging:
    print_every_trades: 1000           # 打印频率
    progress_interval_seconds: 60      # 进度间隔

# 日志配置
logging:
  level: "INFO"                        # 日志级别
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  file:
    enabled: true
    filename: "system.log"
    max_size_mb: 100
    backup_count: 5
  
  console:
    enabled: true
    level: "INFO"

# 组件配置
components:
  cvd:
    enabled: true
    config_file: "profiles/analysis.env"
  
  ofi:
    enabled: true
    config_file: "binance_config.yaml"
  
  ai:
    enabled: false                     # Stage 3
  
  trading:
    enabled: false                     # Stage 2
```

### environments/development.yaml

**开发环境配置** - 适合本地开发和调试

```yaml
# 覆盖系统配置
performance:
  queue:
    max_size: 10000                    # 较小的队列
  
  flush:
    watermark_interval_ms: 100         # 更频繁的刷新
    metrics_interval_ms: 5000
  
  logging:
    print_every_trades: 100            # 更频繁的打印
    progress_interval_seconds: 30

logging:
  level: "DEBUG"                       # 详细日志
  format: "%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s"

features:
  verbose_logging: true                # 详细日志
  profiling: true                      # 性能分析
  experimental: true                   # 实验性功能
```

### environments/testing.yaml

**测试环境配置** - 适合集成测试和验证

```yaml
performance:
  queue:
    max_size: 25000                    # 中等队列

logging:
  level: "INFO"                        # 标准日志

features:
  verbose_logging: false
  profiling: false
  experimental: false

testing:
  mode: "integration"
  coverage: true
```

### environments/production.yaml

**生产环境配置** - 适合实盘交易

```yaml
performance:
  queue:
    max_size: 100000                   # 大队列
  
  flush:
    metrics_interval_ms: 30000         # 减少刷新频率
  
  logging:
    print_every_trades: 5000           # 减少打印
    progress_interval_seconds: 300

logging:
  level: "WARNING"                     # 只记录警告和错误
  
  console:
    level: "ERROR"                     # 控制台只显示错误

monitoring:
  enabled: true                        # 必须启用监控

security:
  encrypt_api_keys: true
  rate_limiting:
    enabled: true
    max_requests_per_minute: 60

notifications:
  enabled: true                        # 启用通知
```

---

## 使用方法

### Python API

#### 基础用法

```python
from src.utils.config_loader import load_config, get_config, reload_config

# 1. 加载完整配置
config = load_config()
print(config['system']['name'])

# 2. 获取特定配置项（推荐）
queue_size = get_config('performance.queue.max_size')
log_level = get_config('logging.level')
data_dir = get_config('paths.data_dir')

# 3. 获取配置项（带默认值）
unknown_value = get_config('some.unknown.key', default='default_value')

# 4. 重新加载配置
config = reload_config()
```

#### 高级用法

```python
from src.utils.config_loader import ConfigLoader

# 1. 自定义配置目录
loader = ConfigLoader(config_dir='/path/to/config')
config = loader.load()

# 2. 强制重新加载
config = loader.load(reload=True)

# 3. 获取特定配置
value = loader.get('performance.queue.max_size', default=50000)
```

### 命令行用法

#### 指定环境

```bash
# Linux/Mac
export ENV=production
python examples/run_realtime_cvd.py

# Windows CMD
set ENV=production
python examples/run_realtime_cvd.py

# Windows PowerShell
$env:ENV="production"
python examples/run_realtime_cvd.py

# 一次性指定（推荐）
ENV=production python examples/run_realtime_cvd.py
```

#### 覆盖配置

```bash
# 覆盖单个配置
PERFORMANCE_QUEUE_MAX_SIZE=100000 python script.py

# 覆盖多个配置
ENV=production \
LOGGING_LEVEL=INFO \
PERFORMANCE_QUEUE_MAX_SIZE=100000 \
python script.py
```

---

## 环境变量覆盖

### 命名规则

**格式**: `SECTION_SUBSECTION_KEY`（大写，下划线分隔）

### 支持的层级

| 配置路径 | 环境变量 |
|---------|---------|
| `system.environment` | `SYSTEM_ENVIRONMENT` |
| `logging.level` | `LOGGING_LEVEL` |
| `performance.queue.max_size` | `PERFORMANCE_QUEUE_MAX_SIZE` |
| `features.verbose_logging` | `FEATURES_VERBOSE_LOGGING` |

### 类型自动转换

配置加载器会根据原始类型自动转换：

```bash
# 整数
PERFORMANCE_QUEUE_MAX_SIZE=100000       → 100000 (int)

# 浮点数
PERFORMANCE_FLUSH_WATERMARK_INTERVAL_MS=200.5  → 200.5 (float)

# 布尔值
FEATURES_VERBOSE_LOGGING=true           → True (bool)
FEATURES_VERBOSE_LOGGING=false          → False (bool)
FEATURES_VERBOSE_LOGGING=1              → True (bool)
FEATURES_VERBOSE_LOGGING=0              → False (bool)

# 字符串
LOGGING_LEVEL=DEBUG                     → "DEBUG" (str)
```

### 示例

```bash
# 示例1: 开发环境，但使用大队列
ENV=development \
PERFORMANCE_QUEUE_MAX_SIZE=100000 \
python examples/run_realtime_cvd.py

# 示例2: 生产环境，但启用详细日志（临时调试）
ENV=production \
LOGGING_LEVEL=DEBUG \
FEATURES_VERBOSE_LOGGING=true \
python examples/run_realtime_cvd.py

# 示例3: 测试环境，覆盖多个参数
ENV=testing \
PERFORMANCE_QUEUE_MAX_SIZE=50000 \
PERFORMANCE_FLUSH_WATERMARK_INTERVAL_MS=100 \
LOGGING_LEVEL=INFO \
python examples/run_realtime_cvd.py
```

---

## 配置验证

### 自动验证

配置加载器会自动验证：

1. ✅ **必需配置项**: 检查必需的配置节是否存在
2. ✅ **YAML格式**: 验证YAML文件格式正确
3. ✅ **类型正确性**: 确保配置值类型正确
4. ✅ **路径有效性**: 验证路径配置有效

### 验证失败处理

如果配置验证失败，会抛出 `ValueError` 并提示具体错误：

```python
try:
    config = load_config()
except ValueError as e:
    print(f"配置验证失败: {e}")
    # 处理错误...
```

### 手动验证

```python
from src.utils.config_loader import ConfigLoader

loader = ConfigLoader()
try:
    loader._validate_config(config)
    print("✅ 配置验证通过")
except ValueError as e:
    print(f"❌ 配置验证失败: {e}")
```

---

## 最佳实践

### 1. 环境管理

```bash
# ✅ 推荐: 明确指定环境
ENV=production python script.py

# ❌ 不推荐: 依赖默认环境（可能是development）
python script.py
```

### 2. 敏感信息处理

```yaml
# ❌ 不要在配置文件中写敏感信息
database:
  username: "admin"          # 不要这样做！
  password: "password123"    # 不要这样做！

# ✅ 使用环境变量占位符（说明）
database:
  # 从环境变量读取: DB_USER, DB_PASSWORD
  username: ${DB_USER}
  password: ${DB_PASSWORD}
```

```bash
# ✅ 通过环境变量传递
export DB_USER="admin"
export DB_PASSWORD="secure_password"
python script.py
```

### 3. 配置文件版本控制

```bash
# ✅ 提交到Git
git add config/system.yaml
git add config/environments/*.yaml

# ❌ 不要提交包含敏感信息的文件
# 在 .gitignore 中添加:
config/secrets.yaml
config/.env.local
```

### 4. 配置修改

```bash
# ✅ 修改配置文件
vi config/system.yaml
# 重启应用以加载新配置

# ✅ 临时覆盖（不修改文件）
PERFORMANCE_QUEUE_MAX_SIZE=100000 python script.py

# ❌ 不要在代码中硬编码配置
queue_size = 50000  # 硬编码，不推荐
```

### 5. 多环境测试

```bash
# ✅ 测试所有环境
ENV=development python -m src.utils.config_loader
ENV=testing python -m src.utils.config_loader
ENV=production python -m src.utils.config_loader
```

---

## 故障排除

### 问题1: 配置文件未找到

**错误**: `Configuration file not found: config/system.yaml`

**解决**:
```bash
# 检查当前目录
pwd

# 确保在项目根目录
cd /path/to/v13_ofi_ai_system

# 检查文件是否存在
ls -la config/system.yaml
```

### 问题2: YAML解析错误

**错误**: `Error parsing YAML file ...`

**解决**:
```bash
# 验证YAML格式
python -c "import yaml; yaml.safe_load(open('config/system.yaml'))"

# 检查缩进（必须使用空格，不要用Tab）
# 检查是否有未闭合的引号
```

### 问题3: 环境变量未生效

**错误**: 环境变量设置后，配置没有更新

**解决**:
```python
# 确保环境变量名称正确（大写、下划线）
# 错误: performance_queue_max_size
# 正确: PERFORMANCE_QUEUE_MAX_SIZE

# 检查环境变量是否正确设置
import os
print(os.getenv('PERFORMANCE_QUEUE_MAX_SIZE'))

# 强制重新加载配置
from src.utils.config_loader import reload_config
config = reload_config()
```

### 问题4: 路径错误

**错误**: 无法找到数据目录或日志目录

**解决**:
```python
# 检查解析后的路径
from src.utils.config_loader import load_config
config = load_config()
print(f"Data dir: {config['paths']['data_dir']}")
print(f"Logs dir: {config['paths']['logs_dir']}")

# 确保路径存在
import os
os.makedirs(config['paths']['data_dir'], exist_ok=True)
```

### 问题5: 配置优先级不符合预期

**错误**: 环境配置没有覆盖系统配置

**解决**:
```bash
# 确认环境设置正确
echo $ENV

# 检查环境配置文件是否存在
ls -la config/environments/$ENV.yaml

# 手动测试配置加载
python -m src.utils.config_loader
```

---

## 迁移指南

### 从 .env 迁移到 system.yaml

**重要**: 不需要立即迁移！系统完全向后兼容现有 `.env` 文件。

#### 阶段1: 共存（当前阶段）

```python
# 现有代码继续使用 .env
# 无需修改任何代码
python examples/run_realtime_cvd.py --symbol ETHUSDT
```

#### 阶段2: 可选启用（未来）

```python
# 在脚本中添加可选参数
if args.use_system_config:
    config = load_config()  # 使用新配置
else:
    config = load_env()     # 使用现有 .env（默认）
```

#### 阶段3: 完全迁移（更远的未来）

```python
# 所有新功能默认使用 system.yaml
# .env 文件仅用于组件特定配置
```

### 迁移清单

- [ ] 创建 `config/system.yaml`（已完成）
- [ ] 创建环境配置（已完成）
- [ ] 测试配置加载器（进行中）
- [ ] 更新文档（进行中）
- [ ] 在新功能中使用新配置
- [ ] 保持 `.env` 文件继续工作

---

## 附录

### 配置模板

完整的配置模板见：
- `config/system.yaml`
- `config/environments/development.yaml`
- `config/environments/testing.yaml`
- `config/environments/production.yaml`

### 相关文档

- **配置加载器源码**: `src/utils/config_loader.py`
- **配置目录说明**: `config/README.md`
- **CVD系统文件指南**: `docs/CVD_SYSTEM_FILES_GUIDE.md`
- **配置参数对比**: `docs/CONFIG_PARAMETERS_GUIDE.md`

### 测试配置加载器

```bash
# 运行配置加载器测试
cd v13_ofi_ai_system
python -m src.utils.config_loader

# 应该看到:
# ✅ Configuration loaded successfully!
# 📋 System: OFI_CVD_AI_Trading_System v13.0
# 🌍 Environment: development
# 📁 Data directory: /path/to/v13_ofi_ai_system/data
# 🔧 Queue size: 10000
# 📊 Log level: DEBUG
```

---

**版本**: v13.0  
**创建日期**: 2025-10-19  
**最后更新**: 2025-10-19  
**维护者**: V13 Team

---

**下一步**: 
- 📖 阅读 `config/README.md` 了解配置文件结构
- 🧪 运行 `python -m src.utils.config_loader` 测试配置系统
- 🚀 在新功能中使用 `load_config()` 加载配置

