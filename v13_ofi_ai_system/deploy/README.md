# harvestd 部署指南

24x7自愈型OFI+CVD数据采集守护服务，带极简监控界面。

## 快速开始

### 方案A: Docker Compose (推荐)

```bash
cd v13_ofi_ai_system
docker-compose -f deploy/docker-compose.yml up -d
```

访问监控界面: http://localhost:8088/

### 方案B: Windows BAT脚本

1. 双击运行 `deploy\start_harvestd.bat` 启动守护进程
2. 访问监控界面: http://localhost:8088/
3. 使用 `deploy\check_status.bat` 检查状态
4. 按 Ctrl+C 停止守护进程

**注意**: 需要 Python 3.11+ 和 curl 命令（Windows 10+ 自带 curl）

### 方案C: Systemd (Linux VM/裸机)

1. 复制服务文件:
```bash
sudo cp deploy/harvestd.service /etc/systemd/system/
sudo nano /etc/systemd/system/harvestd.service  # 编辑WorkingDirectory和User
```

2. 启用并启动:
```bash
sudo systemctl daemon-reload
sudo systemctl enable harvestd
sudo systemctl start harvestd
```

3. 检查状态:
```bash
sudo systemctl status harvestd
curl http://localhost:8088/health
```

## 核心功能

### 自动恢复
- **进程级**: 子进程崩溃 → 指数退避重启 (1s → 2s → 4s → ... ≤ 60s)
- **系统级**: 通过 `Restart=always` (systemd) 或 `restart: unless-stopped` (Docker)

### 数据质量监控
- 每小时验证: 自动运行 `validate_ofi_cvd_harvest.py`
- DoD检查: 完整性、去重、延迟、信号量、一致性、2×2场景覆盖
- 告警阈值: 连续2次失败 → `/health` 返回 503

### 极简Web界面端点

| 端点 | 描述 |
|------|------|
| `/` | 状态页（每5秒自动刷新） |
| `/health` | 健康检查（200=正常，503=降级） |
| `/logs` | 最近日志行 |
| `/dq` | 最新数据质量报告（JSON） |
| `/orderbook` | 订单簿数据收集状态（JSON） |
| `/metrics` | Prometheus指标 |

## 配置

### 环境变量

#### 守护进程配置
- `HARVESTD_PORT`: HTTP UI端口（默认: 8088）
- `VALIDATE_INTERVAL_MIN`: 数据质量检查间隔（分钟，默认: 60）
- `RESTART_BACKOFF_MAX_SEC`: 最大重启延迟（秒，默认: 60）
- `DQ_FAIL_MAX_TOL`: 触发告警的最大连续失败次数（默认: 2）

#### 数据采集配置（来自Task 1.3.1）
- `SYMBOLS`: 逗号分隔的交易对
- `RUN_HOURS`: 采集时长（默认: 72小时）
- `PARQUET_ROTATE_SEC`: 文件轮转间隔（秒，默认: 60）
- `WSS_PING_INTERVAL`: WebSocket心跳间隔（秒，默认: 20）
- `DEDUP_LRU`: 去重缓存大小（默认: 8192）

#### 订单簿数据收集配置（新增）
- `ENABLE_ORDERBOOK`: 启用订单簿数据收集（默认: 1）
- `ORDERBOOK_ROTATE_SEC`: 订单簿数据轮转间隔（秒，默认: 60）

#### 场景标签配置
- `SCENARIO_SCHEME`: 启用2×2场景（默认: regime2x2）
- `WIN_SECS`: 活跃度/波动率时间窗口（秒，默认: 300）
- `ACTIVE_TPS`: 活跃模式阈值（默认: 2.0）
- `VOL_SPLIT`: 波动率百分位分割点（默认: 0.5）

完整环境变量参考参见 Task 1.3.1 文档。

## 监控

### 健康检查

```bash
# 基本健康检查
curl http://localhost:8088/health

# 订单簿数据收集状态
curl http://localhost:8088/orderbook

# 使用监控工具
watch -n 5 'curl -s http://localhost:8088/health'
```

### Prometheus集成

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'harvestd'
    static_configs:
      - targets: ['localhost:8088']
```

暴露的指标:
- `harvestd_restarts_total` (计数器)
- `harvestd_dq_fail_streak` (仪表盘)
- `harvestd_uptime_seconds` (仪表盘)

### 日志

**Systemd:**
```bash
sudo journalctl -u harvestd -f
```

**Docker:**
```bash
docker-compose -f deploy/docker-compose.yml logs -f
```

**文件位置:**
- 守护进程日志: `artifacts/run_logs/`
- 数据质量报告: `artifacts/dq_reports/`
- 采集数据: `data/ofi_cvd/`

## 目录结构

```
v13_ofi_ai_system/
├── tools/
│   └── harvestd.py                      # 守护进程实现
├── examples/
│   └── run_success_harvest.py           # 数据采集脚本
├── scripts/
│   └── validate_ofi_cvd_harvest.py      # 数据质量验证脚本
├── deploy/
│   ├── docker-compose.yml               # Docker部署配置
│   ├── harvestd.service                 # Systemd服务文件
│   ├── start_harvestd.bat               # Windows启动脚本
│   ├── restart_harvestd.bat             # Windows重启脚本（支持订单簿）
│   ├── check_status.bat                 # Windows状态检查脚本
│   ├── check_orderbook.bat             # 订单簿数据收集状态检查
│   ├── diagnose.bat                     # 诊断工具
│   ├── quick_test.bat                   # 快速测试脚本
│   └── README.md                        # 本文档
├── data/
│   └── ofi_cvd/                         # 采集数据输出
└── artifacts/
    ├── run_logs/                        # 守护进程日志
    └── dq_reports/                      # 数据质量报告
```

## 故障排查

### 无法访问 http://localhost:8088/

**Windows平台快速诊断：**

1. 运行 `deploy\diagnose.bat` - 完整诊断工具
2. 运行 `deploy\quick_test.bat` - 快速端口测试

**可能原因及解决方案：**

#### 1. 守护进程未启动

```bash
# 检查进程是否运行
tasklist | findstr python

# 手动启动并查看输出
cd v13_ofi_ai_system
python tools\harvestd.py
```

**检查点：**
- 是否显示 `[ui] HTTP listening on :8088`
- 是否有错误堆栈
- Python版本是否 >= 3.11

#### 2. 端口被占用

```bash
# 查看端口占用
netstat -ano | findstr ":8088"

# 解决方案1: 停止占用进程
taskkill /PID <进程ID> /F

# 解决方案2: 修改端口
set HARVESTD_PORT=8089
python tools\harvestd.py
```

#### 3. 防火墙阻止

```powershell
# 临时添加端口例外
New-NetFirewallRule -DisplayName "harvestd" -Direction Inbound -LocalPort 8088 -Protocol TCP -Action Allow
```

#### 4. Python环境问题

```bash
# 检查Python版本
python --version
# 需要 >= 3.11

# 检查必要模块
python -c "import subprocess, threading, http.server"
```

### 进程不断重启

```bash
# Linux: 检查日志
sudo journalctl -u harvestd -n 100

# Windows: 查看控制台输出
# 注意看重启原因（backoff、异常等）

# 检查环境变量
systemctl show harvestd | grep Environment
```

### 数据质量检查失败

```bash
# 查看最新DQ报告
curl http://localhost:8088/dq

# 手动运行验证脚本
python scripts/validate_ofi_cvd_harvest.py --base-dir data/ofi_cvd
```

### 端口被占用

```bash
# 更改端口
export HARVESTD_PORT=8089
# 或 Windows
set HARVESTD_PORT=8089
python tools\harvestd.py
```

### WebSocket连接失败

```bash
# 检查网络连接
ping binance.com

# 检查代理设置
# 如果使用代理，确保WebSocket流量未被阻止
```

## 数据流

```
harvestd (守护进程)
  ├─> harvester 子进程: run_success_harvest.py
  │   └─> WebSocket → Binance Futures
  │   └─> OFI/CVD/Fusion 计算
  │   └─> 写入到 data/ofi_cvd/
  │
  ├─> validator 任务: validate_ofi_cvd_harvest.py (每小时)
  │   └─> 读取 data/ofi_cvd/
  │   └─> 生成 artifacts/dq_reports/dq_*.json
  │
  └─> HTTP UI: http://localhost:8088/
      ├─> /        状态页
      ├─> /health  健康检查
      ├─> /logs    查看日志
      ├─> /dq      数据质量报告
      └─> /metrics Prometheus指标
```

## 升级/更新

```bash
# 拉取最新代码
git pull

# 重启服务
sudo systemctl restart harvestd

# 或使用Docker
docker-compose -f deploy/docker-compose.yml restart
```

## 安全注意事项

- **网络**: 考虑使用防火墙规则限制UI访问
- **用户权限**: 服务运行在专用用户下（非root）
- **文件权限**: 确保 `data/` 和 `artifacts/` 对服务用户可写
- **环境变量**: 敏感配置保存在环境变量中，不要写入文件

## 支持

- 文档: 参见 `Task_1.3.1_收集历史OFI+CVD数据.md`
- 数据模式: 参见验证脚本输出 `artifacts/dq_reports/`
- 问题: 查看GitHub issues或联系维护者
