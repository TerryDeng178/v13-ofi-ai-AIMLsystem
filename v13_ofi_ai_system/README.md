# V13 OFI CVD Framework

V13 订单流不平衡 (Order Flow Imbalance) 和累积成交量差 (Cumulative Volume Delta) 框架

## 快速开始

### 运行策略模式管理器测试

```bash
# 运行冒烟测试
pytest tests/test_strategy_mode_smoke.py -v

# 运行手动走查脚本
python scripts/manual_mode_walkthrough.py
```

### 运行背离检测测试

```bash
# 运行单元测试
pytest tests/test_divergence_detector.py -v

# 运行端到端测试（使用真实数据）
python tests/test_divergence_e2e.py
```

### Pytest 配置

项目已配置 `pytest.ini` 和 `pyproject.toml` 来解决 pytest_asyncio 警告：

```ini
[tool:pytest]
asyncio_default_fixture_loop_scope = function
asyncio_default_test_loop_scope = function
```

如果仍然看到警告，可以运行：
```bash
# 使用配置文件运行
pytest --config-file=pytest.ini tests/test_strategy_mode_smoke.py -v
```

## 策略模式管理器测试

### 测试覆盖

策略模式管理器冒烟测试 (`tests/test_strategy_mode_smoke.py`) 覆盖以下场景：

1. **OR/AND 组合逻辑测试**
   - OR 逻辑：schedule OR market 任一满足即可进入 ACTIVE
   - AND 逻辑：schedule AND market 必须同时满足才进入 ACTIVE

2. **迟滞机制测试**
   - 需要连续确认才切换模式
   - Active 需要 `min_active_windows` 次确认
   - Quiet 需要 `min_quiet_windows` 次确认

3. **无副作用测试**
   - `decide_mode` 不重复调用状态函数
   - `_get_trigger_snapshot` 避免副作用
   - 指标更新正确

4. **指标更新测试**
   - 模式切换时更新 `strategy_mode_*` 指标
   - 不切换时只更新 `strategy_time_in_mode_seconds_total`
   - 事件结构完整

### 手动走查脚本

手动走查脚本 (`scripts/manual_mode_walkthrough.py`) 提供交互式验证：

#### 样例输出

```
============================================================
 策略模式管理器手动走查
============================================================
本脚本用于验证策略模式管理器的行为
包括模式切换、事件生成、指标更新等

============================================================
 关键指标键名
============================================================
模式相关:
  - strategy_mode_current: 当前模式 (0=quiet, 1=active)
  - strategy_mode_last_change_timestamp: 最后切换时间戳
  - strategy_time_in_mode_seconds_total: 各模式累计时长
  - strategy_mode_transitions_total: 模式切换次数

市场门控:
  - strategy_market_gate_basic_pass: 基础门槛 (0/1)
  - strategy_market_gate_quality_pass: 质量过滤 (0/1)
  - strategy_market_gate_window_pass: 窗口门槛 (0/1)
  - strategy_market_samples_window_size: 窗口样本数
  - strategy_market_samples_coverage_seconds: 窗口覆盖时长

触发器:
  - strategy_trigger_schedule_active: 时间表活跃 (0/1)
  - strategy_trigger_market_active: 市场活跃 (0/1)

============================================================
 OR vs AND 逻辑对比
============================================================

>>> OR 逻辑测试
============================================================
 配置信息
============================================================
组合逻辑: OR
迟滞窗口: 60秒
Active确认窗口: 1
Quiet确认窗口: 1
市场窗口: 60秒
市场门槛: trades>=100, quotes>=20

--- 第 1 次更新 ---
目标模式: active
处理耗时: 2.34ms
事件:
{
  "event": {
    "mode": "active",
    "reason": "schedule",
    "timestamp": "2025-01-27T10:30:15.123456+08:00",
    "triggers": {
      "schedule_active": true,
      "market_active": false,
      "schedule_market_logic": "OR",
      "current_mode": "quiet"
    },
    "update_duration_ms": 2.34
  }
}

>>> AND 逻辑测试
============================================================
 配置信息
============================================================
组合逻辑: AND
迟滞窗口: 60秒
Active确认窗口: 1
Quiet确认窗口: 1
市场窗口: 60秒
市场门槛: trades>=100, quotes>=20

--- 第 1 次更新 ---
目标模式: quiet
处理耗时: 1.89ms
无事件 (未切换)

============================================================
 迟滞机制测试
============================================================
============================================================
 配置信息
============================================================
组合逻辑: OR
迟滞窗口: 60秒
Active确认窗口: 2
Quiet确认窗口: 3
市场窗口: 60秒
市场门槛: trades>=100, quotes>=20

>>> 切换到 ACTIVE (需要2次确认)
============================================================
 场景: 活跃市场 (Schedule: ON)
============================================================

--- 第 1 次更新 ---
目标模式: quiet
处理耗时: 1.45ms
无事件 (未切换)

--- 第 2 次更新 ---
目标模式: active
处理耗时: 1.67ms
事件:
{
  "event": {
    "mode": "active",
    "reason": "schedule",
    "timestamp": "2025-01-27T10:30:17.234567+08:00",
    "triggers": {
      "schedule_active": true,
      "market_active": true,
      "schedule_market_logic": "OR",
      "current_mode": "quiet"
    },
    "update_duration_ms": 1.67
  }
}

>>> 切换到 QUIET (需要3次确认)
============================================================
 场景: 不活跃市场 (Schedule: OFF)
============================================================

--- 第 1 次更新 ---
目标模式: active
处理耗时: 1.23ms
无事件 (未切换)

--- 第 2 次更新 ---
目标模式: active
处理耗时: 1.34ms
无事件 (未切换)

--- 第 3 次更新 ---
目标模式: quiet
处理耗时: 1.56ms
事件:
{
  "event": {
    "mode": "quiet",
    "reason": "schedule",
    "timestamp": "2025-01-27T10:30:20.345678+08:00",
    "triggers": {
      "schedule_active": false,
      "market_active": false,
      "schedule_market_logic": "OR",
      "current_mode": "active"
    },
    "update_duration_ms": 1.56
  }
}

============================================================
 走查完成
============================================================
关键观察点:
1. OR逻辑: schedule OR market 任一满足即可
2. AND逻辑: schedule AND market 必须同时满足
3. 迟滞机制: 需要连续确认才切换
4. 事件结构: 包含完整的触发信息
5. 指标更新: 实时反映当前状态
```

## 配置说明

### 策略模式管理器配置

策略模式管理器从 `strategy.triggers` 读取配置：

```yaml
strategy:
  mode: auto
  hysteresis:
    window_secs: 60
    min_active_windows: 2
    min_quiet_windows: 3
  triggers:
    combine_logic: OR  # OR | AND
    schedule:
      enabled: true
      timezone: Asia/Hong_Kong
      calendar: CRYPTO
      enabled_weekdays: [Mon, Tue, Wed, Thu, Fri, Sat, Sun]
      holidays: []
      active_windows:
        - start: "00:00"
          end: "23:59"
          timezone: Asia/Hong_Kong
      wrap_midnight: true
    market:
      enabled: true
      window_secs: 60
      min_trades_per_min: 100
      min_quote_updates_per_sec: 20
      max_spread_bps: 5
      min_volatility_bps: 2
      min_volume_usd: 100000
      use_median: false
      winsorize_percentile: 95
      quality_multipliers: {}
      max_samples_per_sec: 2
```

### 关键参数说明

- **combine_logic**: 组合逻辑，`OR` 表示任一触发器满足即可，`AND` 表示必须同时满足
- **min_active_windows**: Active 模式确认窗口数，需要连续 N 次确认才切换
- **min_quiet_windows**: Quiet 模式确认窗口数，需要连续 N 次确认才切换
- **market.window_secs**: 市场数据滑动窗口时长
- **market.min_trades_per_min**: 每分钟最小交易数门槛
- **market.min_quote_updates_per_sec**: 每秒最小报价更新数门槛

## 架构组件

### 核心组件

- **StrategyModeManager**: 策略模式管理器，负责自动切换活跃/不活跃模式
- **DivergenceDetector**: 背离检测器，识别价格与指标的背离信号
- **OFICVDFusion**: OFI/CVD 融合计算器，生成综合指标
- **RealOFICalculator**: 实时 OFI 计算器
- **RealCVDCalculator**: 实时 CVD 计算器

### 测试框架

- **Pytest**: 单元测试框架
- **Mock**: 模拟对象测试
- **JUnit XML**: 测试报告格式
- **手动走查**: 交互式验证脚本

## 开发指南

### 添加新测试

1. 在 `tests/` 目录下创建测试文件
2. 使用 `pytest` 框架编写测试用例
3. 确保测试覆盖核心功能和边界情况
4. 添加适当的断言和验证

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_strategy_mode_smoke.py

# 运行测试并生成报告
pytest --junitxml=reports/junit.xml tests/

# 运行测试并显示覆盖率
pytest --cov=src tests/
```

### 调试技巧

1. 使用手动走查脚本验证行为
2. 检查 Prometheus 指标确认状态
3. 查看事件日志了解切换原因
4. 使用 Mock 对象隔离测试环境

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。
