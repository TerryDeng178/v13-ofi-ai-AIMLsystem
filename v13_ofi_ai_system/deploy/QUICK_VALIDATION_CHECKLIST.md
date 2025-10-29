# 快速验证清单 - 确认"长跑稳定"

## 必须验证项（核心功能）

### 1. 取消错误处理修复 ✅
**验证目标**：不再出现"取消导致的错误日志 + 重连计数跳变"

**观察指标**：
- ✅ 启动时看到：`[CONFIG] 超时关系校验: STREAM_IDLE_SEC(120) < TRADE_TIMEOUT(150) ✓`
- ✅ 重连时看到：`统一交易流任务被取消（重连编排预期行为）`
- ✅ 不再看到：`统一交易流连接错误: CancelledError` 或类似错误
- ✅ `reconnect_count` 增长合理，不会因正常取消而虚高

**预期日志**：
```
[CONFIG] 超时关系校验: STREAM_IDLE_SEC(120) < TRADE_TIMEOUT(150) ✓
[CONFIG] 超时关系校验: STREAM_IDLE_SEC(120) < ORDERBOOK_TIMEOUT(180) ✓
统一交易流任务被取消（重连编排预期行为）
统一订单簿流任务被取消（重连编排预期行为）
```

### 2. 轮转日志健康 ✅
**验证目标**：`[ROTATE]` 打点中的关键指标持续正常

**观察指标**：
- ✅ `trade_delta` 连续低于 150s（TRADE_TIMEOUT）
- ✅ `ob_delta` 连续低于 180s（ORDERBOOK_TIMEOUT）
- ✅ `queue_dropped_delta` 长期为 0 或很小
- ✅ `buffers` 大小稳定，不会持续增长

**预期日志**：
```
[ROTATE] BTCUSDT: trade_delta=45.2s, ob_delta=12.1s, buffers={'prices': 150, 'ofi': 80}, reconnect_count=2, mode=NORMAL, queue_dropped_delta=0
[ROTATE] ETHUSDT: trade_delta=38.7s, ob_delta=15.3s, buffers={'prices': 120, 'ofi': 65}, reconnect_count=2, mode=NORMAL, queue_dropped_delta=0
```

### 3. 文件持续产出 ✅
**验证目标**：`kind=prices` 与 `kind=orderbook` 的目录持续产出文件

**观察指标**：
- ✅ `date=YYYY-MM-DD/hour=HH/symbol=BTCUSDT/kind=prices/` 目录持续更新
- ✅ `date=YYYY-MM-DD/hour=HH/symbol=BTCUSDT/kind=orderbook/` 目录持续更新
- ✅ 文件命名格式：`part-{timestamp}-{uuid}.parquet`
- ✅ 文件大小合理（不会过大或过小）

**预期文件结构**：
```
output/
├── date=2024-01-15/
│   ├── hour=10/
│   │   ├── symbol=BTCUSDT/
│   │   │   ├── kind=prices/
│   │   │   │   ├── part-1705312345678901234-abc123.parquet
│   │   │   │   └── part-1705312405678901234-def456.parquet
│   │   │   └── kind=orderbook/
│   │   │       ├── part-1705312345678901234-ghi789.parquet
│   │   │       └── part-1705312405678901234-jkl012.parquet
```

### 4. Manifest每小时重置 ✅
**验证目标**：`slices_manifest` 每小时正确重置和滚动

**观察指标**：
- ✅ `slices_manifest_YYYYMMDD_HH.json` 每小时生成
- ✅ `substream_timeout_detected` 每小时重置为 `false`
- ✅ `hourly_write_counts` 每小时重置并重新统计
- ✅ 场景覆盖统计按小时滚动

**预期manifest内容**：
```json
{
  "timestamp": "2024-01-15T10:00:00.000Z",
  "symbols": {
    "BTCUSDT": {
      "scenario_2x2": {"A_H": 45, "A_L": 23, "Q_H": 12, "Q_L": 8},
      "total_samples": 88,
      "coverage_ratio": {"A_H": 0.51, "A_L": 0.26, "Q_H": 0.14, "Q_L": 0.09}
    }
  },
  "hour_stats": {
    "prices_rows": 150,
    "orderbook_rows": 120,
    "reconnect_count": 2,
    "queue_dropped": 0,
    "substream_timeout_detected": false,
    "hourly_write_counts": {
      "prices": 5000,
      "orderbook": 4000,
      "ofi": 2000,
      "cvd": 2000,
      "fusion": 2000,
      "events": 1000,
      "features": 1000
    }
  }
}
```

## 可选验证项（增强功能）

### 5. Features盘口字段补齐 ✅
**验证目标**：features表包含盘口强相关字段

**观察指标**：
- ✅ `best_bid`, `best_ask`, `spread_bps` 字段不为空
- ✅ 字段值与orderbook数据一致

### 6. 健康检查自检 ✅
**验证目标**：每10轮健康检查进行超时关系自检

**观察指标**：
- ✅ 每10轮看到：`[HEALTH_SELF_CHECK] 超时关系正常: STREAM_IDLE_SEC(120) < TRADE_TIMEOUT(150)`
- ✅ 如果配置错误会看到警告

### 7. WebSocket参数护栏 ✅
**验证目标**：WebSocket连接使用护栏参数

**观察指标**：
- ✅ 连接日志显示使用 `max_size=8388608, close_timeout=5`
- ✅ 异常峰值时连接更可控

### 8. 去重LRU增量告警 ✅
**验证目标**：连续丢弃数据时及时告警

**观察指标**：
- ✅ 连续2轮丢弃时看到：`[DEDUP_WARNING] BTCUSDT 连续2轮丢弃数据，可能重复流/重放源`
- ✅ 正常时连续计数重置为0

### 9. 订单簿levels语义 ✅
**验证目标**：levels字段使用更准确的语义

**观察指标**：
- ✅ `levels` 字段值为 `min(len(bids), len(asks))`
- ✅ 更贴近"有效价差范围内的匹配深度"

### 10. 环境变量缓存 ✅
**验证目标**：热路径不再频繁查询环境变量

**观察指标**：
- ✅ CVD计算使用缓存的 `self.cvd_sigma_floor_k`
- ✅ 融合计算使用缓存的 `self.w_ofi`, `self.w_cvd`

## 运行测试命令

### 基础测试（推荐）
```bash
# 使用默认配置运行1小时
python run_success_harvest.py

# 观察关键日志
tail -f logs/harvest.log | grep -E "(ROTATE|CONFIG|HEALTH|DEDUP)"
```

### 压力测试
```bash
# 使用激进配置测试
export STREAM_IDLE_SEC=30
export TRADE_TIMEOUT=45
export ORDERBOOK_TIMEOUT=60
export HEALTH_CHECK_INTERVAL=10
export MAX_ROWS_PER_FILE=10000
export EXTREME_TRAFFIC_THRESHOLD=5000

python run_success_harvest.py
```

### 验证脚本
```bash
# 检查文件产出
find output/ -name "*.parquet" -mmin -5 | wc -l

# 检查manifest生成
ls -la output/slices_manifest_*.json | tail -5

# 检查日志健康
grep -c "统一.*流任务被取消" logs/harvest.log
grep -c "统一.*流连接错误" logs/harvest.log
```

## 异常情况排查

### 如果看到错误日志
1. **`统一交易流连接错误: CancelledError`**
   - ❌ 说明取消错误处理未生效
   - 🔧 检查 `except asyncio.CancelledError` 是否在 `except Exception` 之前

2. **`[CONFIG] 警告: STREAM_IDLE_SEC >= TRADE_TIMEOUT`**
   - ❌ 说明超时参数配置错误
   - 🔧 调整环境变量，确保 `STREAM_IDLE_SEC < TRADE_TIMEOUT < ORDERBOOK_TIMEOUT`

3. **`[DEDUP_WARNING] 连续丢弃数据`**
   - ⚠️ 说明可能有重复流或重放源
   - 🔧 检查数据源，调整 `DEDUP_LRU` 参数

4. **长时间无新文件生成**
   - ❌ 说明数据流中断
   - 🔧 检查 `trade_delta` 和 `ob_delta`，查看重连日志

### 性能指标参考
- **正常轮转间隔**：60秒（NORMAL模式）或30秒（EXTREME模式）
- **健康检查间隔**：25秒
- **重连频率**：每小时0-5次（网络抖动正常）
- **丢弃率**：`queue_dropped_delta` 应该长期为0
- **缓冲区大小**：prices < 1000, orderbook < 500（正常情况）

## 成功标准

### 必须满足（100%）
- ✅ 无取消错误日志
- ✅ 轮转日志健康
- ✅ 文件持续产出
- ✅ Manifest每小时重置

### 建议满足（80%+）
- ✅ Features盘口字段补齐
- ✅ 健康检查自检正常
- ✅ WebSocket参数护栏生效
- ✅ 去重LRU增量告警正常
- ✅ 订单簿levels语义正确
- ✅ 环境变量缓存生效

**总体评价**：如果所有必须满足项都通过，系统已经足够稳健用于"长跑"生产环境！
