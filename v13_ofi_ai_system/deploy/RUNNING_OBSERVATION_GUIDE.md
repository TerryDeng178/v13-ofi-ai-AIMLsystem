# 运行观察小抄 - 确认"长跑稳定"

## 正常运行指标

### 1. 定时轮转日志（每~60秒）
```
执行定时轮转: 60.1秒 (模式: NORMAL)
[ROTATE] BTCUSDT: trade_delta=45.2s, ob_delta=12.1s, buffers={'prices': 150, 'ofi': 80}, reconnect_count=2, mode=NORMAL, queue_dropped_delta=0
```

**关键指标解读**：
- `trade_delta` 与 `ob_delta` 应持续低于各自超时阈值（150s/180s）
- `reconnect_count` 间歇性上涨是正常的（网络抖动/超时自愈）
- `queue_dropped_delta` 应该很小或为0
- `buffers` 大小应该稳定，不会持续增长

### 2. 重连日志（正常现象）
```
[TRADE] 120s 未收到消息，触发重连
统一交易流连接错误: ...
统一流重连(#3)
连接统一交易流: 6个symbol
统一交易流连接成功
```

**预期行为**：
- 出现 `[TRADE] 120s 未收到消息，触发重连` 是正常的
- 应该伴随 `统一交易流连接成功` 重新建立
- `kind=prices` 的新 `part-*.parquet` 会继续滚动生成

### 3. 超时关系校验（启动时）
```
[CONFIG] 超时关系校验: STREAM_IDLE_SEC(120) < TRADE_TIMEOUT(150) ✓
[CONFIG] 超时关系校验: STREAM_IDLE_SEC(120) < ORDERBOOK_TIMEOUT(180) ✓
```

**必须看到**：两个校验都显示 ✓，确保读超时早于健康告警

### 4. 健康检查日志（每25秒）
```
[HEALTH][TRADE] BTCUSDT 交易流超时 185.2s
[HEALTH][OB] ETHUSDT 订单簿流超时 305.1s
```

**注意**：如果频繁出现这些警告，说明流确实有问题

## 异常情况识别

### 1. 数据中断
**症状**：长时间没有新的 `part-*.parquet` 文件生成
**排查**：
- 检查 `trade_delta` 和 `ob_delta` 是否持续增长
- 查看是否有重连日志
- 检查网络连接状态

### 2. 内存压力
**症状**：`buffers` 大小持续增长
**排查**：
- 检查是否进入极端流量模式
- 查看是否有 `[SPILL]` 日志
- 检查磁盘空间

### 3. 频繁重连
**症状**：`reconnect_count` 增长过快
**排查**：
- 检查网络稳定性
- 调整 `STREAM_IDLE_SEC` 参数
- 查看是否有 `[CONFIG] 警告` 日志

### 4. 数据丢失
**症状**：`queue_dropped_delta` 持续增长
**排查**：
- 检查 `[DEDUP]` 警告日志
- 调整 `DEDUP_LRU` 参数
- 查看 `[DEADLETTER]` 日志

## 性能监控

### 1. 关键指标
- **数据完整性**：`queue_dropped` 应该很小
- **连接稳定性**：`reconnect_count` 增长合理
- **内存使用**：`buffers` 大小稳定
- **文件生成**：`part-*.parquet` 持续生成

### 2. 告警阈值
- `trade_delta` > 120s：可能有问题
- `ob_delta` > 150s：可能有问题
- `queue_dropped_delta` > 100：需要关注
- `reconnect_count` 增长 > 10/小时：需要关注

### 3. 日志关键词
- `[ROTATE]`：正常轮转
- `[TRADE]`：交易流重连
- `[ORDERBOOK]`：订单簿流重连
- `[HEALTH]`：健康检查警告
- `[DEDUP]`：去重警告
- `[DEADLETTER]`：数据丢失
- `[SPILL]`：内存溢出
- `[EXTREME_TRAFFIC]`：极端流量模式

## 故障排除

### 1. 流卡死
**现象**：长时间无数据
**解决**：
- 检查 `STREAM_IDLE_SEC` 设置
- 确认 `ping_timeout` 配置
- 查看网络连接状态

### 2. 内存溢出
**现象**：`[SPILL]` 日志频繁
**解决**：
- 调整 `EXTREME_TRAFFIC_THRESHOLD`
- 减少 `MAX_ROWS_PER_FILE`
- 增加 `SAVE_CONCURRENCY`

### 3. 数据丢失
**现象**：`queue_dropped` 增长
**解决**：
- 增加 `DEDUP_LRU` 大小
- 检查去重逻辑
- 查看 `[DEADLETTER]` 目录

### 4. 性能问题
**现象**：轮转延迟
**解决**：
- 增加 `SAVE_CONCURRENCY`
- 优化磁盘I/O
- 检查系统资源

## 长期运行检查清单

### 每日检查
- [ ] 查看最新的 `part-*.parquet` 文件时间戳
- [ ] 检查 `queue_dropped` 计数
- [ ] 查看 `reconnect_count` 增长趋势
- [ ] 检查磁盘空间使用

### 每周检查
- [ ] 分析 `slices_manifest` 报告
- [ ] 检查场景覆盖统计
- [ ] 查看 `[DEADLETTER]` 目录大小
- [ ] 分析性能趋势

### 每月检查
- [ ] 检查数据完整性
- [ ] 分析重连模式
- [ ] 优化配置参数
- [ ] 更新监控告警

## 配置调优建议

### 生产环境
```bash
export STREAM_IDLE_SEC=120
export TRADE_TIMEOUT=150
export ORDERBOOK_TIMEOUT=180
export HEALTH_CHECK_INTERVAL=25
export SAVE_CONCURRENCY=2
export MAX_ROWS_PER_FILE=50000
export EXTREME_TRAFFIC_THRESHOLD=30000
```

### 高负载环境
```bash
export STREAM_IDLE_SEC=90
export TRADE_TIMEOUT=120
export ORDERBOOK_TIMEOUT=150
export HEALTH_CHECK_INTERVAL=20
export SAVE_CONCURRENCY=4
export MAX_ROWS_PER_FILE=30000
export EXTREME_TRAFFIC_THRESHOLD=20000
```

### 测试环境
```bash
export STREAM_IDLE_SEC=30
export TRADE_TIMEOUT=45
export ORDERBOOK_TIMEOUT=60
export HEALTH_CHECK_INTERVAL=10
export SAVE_CONCURRENCY=2
export MAX_ROWS_PER_FILE=10000
export EXTREME_TRAFFIC_THRESHOLD=5000
```

记住：**稳定运行的关键是监控和及时响应异常**！
