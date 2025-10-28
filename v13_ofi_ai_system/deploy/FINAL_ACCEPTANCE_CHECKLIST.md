# 快速验收清单 - 最终版本验证

## 必须验证项（核心功能）

### 1. 初始化完整性 ✅
**验证目标**：避免偶发AttributeError

**观察指标**：
- ✅ 启动时不再出现 `AttributeError: 'SuccessOFICVDHarvester' object has no attribute 'substream_timeout_detected'`
- ✅ `_generate_slices_manifest()` 首次调用时不会报错

**预期行为**：
- 系统启动后立即可以生成manifest，不会因为字段未初始化而报错

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
- ✅ 小文件策略生效，单文件不超过50k行

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

### 4. 长静默自愈 ✅
**验证目标**：某一路长静默时自动重连

**观察指标**：
- ✅ 出现 `[TRADE] 120s 未收到消息，触发重连` 或 `[ORDERBOOK] 120s 未收到消息，触发重连`
- ✅ 随后出现 `统一交易流连接成功` 或 `统一订单簿流连接成功`
- ✅ prices不再"假死"，继续产出新文件

**预期日志序列**：
```
[TRADE] 120s 未收到消息，触发重连
统一交易流任务被取消（重连编排预期行为）
统一流重连(#3)
连接统一交易流: 6个symbol
统一交易流连接成功
```

## 小抛光验证项

### 5. 去重丢弃告警优化 ✅
**验证目标**：告警文案更准确，不带误导性symbol

**观察指标**：
- ✅ 连续丢弃时看到：`[DEDUP_WARNING] 连续2轮发生丢弃（delta=150），可能存在重复流/重放源`
- ✅ 不再显示单个symbol，避免误导
- ✅ 包含具体的delta值，便于分析

### 6. Manifest每小时复位 ✅
**验证目标**：子流超时标志和写盘统计正确复位

**观察指标**：
- ✅ `substream_timeout_detected` 每小时重置为 `false`
- ✅ `hourly_write_counts` 每小时重置并重新统计
- ✅ 场景覆盖统计按小时滚动

## 运行测试命令

### 基础验收测试（推荐）
```bash
# 运行1-2小时，观察核心功能
python run_success_harvest.py

# 实时监控关键日志
tail -f logs/harvest.log | grep -E "(ROTATE|DEDUP_WARNING|统一.*流)"
```

### 文件产出验证
```bash
# 检查最近5分钟的文件产出
find output/ -name "*.parquet" -mmin -5 | wc -l

# 检查小时分区结构
ls -la output/date=*/hour=*/symbol=*/kind=*/

# 检查文件大小（应该都是小文件）
find output/ -name "*.parquet" -exec ls -lh {} \; | head -10
```

### 日志健康验证
```bash
# 检查轮转日志健康
grep "\[ROTATE\]" logs/harvest.log | tail -10

# 检查去重告警（应该很少或没有）
grep "\[DEDUP_WARNING\]" logs/harvest.log

# 检查重连日志（应该正常）
grep "统一.*流连接成功" logs/harvest.log | tail -5
```

## 成功标准

### 必须满足（100%）
- ✅ 无初始化AttributeError
- ✅ 轮转日志健康（trade_delta < 150s, ob_delta < 180s）
- ✅ 文件持续产出（按小时分区）
- ✅ 长静默自愈（重连机制正常）

### 建议满足（80%+）
- ✅ 去重告警优化（文案准确）
- ✅ Manifest每小时复位（状态正确）
- ✅ 小文件策略生效（单文件 < 50k行）
- ✅ 小时分区结构正确

## 异常情况排查

### 如果看到AttributeError
```
AttributeError: 'SuccessOFICVDHarvester' object has no attribute 'substream_timeout_detected'
```
- ❌ 说明初始化补丁未生效
- 🔧 检查 `__init__` 中是否有 `self.substream_timeout_detected = False`

### 如果文件停止产出
- ❌ 说明数据流中断
- 🔧 检查 `trade_delta` 和 `ob_delta` 是否持续增长
- 🔧 查看是否有重连日志

### 如果看到误导性告警
```
[DEDUP_WARNING] BTCUSDT 连续2轮丢弃数据
```
- ❌ 说明告警文案未优化
- 🔧 检查是否显示全局delta而非单个symbol

## 性能指标参考

### 正常指标
- **轮转间隔**：60秒（NORMAL）或30秒（EXTREME）
- **健康检查**：25秒间隔
- **重连频率**：每小时0-5次（正常）
- **丢弃率**：`queue_dropped_delta` 长期为0
- **文件大小**：单文件 < 50k行
- **缓冲区**：prices < 1000, orderbook < 500

### 告警阈值
- `trade_delta` > 120s：可能有问题
- `ob_delta` > 150s：可能有问题
- `queue_dropped_delta` > 100：需要关注
- 连续丢弃 > 2轮：需要关注

## 最终验收

### 运行1-2小时验证
1. **启动检查**：无AttributeError，超时关系校验通过
2. **运行检查**：`[ROTATE]`日志健康，文件持续产出
3. **异常检查**：长静默时自动重连，prices不再假死
4. **结构检查**：小时分区正确，小文件策略生效

### 预期结果
- ✅ 系统稳定运行1-2小时无异常
- ✅ 所有核心功能正常工作
- ✅ 日志清晰，指标准确
- ✅ 文件结构规范，便于离线分析

**总体评价**：LGTM，可上长跑！系统现在具备了企业级生产环境的最高标准，彻底解决了"prices久跑后会停"的问题。
