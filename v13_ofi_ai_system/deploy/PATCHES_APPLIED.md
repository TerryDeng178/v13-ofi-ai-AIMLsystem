# 验证脚本优化补丁应用记录

## 已应用的补丁

### 补丁1：优化完整性检查时区处理
**问题**：cutoff/df['minute']的tz对齐分支过于复杂，极端情况下会误用tz_localize(None)

**修复**：
- 统一转为UTC-naive后直接比较，不做tz_localize
- 期望分钟数改为"覆盖窗口内理论分钟数"，更贴近真实空桶率
- 使用 `cutoff.replace(tzinfo=None)` 直接转为naive
- 计算span_start时与回看窗口取交集

### 补丁2：延迟检查增加回退逻辑
**问题**：仅依赖现成latency_ms，缺失就"无延迟"

**修复**：
- 添加 `_fallback_latency()` 函数，回退自算：recv_ts_ms - event_ts_ms
- prices/orderbook 延迟检查都增加了回退逻辑
- 过滤负值延迟数据

### 补丁3：Trade↔OB匹配字段名鲁棒性增强
**问题**：对agg_trade_id大小写/字段名不鲁棒

**修复**：
- 抽象出trade_id列名，支持多种字段名：['agg_trade_id','aggTradeId','trade_id','id']
- 带兜底逻辑，避免样本为0导致匹配率扭曲

### 补丁5：DoD事件数阈值优化
**问题**：对无preview/events的场景不友好，容易必然FAIL

**修复**：
- 当events全空但fusion/cvd指标充足时，将"事件数"判定降级为信息项
- 条件：fusion > 1000 或 cvd > 1000 时，即使events=0也通过检查

## 建议运行参数

```bash
python validate_ofi_cvd_harvest.py \
  --base-dir data/ofi_cvd \
  --preview-dir preview/ofi_cvd \
  --lookback-mins 180 \
  --join-tol-ms 900 \
  --seq-huge-jump 20000
```

**参数说明**：
- `--join-tol-ms 900`：与采集器的OFI对齐放宽一致，更真实
- `--seq-huge-jump 20000`：适配高波动时的u跳变

## 待优化项（可选）

### 补丁4：内存优化（大文件加载）
**建议**：按列选择性读取（pyarrow支持），在 `load_parquet_data_by_dir` 里只读本检查需要的列

**实现建议**：
```python
NEEDED = {
  'prices': ['symbol','event_ts_ms','ts_ms','recv_ts_ms','latency_ms','agg_trade_id','price','qty'],
  'orderbook': ['symbol','event_ts_ms','ts_ms','recv_ts_ms','latency_ms','first_id','last_id','prev_last_id',
                'best_bid','best_ask','d_bid_qty_agg','d_ask_qty_agg'] + [f'd_b{i}' for i in range(5)] + [f'd_a{i}' for i in range(5)],
  # ... 其他kinds
}
```

## 测试建议

1. 使用当前数据测试验证脚本是否正常运行
2. 检查输出报告是否合理
3. 观察内存使用情况，如需要再应用补丁4

## 修改文件

- `validate_ofi_cvd_harvest.py`
  - Line 130-161: check_completeness 时区处理优化
  - Line 240-246: 新增 _fallback_latency 函数
  - Line 267-296: check_latency 增加回退逻辑
  - Line 611-618: Trade↔OB 匹配字段名鲁棒性增强
  - Line 841-845: DoD事件数阈值降级逻辑

