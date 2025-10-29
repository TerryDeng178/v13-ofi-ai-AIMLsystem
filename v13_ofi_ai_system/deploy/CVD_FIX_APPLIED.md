# CVD数据修复日志

## 修复时间
2025-10-27

## 修复内容
在 `run_success_harvest.py` 的 `_calculate_cvd` 方法中添加了 `None` 值处理逻辑：

```python
# 处理z_cvd为None的情况（warmup期间）
z_cvd_value = result.get('z_cvd')
if z_cvd_value is None:
    z_cvd_value = 0.0  # warmup期间使用0.0

return {
    ...
    'z_cvd': z_cvd_value,  # 使用处理后的值
    ...
}
```

## 影响
- warmup期间CVD数据现在会被保存（z_cvd=0.0）
- CVD数据量应该接近prices数据量
- 不再丢失warmup期间的CVD信息

## 验证方法
重新运行采集器至少30分钟，然后检查：
```bash
python check_data_summary.py
```

预期结果：CVD数据量应该在50000行以上（接近prices数据量）


