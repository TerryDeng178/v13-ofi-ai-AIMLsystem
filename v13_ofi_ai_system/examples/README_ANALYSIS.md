# Analysis.py 使用说明

## 📋 概述

`analysis.py` 是 V13 系统的数据分析和验证工具，用于检查OFI/CVD数据质量并生成验收报告。

**模块**: `v13_ofi_ai_system/examples/analysis.py`  
**任务**: Task 1.2.5 完成后可复用于所有测试任务  
**功能**: 数据验证、统计分析、图表生成、验收报告

---

## 🚀 快速开始

### 基本用法

```bash
# 分析单个数据文件
python analysis.py \
    --data v13_ofi_ai_system/data/DEMO-USD/20251017_1826.parquet \
    --out v13_ofi_ai_system/figs \
    --report ANALYSIS_REPORT.md

# 分析整个目录（自动合并所有parquet文件）
python analysis.py \
    --data v13_ofi_ai_system/data/DEMO-USD \
    --out v13_ofi_ai_system/figs \
    --report ANALYSIS_REPORT.md
```

### 参数说明

| 参数 | 必选 | 说明 | 示例 |
|------|------|------|------|
| `--data` | ✅ | 数据文件或目录路径 | `data/DEMO-USD/file.parquet` 或 `data/DEMO-USD/` |
| `--out` | ❌ | 图表输出目录 | `figs/` (默认: `v13_ofi_ai_system/examples/figs`) |
| `--report` | ❌ | 报告输出文件 | `report.md` (默认: `v13_ofi_ai_system/examples/TASK_1_2_5_REPORT.md`) |

---

## 📊 验收标准

`analysis.py` 会自动检查以下验收标准：

### 1. 数据覆盖 (3项)

| 检查项 | 阈值 | 说明 |
|--------|------|------|
| **采样点数** | ≥300,000 | 数据量是否充足 |
| **数据连续性** | max_gap ≤2000ms | 时间戳连续性 |
| **时间跨度** | ≥2小时 | 采集时长是否足够 |

### 2. 功能正确性 (2项)

| 检查项 | 阈值 | 说明 |
|--------|------|------|
| **分量和校验** | >99% | `abs(k_components_sum - ofi) < 1e-9` |
| **非空字段** | 100% | 必需字段无NULL值 |

### 3. Z-score标准化 (6项)

| 检查项 | 阈值 | 说明 |
|--------|------|------|
| **中位数居中** | ∈[-0.1, +0.1] | Z-score分布是否居中 |
| **IQR合理** | ∈[0.8, 1.6] | 四分位距是否正常 |
| **\|Z\|>2占比** | ∈[1%, 8%] | 尾部占比是否合理 |
| **\|Z\|>3占比** | ≤1.5% | 极端值是否过多 |
| **std_zero标记** | ==0 | 标准差为0的次数 |
| **warmup占比** | ≤10% | 预热期占比 |

### 4. 数据质量 (2项)

| 检查项 | 阈值 | 说明 |
|--------|------|------|
| **坏数据点率** | ≤0.1% | 无效数据占比 |
| **解析错误** | ==0 | 解析失败次数 |

### 5. 性能指标 (3项)

| 检查项 | 阈值 | 说明 |
|--------|------|------|
| **处理延迟p95** | <5ms | 95分位延迟 |
| **重连频率** | ≤3次/小时 | WebSocket重连频率 |
| **队列丢弃率** | ≤0.5% | backpressure触发率 |

---

## 📈 输出文件

### 1. 图表文件 (4张必选)

生成在 `--out` 指定目录：

- **hist_z.png** - Z-score分布直方图
  - 验证正态性
  - 显示中位数、Q25、Q75

- **ofi_timeseries.png** - OFI原始时间序列
  - 采样10000点
  - 展示原始波动

- **z_timeseries.png** - Z-score时间序列
  - 仅非warmup期数据
  - 标记±2和±3阈值线

- **latency_box.png** - 处理延迟箱线图
  - 展示延迟分布
  - 识别异常值

### 2. 分析报告

生成在 `--report` 指定路径：

```markdown
# Task 1.2.5 OFI计算测试报告

**测试执行时间**: 2025-10-17 20:33:43
**数据源**: `../data/DEMO-USD/20251017_1826.parquet`

## 验收标准对照结果

### 1. 数据覆盖
- [x] 采样点数: 352,778 (≥300,000)
- [x] 数据连续性: max_gap=457.11ms (≤2000ms)
- [ ] 时间跨度: 1.99小时 (≥2小时)

...

## 结论
**✅ 所有验收标准通过，可继续下一任务**
```

### 3. JSON详细结果

生成在 `<out_dir>/analysis_results.json`：

```json
{
  "total_points": 352778,
  "time_span_hours": 1.99,
  "max_gap_ms": 457.11,
  "continuity_pass": true,
  "z_score": {
    "median": 0.0003,
    "iqr": 1.3696,
    "tail2_pct": 4.52,
    "tail3_pct": 0.20
  },
  ...
}
```

---

## 💡 典型使用场景

### 场景1: Task验收（单次测试）

```bash
# Task 1.2.5: OFI计算测试验收
python analysis.py \
    --data v13_ofi_ai_system/data/DEMO-USD/20251017_1826.parquet \
    --out v13_ofi_ai_system/figs \
    --report v13_ofi_ai_system/examples/TASK_1_2_5_REPORT.md

# 查看报告
cat v13_ofi_ai_system/examples/TASK_1_2_5_REPORT.md
```

### 场景2: 批量分析（多次测试）

```bash
# 分析整个目录（自动合并）
python analysis.py \
    --data v13_ofi_ai_system/data/DEMO-USD \
    --out v13_ofi_ai_system/figs \
    --report BATCH_ANALYSIS_REPORT.md
```

### 场景3: CVD数据验证（后续任务）

```bash
# Task 1.2.10: CVD计算测试验收
python analysis.py \
    --data v13_ofi_ai_system/data/CVD-USD/20251018_1000.parquet \
    --out v13_ofi_ai_system/figs/cvd \
    --report TASK_1_2_10_CVD_REPORT.md
```

### 场景4: 快速质量检查

```bash
# 只生成图表，不生成报告
python analysis.py \
    --data latest_data.parquet \
    --out quick_check/
```

---

## 📋 数据字段要求

### 必需字段

分析脚本要求输入数据包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `ts` | int64 | 本地时间戳（UTC毫秒） |
| `ofi` | float64 | OFI原始值 |
| `z_ofi` | float64 | Z-score标准化值（warmup期可为NULL） |
| `ema_ofi` | float64 | EMA平滑值 |
| `warmup` | bool | warmup标记 |
| `std_zero` | bool | 标准差为0标记 |
| `k_components_sum` | float64 | 各档OFI分量之和（用于校验） |

### 可选字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `event_time_ms` | int64 | 交易所事件时间（真实流必需） |
| `latency_ms` | float64 | 处理延迟（毫秒） |
| `queue_dropped` | int | 队列丢弃累计计数 |
| `reconnect_count` | int | 重连累计计数 |
| `bad_points` | int | 坏数据点累计计数 |

---

## ⚠️ 注意事项

### 数据格式

1. **Parquet格式**
   - 推荐使用Parquet格式（高效压缩）
   - 支持单文件或目录（自动合并）

2. **时间戳排序**
   - 脚本会自动按 `ts` 排序
   - 排序后才计算连续性

3. **多文件合并**
   - 目录模式自动合并所有 `.parquet` 文件
   - 脚本会为每个文件添加 `run_id` 列（取自文件名）
   - 便于区分不同运行、支持批量统计分析

### 性能建议

1. **数据量**
   - 单文件: 推荐<1000万点
   - 多文件: 推荐<10个文件
   - 大数据量会影响图表生成速度

2. **图表采样**
   - 时间序列图自动采样10000点
   - 直方图使用全量数据

3. **内存占用**
   - 100万点约需 100MB 内存
   - 1000万点约需 1GB 内存

### 常见问题

**Q: 分析失败怎么办？**  
A: 检查数据文件是否包含必需字段，查看错误信息。

**Q: 图表不显示怎么办？**  
A: 确认输出目录存在且有写入权限。

**Q: 多个文件如何合并？**  
A: 直接指定目录路径，脚本自动合并所有 `.parquet` 文件。

**Q: 如何只生成特定图表？**  
A: 当前版本生成全部4张图表，无法单独指定。

---

## 🔧 高级用法

### 1. 修改验收阈值

如需自定义阈值，直接编辑 `analysis.py`（按关键字搜索定位）：

```python
# 搜索 "total_points >= 300000" 修改采样点数阈值
results['total_points'] >= 300000  # 改为你的阈值

# 搜索 "queue_dropped_rate <= 0.005" 修改队列丢弃率阈值
results['queue_dropped_pass'] = queue_dropped_rate <= 0.005  # 改为 0.01
```

### 2. 添加自定义检查

在脚本中添加新的验证逻辑：

```python
# 在 main() 函数中添加
custom_check = df['ofi'].abs().max() < 10.0  # 示例：检查OFI极值
results['custom_check_pass'] = custom_check
```

### 3. 导出原始数据

```python
import pandas as pd

# 读取分析结果
df = pd.read_parquet('data.parquet')

# 导出特定列
df[['ts', 'ofi', 'z_ofi']].to_csv('ofi_series.csv', index=False)
```

---

## 🔗 相关文档

- `run_realtime_ofi.py` - 生成待分析的数据
- `README_realtime_ofi.md` - 数据采集文档
- `README_OFI_CALCULATOR.md` - OFI计算器说明
- Task卡: `✅Task_1.2.5_OFI计算测试.md`

---

## 📊 退出码

脚本退出码表示验收结果：

- **0**: ✅ 所有验收标准通过
- **1**: ❌ 部分验收标准未通过或执行异常

**说明**: 脚本异常（如文件不存在、字段缺失）也会返回1，无需区分2。Shell中判断非零即失败。

可在CI/CD中使用：

```bash
python analysis.py --data data.parquet --out figs/ --report report.md
if [ $? -eq 0 ]; then
    echo "✅ 验收通过"
else
    echo "❌ 验收失败或异常"
    exit 1
fi
```

---

## 🎯 最佳实践

1. **文件命名**
   - 数据文件: `YYYYMMDD_HHMM.parquet`
   - 报告文件: `TASK_X_X_X_REPORT.md`
   - 图表目录: 按任务分组 (`figs/task_1.2.5/`)

2. **版本管理**
   - 数据文件: 添加到 `.gitignore`（太大）
   - 图表: 可选择性提交（用于文档）
   - 报告: 建议提交（可追溯）

3. **批量分析**
   ```bash
   # 分析多个任务的数据
   for task in task_1.2.5 task_1.2.10; do
       python analysis.py \
           --data data/$task \
           --out figs/$task \
           --report reports/${task}_report.md
   done
   ```

4. **CI集成**
   ```yaml
   # .github/workflows/test.yml
   - name: Run Analysis
     run: |
       python analysis.py \
           --data test_data.parquet \
           --out artifacts/ \
           --report test_report.md
       exit_code=$?
       if [ $exit_code -ne 0 ]; then
           exit 1
       fi
   ```

---

**版本**: V13.1.2.5  
**最后更新**: 2025-10-17  
**维护者**: V13 OFI+CVD+AI System Team

