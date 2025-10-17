# 🎯 Task 1.2.5 - 正式2小时测试指南

**测试状态**: ✅ **已启动** (2025-10-17 17:56)  
**预计完成**: 2025-10-17 19:56 (约2小时后)

---

## 📊 测试概览

### 测试参数
| 参数 | 值 | 说明 |
|------|-----|------|
| **运行时长** | 2小时 (7200秒) | 满足任务卡要求 |
| **数据频率** | 50 Hz | 50 msgs/s |
| **预期数据点** | ≈360,000点 | 7200秒 × 50 Hz |
| **采集模式** | DEMO模式 | 合成订单簿数据 |
| **数据目录** | `v13_ofi_ai_system/data/DEMO-USD/` | 从根目录运行，路径正确 |

### 验收标准
| 标准项 | 要求 | 预期结果 |
|--------|------|---------|
| **采样点数** | ≥300,000 | ✅ ≈360,000点 |
| **时间跨度** | ≥2小时 | ✅ ≈2小时 |
| **数据连续性** | max_gap≤2000ms | ✅ (冒烟测试520ms) |
| **分量和校验** | >99% | ✅ (冒烟测试100%) |
| **非空字段** | 无NULL | ✅ (冒烟测试通过) |
| **Z-score稳健性** | 6项指标 | ✅ (冒烟测试全通过) |
| **数据质量** | 2项指标 | ✅ (冒烟测试全通过) |
| **性能指标** | 4项指标 | ✅ (冒烟测试全通过) |

**预期通过率**: **17/17 (100%)** ✅

---

## ⏰ 时间线

| 时间 | 事件 | 说明 |
|------|------|------|
| 17:56 | ✅ 测试启动 | 进程已启动 (PID: 8656, 29456) |
| 18:06 | ⏱️ 10分钟 | 第一次进度检查 |
| 18:26 | ⏱️ 30分钟 | 应有≈90k点 |
| 18:56 | ⏱️ 1小时 | 应有≈180k点 |
| 19:26 | ⏱️ 1.5小时 | 应有≈270k点 |
| 19:56 | ✅ 2小时 | 测试完成，应有≈360k点 |
| 20:00 | 📊 运行分析 | 生成报告和图表 |

---

## 📁 预期产物

### 1. 数据文件
```
v13_ofi_ai_system/data/DEMO-USD/
└── 20251017_HHMM.parquet  (约4-5 MB)
    - 数据点: ≈360,000
    - 字段: 12个必需字段
    - 格式: Parquet
```

### 2. 日志文件
```
v13_ofi_ai_system/logs/
└── ws_YYYYMMDD_HHMM.log
```

### 3. 分析输出（测试完成后运行分析脚本）
```
v13_ofi_ai_system/examples/figs/
├── hist_z.png              (Z-score直方图)
├── ofi_timeseries.png      (OFI时间序列)
├── z_timeseries.png        (Z-score时间序列)
├── latency_box.png         (延迟箱线图)
└── analysis_results.json   (详细结果)

v13_ofi_ai_system/examples/
└── TASK_1_2_5_REPORT.md    (验收报告)
```

---

## 🔍 监控测试进度

### 方法1: 检查数据文件大小

```powershell
# 每10分钟运行一次
Get-ChildItem v13_ofi_ai_system\data\DEMO-USD\*.parquet | `
    Select-Object Name, `
    @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}, `
    LastWriteTime | Format-Table -AutoSize
```

**预期大小增长**:
- 10分钟: ≈0.4 MB (≈30k点)
- 30分钟: ≈1.2 MB (≈90k点)
- 1小时: ≈2.4 MB (≈180k点)
- 2小时: ≈4.8 MB (≈360k点)

### 方法2: 检查Python进程

```powershell
Get-Process python -ErrorAction SilentlyContinue | `
    Where-Object {$_.StartTime -gt (Get-Date).AddHours(-3)} | `
    Select-Object Id, @{Name='CPU(s)';Expression={[math]::Round($_.CPU,1)}}, `
    @{Name='Memory(MB)';Expression={[math]::Round($_.WorkingSet/1MB,1)}}, `
    @{Name='RunTime';Expression={(Get-Date) - $_.StartTime}} | `
    Format-Table -AutoSize
```

### 方法3: 快速检查脚本

```python
# 创建 check_progress.py
from pathlib import Path
import pandas as pd

data_dir = Path("v13_ofi_ai_system/data/DEMO-USD")
if data_dir.exists():
    files = list(data_dir.glob("*.parquet"))
    if files:
        df = pd.read_parquet(files[0])
        time_span_hours = (df['ts'].max() - df['ts'].min()) / 1000 / 3600
        print(f"数据点数: {len(df):,}")
        print(f"时间跨度: {time_span_hours:.2f} 小时")
        print(f"进度: {(time_span_hours / 2) * 100:.1f}%")
    else:
        print("未找到数据文件")
else:
    print("数据目录不存在")
```

---

## ⚠️ 注意事项

### 1. 不要关闭终端窗口
- 测试正在后台运行
- 关闭终端会导致进程终止

### 2. 不要手动停止进程
- 让测试自动运行满2小时
- 除非需要紧急停止

### 3. 电脑不要休眠
- 确保电源设置为"从不休眠"
- 或保持电脑活跃状态

### 4. 磁盘空间
- 预计占用≈5 MB
- 确保有足够磁盘空间

---

## 🐛 故障排查

### 问题1: 数据文件未生成

**检查**:
```powershell
Get-Process python | Where-Object {$_.StartTime -gt (Get-Date).AddHours(-1)}
```

**如果进程不存在**:
- 测试可能已崩溃
- 检查错误日志
- 重新启动测试

**如果进程存在但无数据**:
- 等待至少60秒（flush间隔）
- 检查环境变量 `ENABLE_DATA_COLLECTION=1`

### 问题2: 进程CPU占用过高

**正常情况**:
- CPU占用应在5-15%之间
- 如果>50%，可能有问题

**解决**:
- 检查是否有其他Python进程
- 考虑重启测试

### 问题3: 测试意外停止

**检查日志**:
```powershell
Get-Content v13_ofi_ai_system\logs\ws_*.log -Tail 50
```

**重新启动**:
```powershell
cd C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework
python v13_ofi_ai_system\examples\run_2hour_official_test.py
```

---

## 📝 测试完成后的操作

### 步骤1: 验证数据文件

```powershell
# 检查数据文件
Get-ChildItem v13_ofi_ai_system\data\DEMO-USD\*.parquet

# 快速统计
python -c "import pandas as pd; df = pd.read_parquet('v13_ofi_ai_system/data/DEMO-USD/20251017_HHMM.parquet'); print(f'Points: {len(df):,}, Hours: {(df[\"ts\"].max()-df[\"ts\"].min())/3600000:.2f}')"
```

**验收标准**:
- ✅ 数据点数 ≥ 300,000
- ✅ 时间跨度 ≥ 2小时

### 步骤2: 运行分析脚本

```powershell
cd C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework

python v13_ofi_ai_system\examples\analysis.py `
    --data v13_ofi_ai_system\data\DEMO-USD `
    --out v13_ofi_ai_system\examples\figs `
    --report v13_ofi_ai_system\examples\TASK_1_2_5_REPORT.md
```

**预期输出**:
```
找到 1 个 parquet 文件
✓ 已添加run_id列，便于分运行统计

总数据点数: 360000
时间跨度: 2.00 小时
✓ 数据已按时间戳排序

============================================================
1. 数据覆盖验证
============================================================
采样点数: 360000 (✓ 通过)
时间跨度: 2.00 小时 (✓ 通过)
最大时间缺口: XXX.XX ms (✓ 通过)
...

最终结果: ✅ 全部通过
```

### 步骤3: 检查生成的文件

```powershell
# 图表
Get-ChildItem v13_ofi_ai_system\examples\figs\*.png

# 报告
Get-Content v13_ofi_ai_system\examples\TASK_1_2_5_REPORT.md

# JSON结果
Get-Content v13_ofi_ai_system\examples\figs\analysis_results.json
```

### 步骤4: 更新任务卡

如果所有验收标准通过：
1. 将 `Task_1.2.5_OFI计算测试.md` 重命名为 `✅Task_1.2.5_OFI计算测试.md`
2. 更新任务卡中的"测试结果"部分
3. 提交Git (可选)

---

## 🎯 成功标准

### 最终验收清单

- [ ] 数据文件已生成 (≈5 MB)
- [ ] 采样点数 ≥ 300,000
- [ ] 时间跨度 ≥ 2小时
- [ ] 数据连续性: max_gap ≤ 2000ms
- [ ] 分量和校验通过率 > 99%
- [ ] 非空字段检查: 全部通过
- [ ] Z-score中位数 ∈ [-0.1, +0.1]
- [ ] Z-score IQR ∈ [0.8, 1.6]
- [ ] |Z|>2 占比 ∈ [1%, 8%]
- [ ] |Z|>3 占比 ≤ 1.5%
- [ ] std_zero标记次数 == 0
- [ ] warmup占比 ≤ 10%
- [ ] 坏数据点率 ≤ 0.1%
- [ ] 解析错误 == 0
- [ ] 处理延迟p95 < 5ms
- [ ] 重连频率 ≤ 3次/小时
- [ ] 队列丢弃率 ≤ 0.5%
- [ ] 4张图表已生成
- [ ] 分析报告已生成
- [ ] JSON结果已生成

**目标**: **17/17 (100%)** ✅

---

## 📞 联系与支持

### 如果遇到问题

1. **检查日志**: `v13_ofi_ai_system/logs/ws_*.log`
2. **检查进程**: `Get-Process python`
3. **检查数据**: `v13_ofi_ai_system/data/DEMO-USD/`
4. **重新启动**: 运行 `run_2hour_official_test.py`

### 参考文档

- **任务卡**: `v13_ofi_ai_system/TASKS/Stage1_真实OFI+CVD核心/Task_1.2.5_OFI计算测试.md`
- **冒烟测试报告**: `v13_ofi_ai_system/examples/SMOKE_TEST_SUCCESS_REPORT.md`
- **快速测试指南**: `v13_ofi_ai_system/examples/QUICK_SMOKE_TEST.md`

---

**测试启动时间**: 2025-10-17 17:56:00  
**预计完成时间**: 2025-10-17 19:56:00  
**状态**: 🔄 **正在运行**  
**进度**: ⏱️ **0/120 分钟**

---

## ⏭️ 测试完成后的下一步

完成Task 1.2.5后，可以继续：

1. **Task 1.2.6**: 性能基准测试
2. **Task 1.2.7**: CVD计算器开发
3. **Task 1.2.8**: CVD+OFI联合计算
4. **Stage 2**: 数据存储与回放系统

**当前进度**: Stage 1 - 真实OFI+CVD核心 (Task 1.2.5/1.2.8)

---

**祝测试顺利！** 🎉

