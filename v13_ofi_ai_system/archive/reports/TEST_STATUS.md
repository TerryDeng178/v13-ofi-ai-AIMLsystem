# 🔍 测试状态实时追踪

## 📊 当前运行的测试

### ⚡ 5分钟快速验证
- **PID**: 1832
- **开始时间**: 2025-10-19 02:20:53
- **预计结束**: 2025-10-19 02:25:53
- **输出目录**: `data/cvd_quick_verify_20251019_0220`
- **状态**: 🟢 运行中

### 🏆 40分钟金测
- **PID**: 23960
- **开始时间**: 2025-10-19 02:15:55
- **预计结束**: 2025-10-19 02:55:55
- **输出目录**: `data/cvd_final_gold_20251019_0215`
- **状态**: 🟢 运行中

---

## 🛠️ 监控方式（3种选择）

### 方式1：双击批处理文件（推荐）
**位置**: `v13_ofi_ai_system/scripts/QUICK_CHECK.bat`

**操作**:
1. 在文件资源管理器中找到这个文件
2. 双击运行
3. 会弹出新窗口显示状态
4. 不阻塞Cursor，随时可以运行

**显示内容**:
- Python进程状态（PID、运行时长、CPU、内存）
- 测试输出目录（大小、最后更新时间）

---

### 方式2：启动持续监控（独立窗口）
**位置**: `v13_ofi_ai_system/scripts/START_MONITOR.bat`

**操作**:
1. 双击这个文件
2. 会打开独立的命令行窗口
3. 每30秒自动刷新显示详细状态
4. 按Ctrl+C停止监控（测试继续运行）

**显示内容**:
- 两个测试的进度百分比
- 已运行时间 / 剩余时间
- CPU和内存使用
- 已采集数据量
- 队列丢弃率
- Z-score实时指标

---

### 方式3：PowerShell命令（手动）
在独立的PowerShell窗口中运行：

```powershell
# 检查进程
Get-Process python | Select Id, StartTime, CPU, WorkingSet

# 查看测试目录
Get-ChildItem C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\data -Directory | Where {$_.Name -like 'cvd_*_2025*'} | Sort LastWriteTime -Desc | Select -First 3

# 查看最新报告
Get-Content C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\data\cvd_quick_verify_20251019_0220\report_*.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

---

## ⏰ 时间线

| 时间 | 事件 |
|------|------|
| 02:15:55 | 40分钟金测开始 |
| 02:20:53 | 5分钟快速验证开始 |
| **02:25:53** | **5分钟验证完成** ⚡ |
| 02:30:00 | 40分钟测试运行15分钟 |
| 02:45:00 | 40分钟测试运行30分钟 |
| **02:55:55** | **40分钟金测完成** 🏆 |

---

## ✅ 完成后的操作

### 当5分钟验证完成后（~02:26）

```powershell
# 切换到examples目录
cd C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\v13_ofi_ai_system\examples

# 分析结果
python analysis_cvd.py --data ..\..\data\cvd_quick_verify_20251019_0220\*.parquet --out ..\..\docs\reports\quick_verify_20251019 --report ..\..\docs\reports\quick_verify_20251019\REPORT.md

# 查看报告
notepad ..\..\docs\reports\quick_verify_20251019\REPORT.md
```

**检查要点**:
- [ ] 启动日志显示正确的Step 1.6配置
- [ ] queue_dropped_rate = 0%
- [ ] 无parse_errors
- [ ] Z-score指标合理

---

### 当40分钟金测完成后（~02:56）

```powershell
# 分析结果
python analysis_cvd.py --data ..\..\data\cvd_final_gold_20251019_0215\*.parquet --out ..\..\docs\reports\final_gold_20251019 --report ..\..\docs\reports\final_gold_20251019\REPORT.md

# 查看报告
notepad ..\..\docs\reports\final_gold_20251019\REPORT.md
```

**验收8/8标准**:
1. [ ] parse_errors = 0
2. [ ] queue_dropped_rate = 0%
3. [ ] p99_interarrival ≤ 5000ms
4. [ ] gaps_over_10s = 0
5. [ ] 逐笔守恒 = 0错误
6. [ ] 首尾守恒 < 相对容差
7. [ ] median|Z| ≤ 1.0
8. [ ] P(|Z|>2) ≤ 8%, P(|Z|>3) ≤ 2%

---

## 🆘 如果测试异常

### 检查进程是否还在运行
```powershell
Get-Process python -ErrorAction SilentlyContinue
```

### 查看输出日志
```powershell
# 5分钟验证
Get-ChildItem C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\data\cvd_quick_verify_20251019_0220

# 40分钟金测
Get-ChildItem C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\data\cvd_final_gold_20251019_0215
```

### 如果进程卡死
```powershell
# 找到进程ID
Get-Process python

# 强制停止（使用实际的PID）
Stop-Process -Id 1832 -Force
Stop-Process -Id 23960 -Force
```

---

## 📝 记录模板

测试完成后记录以下信息：

```
【5分钟快速验证】
- 配置加载: ✓/✗
- queue_dropped_rate: ___%
- 数据量: ___笔
- median|Z|: ___
- 问题: ___

【40分钟金测】
- 8/8验收: _/8通过
- queue_dropped_rate: ___%
- 数据量: ___笔
- P(|Z|>2): ___%
- P(|Z|>3): ___%
- median|Z|: ___
- 是否需要S7微调: ✓/✗
```

---

**现在你可以**:
1. ✅ 双击 `scripts/QUICK_CHECK.bat` 查看当前状态
2. ✅ 双击 `scripts/START_MONITOR.bat` 启动持续监控
3. ✅ 在Cursor中继续和我讨论其他问题

**测试会在后台独立运行，不会被打断！** 🚀

