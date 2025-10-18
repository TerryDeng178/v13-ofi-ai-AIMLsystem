# 🪙 BTCUSDT 40分钟金测执行指南

**测试时间**: 2025-10-19 凌晨  
**预计时长**: 40分钟  
**测试目的**: 验证BTCUSDT是否在凌晨也有足够交易量

---

## 🚀 启动测试（3种方式）

### 方式1：双击批处理文件（推荐）
**位置**: `v13_ofi_ai_system\scripts\START_BTCUSDT_TEST.bat`

**操作**:
1. 在文件资源管理器中找到这个文件
2. 双击运行
3. 会打开新窗口显示测试进度
4. **不要关闭窗口，让它运行40分钟**

---

### 方式2：PowerShell命令
```powershell
cd C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\v13_ofi_ai_system\examples

python run_realtime_cvd.py `
  --symbol BTCUSDT `
  --duration 2400 `
  --output-dir ..\..\data\cvd_gold_btc_$(Get-Date -Format 'yyyyMMdd_HHmm')
```

---

### 方式3：让我在后台启动
告诉我"启动BTCUSDT测试"，我会帮你在后台运行。

---

## 📊 BTCUSDT vs ETHUSDT 对比

| 特点 | ETHUSDT | BTCUSDT |
|------|---------|---------|
| **市值** | 中型 | 最大 |
| **交易量** | 较大 | 最大 |
| **凌晨活跃度** | 低 | 中等 |
| **数据稳定性** | 好 | 极好 |
| **预期采集量** | 1000-5000笔 | 3000-10000笔 |

**ETHUSDT凌晨测试结果**:
- 40分钟采集: 999笔
- 采集速率: 0.42笔/秒
- 结论: 数据量不足

**BTCUSDT预期**:
- 40分钟预期: 3000-8000笔
- 预期速率: 1.2-3.3笔/秒
- 置信度: 中等（比ETHUSDT好，但仍是凌晨）

---

## ⏰ 时间线

| 时间 | 事件 |
|------|------|
| 现在 03:00 | 开始BTCUSDT测试 |
| 03:20 | 测试运行20分钟（一半） |
| **03:40** | **测试完成** |
| 03:45 | 分析结果 |
| 03:50 | 生成报告 |

---

## ✅ 测试完成后的操作

### 自动分析（在同一个窗口）
如果使用批处理文件启动，测试会自动进行分析（需要修改脚本添加此功能）。

### 手动分析
```powershell
cd C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\v13_ofi_ai_system\examples

# 找到实际的时间戳替换
$timestamp = "20251019_0300"

python analysis_cvd.py `
  --data ..\..\data\cvd_gold_btc_$timestamp\*.parquet `
  --out ..\..\docs\reports\cvd_gold_btc_$timestamp `
  --report ..\..\docs\reports\cvd_gold_btc_$timestamp\REPORT.md
```

---

## 🎯 预期结果

### 如果数据量充足（≥3000笔）
- ✅ 可以进行初步验收
- ✅ Z-score统计更可靠
- ✅ 可能达到6-7/8验收标准

### 如果数据量仍不足（<2000笔）
- ⚠️ 说明凌晨时段不适合测试
- ⚠️ 必须等待交易活跃时段（14:00-22:00）
- ✅ 但工程质量仍可验证

---

## 📝 监控方式

### 快速检查
双击运行: `v13_ofi_ai_system\scripts\QUICK_CHECK.bat`

### 实时查看进程
```powershell
Get-Process python | Select Id, StartTime, CPU, WorkingSet
```

### 查看输出目录
```powershell
Get-ChildItem C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework\data -Directory | Where {$_.Name -like 'cvd_gold_btc_*'} | Sort LastWriteTime -Desc
```

---

## 🔍 关键观察点

测试过程中关注：
1. **采集速率**: 是否明显高于ETHUSDT的0.42笔/秒
2. **数据量**: 40分钟是否能达到3000+笔
3. **queue_dropped_rate**: 应该保持0%
4. **连接稳定性**: 无重连

---

## 💡 提示

- ✅ BTCUSDT是全球交易量最大的加密货币对
- ✅ 即使凌晨，也比ETHUSDT活跃
- ⚠️ 但凌晨仍然不是最佳测试时段
- 🎯 如果BTCUSDT凌晨也数据不足，说明必须等活跃时段

---

## 📞 需要帮助？

测试期间有任何问题，随时在Cursor中告诉我：
- "检查BTCUSDT测试状态"
- "BTCUSDT测试出问题了"
- "停止BTCUSDT测试"
- "BTCUSDT测试完成了吗"

---

**祝测试顺利！** 🚀

希望BTCUSDT能带来更好的数据量！

