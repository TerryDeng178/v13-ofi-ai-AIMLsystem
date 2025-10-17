# Task 1.1.6 文件锁定问题解决方案

## 🐛 问题

测试过程中遇到文件锁定错误：
```
另一个程序正在使用此文件，进程无法访问
```

## 🔍 原因分析

可能的原因：
1. ❌ Python进程没有完全退出
2. ❌ gzip.open() 文件句柄没有正确关闭
3. ❌ 异步日志系统持有文件句柄
4. ❌ Windows文件锁定机制

## ✅ 解决方案

### 方案A: 清理所有Python进程（已执行）

```powershell
Get-Process python | Stop-Process -Force
```

### 方案B: 删除被锁定的NDJSON文件

如果方案A无效，删除旧的NDJSON文件：

```powershell
cd C:\Users\user\Desktop\ofi_cvd_framework\ofi_cvd_framework

# 删除可能被锁定的文件
Remove-Item v13_ofi_ai_system\data\order_book\ethusdt_depth.ndjson.gz -Force -ErrorAction SilentlyContinue

# 确认删除
dir v13_ofi_ai_system\data\order_book\*.ndjson.gz
```

### 方案C: 使用新的测试脚本（推荐）

创建一个简化的测试脚本，使用普通文件而不是gzip：

```python
# test_simple.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "v13_ofi_ai_system" / "src"))

from binance_websocket_client import BinanceOrderBookStream

# 临时修改NDJSON路径，避免冲突
client = BinanceOrderBookStream(
    symbol="ETHUSDT",
    depth_levels=5,
    print_interval=5  # 5秒一次
)

# 运行2分钟测试
import threading, time
t = threading.Thread(target=client.run, kwargs={"reconnect": True}, daemon=True)
t.start()

print("测试运行中... 按Ctrl+C停止")
try:
    time.sleep(120)  # 2分钟
except KeyboardInterrupt:
    pass

if client.ws:
    try: client.ws.close()
    except: pass
try: client.listener.stop()
except: pass

print("\n测试完成！")
```

### 方案D: 重启终端

如果以上方案都无效：
1. 关闭当前PowerShell终端
2. 打开新的PowerShell终端
3. 重新运行测试

---

## 🚀 推荐执行顺序

1. ✅ **方案B** - 删除被锁定文件（快速）
2. ✅ **重新运行测试** - 使用简化命令
3. ⚠️ 如果仍失败 → **方案D** - 重启终端

---

## 📝 简化测试命令

```powershell
# 删除旧文件
Remove-Item v13_ofi_ai_system\data\order_book\ethusdt_depth.ndjson.gz -Force -ErrorAction SilentlyContinue

# 进入目录
cd v13_ofi_ai_system\src

# 运行2分钟测试
python binance_websocket_client.py --symbol ETHUSDT --print-interval 5 --run-minutes 2
```

**预期输出**：
- 每5秒一次SUMMARY
- 约24次输出（2分钟）
- 自动停止

---

## ✅ 验收标准（2分钟测试）

| 指标 | 目标 | 说明 |
|------|------|------|
| SUMMARY输出 | ≥ 10次 | 证明程序正常运行 |
| msgs | ≥ 50 | 收到足够数据 |
| rate | ≥ 1.0/s | 接收速率正常 |
| resyncs | 0 | 无重新同步 |
| reconnects | ≤ 1 | 最多1次重连 |

---

**当前时间**: 2025-10-17 07:35:00  
**状态**: 等待用户执行方案B并重新测试

