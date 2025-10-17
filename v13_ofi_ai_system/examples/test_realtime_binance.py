# -*- coding: utf-8 -*-
"""
真实Binance WebSocket环境验证测试
基于 BINANCE_WEBSOCKET_CLIENT_USAGE.md 和 README_realtime_ofi.md 规范
"""
import sys
import io
import os
from pathlib import Path

# Windows UTF-8 输出兼容
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目路径
THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))

try:
    from real_ofi_calculator import RealOFICalculator, OFIConfig
    print("[OK] 成功导入 RealOFICalculator")
except Exception as e:
    print(f"[ERROR] 导入失败: {e}")
    sys.exit(1)

# 设置环境变量（按照 README 规范）
os.environ['WS_URL'] = 'wss://fstream.binancefuture.com/stream?streams=ethusdt@depth@100ms'
os.environ['SYMBOL'] = 'ETHUSDT'
os.environ['K_LEVELS'] = '5'
os.environ['Z_WINDOW'] = '300'
os.environ['EMA_ALPHA'] = '0.2'
os.environ['LOG_LEVEL'] = 'INFO'  # 设置为INFO级别，减少日志

print("=" * 60)
print("[TEST] 真实Binance WebSocket环境验证测试")
print("=" * 60)
print(f"[CONFIG] WebSocket URL: {os.environ['WS_URL']}")
print(f"[CONFIG] 交易对: {os.environ['SYMBOL']}")
print(f"[CONFIG] 订单簿深度: {os.environ['K_LEVELS']}档")
print(f"[CONFIG] 测试时长: 3分钟")
print("=" * 60)
print()

# 导入并运行 run_realtime_ofi
import asyncio
import importlib.util

# 动态加载 run_realtime_ofi.py
spec = importlib.util.spec_from_file_location("run_realtime_ofi", THIS_DIR / "run_realtime_ofi.py")
run_realtime_ofi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_realtime_ofi)

# 运行3分钟测试
print("[START] 开始测试（3分钟）...")
print("[EXPECT] 预期观测:")
print("   - WebSocket连接成功")
print("   - 持续接收订单簿数据")
print("   - OFI值正常计算（非全0）")
print("   - 每60秒输出性能统计")
print("   - 限流机制在网络抖动时触发")
print()

async def run_test():
    """运行3分钟测试"""
    try:
        # 创建main任务
        main_task = asyncio.create_task(run_realtime_ofi.main(demo=False))
        
        # 等待3分钟
        await asyncio.sleep(180)
        
        # 停止测试
        main_task.cancel()
        try:
            await main_task
        except asyncio.CancelledError:
            pass
        
        print()
        print("=" * 60)
        print("[DONE] 测试完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 用户中断测试")
    except Exception as e:
        print(f"\n[ERROR] 测试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 测试被中断")

