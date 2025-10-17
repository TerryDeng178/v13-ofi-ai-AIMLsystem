# -*- coding: utf-8 -*-
"""
Task 1.1.6 - 30分钟完整验证测试
严格按照验收标准执行
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "v13_ofi_ai_system" / "src"))

from binance_websocket_client import BinanceOrderBookStream

def main():
    print("="*80)
    print("Task 1.1.6 - 30分钟完整验证测试")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("测试时长: 30分钟")
    print("打印间隔: 10秒")
    print("日志轮转: 每60秒")
    print("="*80)
    
    # 创建客户端
    client = BinanceOrderBookStream(
        symbol="ETHUSDT",
        depth_levels=5,
        rotate="interval",       # 时间轮转
        rotate_sec=60,           # 每60秒轮转一次
        max_bytes=5_000_000,
        backups=7,
        print_interval=10,       # 每10秒打印SUMMARY
        base_dir=Path(__file__).parent / "v13_ofi_ai_system"
    )
    
    # 在后台线程运行
    import threading
    t = threading.Thread(target=client.run, kwargs={"reconnect": True}, daemon=True)
    t.start()
    
    # 等待30分钟 = 1800秒
    test_duration_seconds = 30 * 60
    print(f"\n正在运行 {test_duration_seconds} 秒 (30分钟)...")
    print("请勿关闭此窗口！\n")
    
    # 每分钟打印进度
    start_time = time.time()
    while (time.time() - start_time) < test_duration_seconds:
        elapsed = time.time() - start_time
        remaining = test_duration_seconds - elapsed
        print(f"\r进度: {elapsed/60:.1f} / 30.0 分钟 (剩余 {remaining/60:.1f} 分钟)", end="", flush=True)
        time.sleep(5)  # 每5秒更新一次进度
    
    print(f"\n\n测试完成！结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 停止WebSocket
    try:
        if client.ws:
            client.ws.close()
        client.listener.stop()
    except Exception as e:
        print(f"停止客户端时出错: {e}")
    
    # 等待清理
    time.sleep(2)
    
    # 收集验收产物
    print("\n收集验收产物...")
    print("-"*80)
    
    data_dir = client.data_dir
    log_dir = client.log_dir
    
    # 1. metrics.json
    metrics_file = data_dir / "metrics.json"
    if metrics_file.exists():
        print(f"✅ metrics.json: {metrics_file}")
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"   - 运行时长: {metrics.get('runtime_seconds', 0):.1f}秒")
        print(f"   - 总消息数: {metrics.get('total_messages', 0)}")
        print(f"   - 接收速率: {metrics.get('recv_rate', 0):.2f}/s")
    else:
        print(f"❌ metrics.json 不存在")
    
    # 2. NDJSON数据文件
    ndjson_file = client.ndjson_path
    if ndjson_file.exists():
        size_mb = ndjson_file.stat().st_size / (1024*1024)
        print(f"✅ NDJSON数据: {ndjson_file}")
        print(f"   - 文件大小: {size_mb:.2f} MB")
    else:
        print(f"❌ NDJSON文件不存在")
    
    # 3. 日志文件（含轮转切片）
    log_files = list(log_dir.glob("ethusdt_*.log*"))
    print(f"✅ 日志文件数量: {len(log_files)}")
    for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime):
        size_kb = log_file.stat().st_size / 1024
        print(f"   - {log_file.name}: {size_kb:.1f} KB")
    
    # 4. 最后两条SUMMARY
    print("\n最后两条SUMMARY:")
    print("-"*80)
    for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            summary_lines = [l.strip() for l in lines if "SUMMARY |" in l]
            if summary_lines:
                for line in summary_lines[-2:]:
                    print(line)
                break
        except Exception as e:
            continue
    
    print("\n"+"="*80)
    print("验收产物收集完成！")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

