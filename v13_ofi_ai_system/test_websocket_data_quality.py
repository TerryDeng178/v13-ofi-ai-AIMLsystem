#!/usr/bin/env python3
"""
WebSocket数据质量测试脚本
验证统一配置是否影响了WebSocket数据接收质量
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 确保项目根目录在sys.path中
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import websockets
except ImportError as e:
    print(f"Missing dependency: websockets - {e}")
    sys.exit(1)

# 测试配置
TEST_SYMBOL = "BTCUSDT"
TEST_DURATION = 60  # 1分钟测试
OUTPUT_DIR = Path("data/websocket_quality_test")

def parse_agg_trade(msg: str) -> dict:
    """解析aggTrade消息"""
    try:
        data = json.loads(msg)
        stream_data = data.get("data", {})
        
        return {
            "symbol": stream_data.get("s"),
            "price": float(stream_data.get("p", 0)),
            "qty": float(stream_data.get("q", 0)),
            "is_buy": stream_data.get("m", False) == False,  # m=False表示买入
            "event_time": int(stream_data.get("T", 0)),
            "trade_id": int(stream_data.get("a", 0))
        }
    except Exception as e:
        return None

async def test_websocket_quality():
    """测试WebSocket数据质量"""
    print("=" * 60)
    print("WebSocket数据质量测试")
    print("=" * 60)
    
    # 测试不同的WebSocket配置
    configs = [
        {
            "name": "原始配置（硬编码）",
            "url": f"wss://fstream.binancefuture.com/stream?streams={TEST_SYMBOL.lower()}@aggTrade",
            "ping_interval": None,
            "close_timeout": 5,
            "heartbeat_timeout": 60
        },
        {
            "name": "统一配置（system.yaml）",
            "url": f"wss://fstream.binancefuture.com/stream?streams={TEST_SYMBOL.lower()}@aggTrade",
            "ping_interval": 20,
            "close_timeout": 10,
            "heartbeat_timeout": 30
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n测试配置: {config['name']}")
        print("-" * 40)
        
        # 统计数据
        total_messages = 0
        valid_trades = 0
        buy_trades = 0
        sell_trades = 0
        price_sum = 0.0
        qty_sum = 0.0
        start_time = time.time()
        
        try:
            async with websockets.connect(
                config["url"],
                ping_interval=config["ping_interval"],
                close_timeout=config["close_timeout"]
            ) as ws:
                print(f"连接成功: {config['url']}")
                
                while time.time() - start_time < TEST_DURATION:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=config["heartbeat_timeout"])
                        total_messages += 1
                        
                        trade = parse_agg_trade(msg)
                        if trade and trade["symbol"] == TEST_SYMBOL:
                            valid_trades += 1
                            
                            if trade["is_buy"]:
                                buy_trades += 1
                            else:
                                sell_trades += 1
                            
                            price_sum += trade["price"]
                            qty_sum += trade["qty"]
                            
                            if valid_trades <= 5:  # 显示前5笔交易
                                print(f"  交易{valid_trades}: 价格={trade['price']:.2f}, 数量={trade['qty']:.4f}, 方向={'买入' if trade['is_buy'] else '卖出'}")
                        
                    except asyncio.TimeoutError:
                        print("  心跳超时，继续等待...")
                        continue
                        
        except Exception as e:
            print(f"连接错误: {e}")
            continue
        
        # 计算统计结果
        elapsed = time.time() - start_time
        rate = valid_trades / elapsed if elapsed > 0 else 0
        buy_ratio = buy_trades / valid_trades if valid_trades > 0 else 0
        avg_price = price_sum / valid_trades if valid_trades > 0 else 0
        avg_qty = qty_sum / valid_trades if valid_trades > 0 else 0
        
        result = {
            "config_name": config["name"],
            "total_messages": total_messages,
            "valid_trades": valid_trades,
            "rate_per_sec": rate,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "buy_ratio": buy_ratio,
            "avg_price": avg_price,
            "avg_qty": avg_qty,
            "elapsed_sec": elapsed
        }
        
        results.append(result)
        
        print(f"  总消息数: {total_messages}")
        print(f"  有效交易: {valid_trades}")
        print(f"  接收速率: {rate:.2f} 笔/秒")
        print(f"  买入比例: {buy_ratio:.1%}")
        print(f"  平均价格: {avg_price:.2f}")
        print(f"  平均数量: {avg_qty:.4f}")
    
    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = OUTPUT_DIR / f"websocket_quality_test_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_info": {
                "symbol": TEST_SYMBOL,
                "duration": TEST_DURATION,
                "timestamp": timestamp
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {result_file}")
    
    # 分析结果
    print("\n" + "=" * 60)
    print("结果分析")
    print("=" * 60)
    
    for result in results:
        print(f"\n{result['config_name']}:")
        print(f"  接收速率: {result['rate_per_sec']:.2f} 笔/秒")
        print(f"  买入比例: {result['buy_ratio']:.1%}")
        print(f"  数据质量: {'正常' if result['buy_ratio'] > 0.3 and result['buy_ratio'] < 0.7 else '异常'}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_websocket_quality())
