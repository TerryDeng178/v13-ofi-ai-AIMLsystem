#!/usr/bin/env python3
"""
v5 策略回测脚本
支持不同的信号逻辑测试
"""

import sys
import os
import yaml
import argparse
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import load_csv
from features import add_feature_block
from strategy_v5 import run_strategy_v5, run_strategy_v5_advanced
from backtest import run as run_backtest

def main():
    parser = argparse.ArgumentParser(description='Run v5 strategy backtest')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--signal-type', type=str, default='ultra_simple', 
                       choices=['ultra_simple', 'reversal', 'momentum_breakout'],
                       help='Signal type to test')
    parser.add_argument('--advanced', action='store_true', help='Use advanced strategy')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    print(f"=== v5 策略回测 ===")
    print(f"配置文件: {args.config}")
    print(f"信号类型: {args.signal_type}")
    print(f"策略模式: {'高级' if args.advanced else '简化'}")
    
    # 加载数据
    print("\n加载数据...")
    df = load_csv(params["data"]["path"])
    print(f"数据长度: {len(df)} 行")
    
    # 添加特征
    print("添加特征...")
    df = add_feature_block(df, params)
    
    # 运行策略
    print("运行策略...")
    if args.advanced:
        trades = run_strategy_v5_advanced(df, params, args.signal_type)
    else:
        trades = run_strategy_v5(df, params, args.signal_type)
    
    # 保存结果
    os.makedirs("examples/out", exist_ok=True)
    
    if not trades.empty:
        trades.to_csv("examples/out/trades_v5.csv", index=False)
        print(f"交易记录已保存到: examples/out/trades_v5.csv")
        
        # 计算基本统计
        total_trades = len(trades)
        winning_trades = len(trades[trades["pnl"] > 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        total_pnl = trades["pnl"].sum()
        total_fees = trades["fee"].sum()
        net_pnl = total_pnl - total_fees
        
        print(f"\n=== 回测结果 ===")
        print(f"总交易数: {total_trades}")
        print(f"盈利交易: {winning_trades}")
        print(f"胜率: {win_rate:.2f}%")
        print(f"总PnL: ${total_pnl:.2f}")
        print(f"总手续费: ${total_fees:.2f}")
        print(f"净PnL: ${net_pnl:.2f}")
        
        if total_trades > 0:
            avg_pnl = trades["pnl"].mean()
            avg_holding = trades["holding_sec"].mean()
            print(f"平均PnL: ${avg_pnl:.2f}")
            print(f"平均持仓时间: {avg_holding:.1f}秒")
        
        # 保存摘要
        summary = {
            "backtest_info": {
                "config_path": args.config,
                "signal_type": args.signal_type,
                "strategy_mode": "advanced" if args.advanced else "simple",
                "run_timestamp": datetime.now().isoformat(),
                "data_path": params["data"]["path"]
            },
            "trade_metrics": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "win_rate": win_rate,
                "avg_pnl": trades["pnl"].mean() if total_trades > 0 else 0,
                "avg_holding_seconds": trades["holding_sec"].mean() if total_trades > 0 else 0
            },
            "performance_summary": {
                "total_pnl": total_pnl,
                "total_fees": total_fees,
                "net_pnl": net_pnl
            }
        }
        
        import json
        with open("examples/out/summary_v5.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"摘要已保存到: examples/out/summary_v5.json")
        
    else:
        print("没有生成任何交易")
    
    print("\n回测完成!")

if __name__ == "__main__":
    main()
