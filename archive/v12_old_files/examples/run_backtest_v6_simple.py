#!/usr/bin/env python3
"""
v6 简化策略回测脚本
"""

import sys
import os
import yaml
import argparse
from datetime import datetime
import json

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import load_csv
from features import add_feature_block
from strategy_v6_simple import run_strategy_v6_simple

def main():
    parser = argparse.ArgumentParser(description='Run v6 simple strategy backtest')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    print(f"=== v6 简化策略回测 ===")
    print(f"配置文件: {args.config}")
    
    # 加载数据
    print("\n加载数据...")
    df = load_csv(params["data"]["path"])
    print(f"数据长度: {len(df)} 行")
    
    # 添加特征
    print("添加特征...")
    df = add_feature_block(df, params)
    
    # 运行策略
    print("运行策略...")
    trades = run_strategy_v6_simple(df, params)
    
    # 保存结果
    os.makedirs("examples/out", exist_ok=True)
    
    if not trades.empty:
        trades.to_csv("examples/out/trades_v6_simple.csv", index=False)
        print(f"交易记录已保存到: examples/out/trades_v6_simple.csv")
        
        # 计算详细统计
        total_trades = len(trades)
        winning_trades = len(trades[trades["pnl"] > 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        total_pnl = trades["pnl"].sum()
        total_fees = trades["fee"].sum()
        net_pnl = total_pnl - total_fees
        
        print(f"\n=== v6 简化策略回测结果 ===")
        print(f"总交易数: {total_trades}")
        print(f"盈利交易: {winning_trades}")
        print(f"胜率: {win_rate:.2f}%")
        print(f"总PnL: ${total_pnl:.2f}")
        print(f"总手续费: ${total_fees:.2f}")
        print(f"净PnL: ${net_pnl:.2f}")
        
        if total_trades > 0:
            avg_pnl = trades["pnl"].mean()
            avg_holding = trades["holding_sec"].mean()
            avg_signal_strength = trades["signal_strength"].mean()
            print(f"平均PnL: ${avg_pnl:.2f}")
            print(f"平均持仓时间: {avg_holding:.1f}秒")
            print(f"平均信号强度: {avg_signal_strength:.3f}")
            
            # 盈亏比分析
            winning_pnl = trades[trades["pnl"] > 0]["pnl"].sum()
            losing_pnl = abs(trades[trades["pnl"] < 0]["pnl"].sum())
            profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
            print(f"盈亏比: {profit_factor:.3f}")
            
            # 最大回撤
            cumulative_pnl = trades["pnl"].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_drawdown = drawdown.min()
            print(f"最大回撤: ${max_drawdown:.2f}")
        
        # 保存摘要
        summary = {
            "backtest_info": {
                "config_path": args.config,
                "strategy_version": "v6_simple",
                "run_timestamp": datetime.now().isoformat(),
                "data_path": params["data"]["path"]
            },
            "trade_metrics": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "win_rate": win_rate / 100,
                "avg_pnl": float(trades["pnl"].mean()) if total_trades > 0 else 0,
                "avg_holding_seconds": float(trades["holding_sec"].mean()) if total_trades > 0 else 0,
                "avg_signal_strength": float(trades["signal_strength"].mean()) if total_trades > 0 else 0
            },
            "performance_summary": {
                "total_pnl": total_pnl,
                "total_fees": total_fees,
                "net_pnl": net_pnl,
                "profit_factor": profit_factor if total_trades > 0 else 0,
                "max_drawdown": float(max_drawdown) if total_trades > 0 else 0
            }
        }
        
        with open("examples/out/summary_v6_simple.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"摘要已保存到: examples/out/summary_v6_simple.json")
        
    else:
        print("没有生成任何交易")
    
    print("\n回测完成!")

if __name__ == "__main__":
    main()
