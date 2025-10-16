#!/usr/bin/env python3
"""
v7 策略回测脚本
支持信号质量筛选和动态仓位管理
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
from strategy_v7 import run_strategy_v7, run_strategy_v7_advanced, run_strategy_v7_multi_signal
from risk_v7 import calculate_advanced_risk_metrics

def main():
    parser = argparse.ArgumentParser(description='Run v7 strategy backtest')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--signal-type', type=str, default='quality_filtered', 
                       choices=['quality_filtered', 'dynamic_threshold', 'momentum_enhanced'],
                       help='Signal type to test')
    parser.add_argument('--strategy-mode', type=str, default='ultra_optimized',
                       choices=['ultra_optimized', 'risk_adjusted'],
                       help='Strategy mode')
    parser.add_argument('--multi-signal', action='store_true', help='Run multi-signal comparison')
    parser.add_argument('--advanced', action='store_true', help='Use advanced strategy')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    print(f"=== v7 策略回测 ===")
    print(f"配置文件: {args.config}")
    print(f"信号类型: {args.signal_type}")
    print(f"策略模式: {args.strategy_mode}")
    print(f"多信号模式: {'是' if args.multi_signal else '否'}")
    
    # 加载数据
    print("\n加载数据...")
    df = load_csv(params["data"]["path"])
    print(f"数据长度: {len(df)} 行")
    
    # 添加特征
    print("添加特征...")
    df = add_feature_block(df, params)
    
    # 运行策略
    print("运行策略...")
    
    if args.multi_signal:
        # 多信号对比模式
        results = run_strategy_v7_multi_signal(df, params)
        
        # 保存所有结果
        os.makedirs("examples/out", exist_ok=True)
        
        print(f"\n=== 多信号对比结果 ===")
        comparison_data = {}
        
        for strategy_key, trades in results.items():
            if not trades.empty:
                risk_metrics = calculate_advanced_risk_metrics(trades)
                
                print(f"\n{strategy_key} 策略:")
                print(f"  交易数: {len(trades)}")
                print(f"  胜率: {risk_metrics.get('win_rate', 0):.2%}")
                print(f"  总PnL: ${trades['pnl'].sum():.2f}")
                print(f"  净PnL: ${trades['pnl'].sum() - trades['fee'].sum():.2f}")
                print(f"  夏普比率: {risk_metrics.get('sharpe_ratio', 0):.3f}")
                print(f"  最大回撤: ${risk_metrics.get('max_drawdown', 0):.2f}")
                print(f"  平均信号强度: {risk_metrics.get('avg_signal_strength', 0):.3f}")
                print(f"  平均质量评分: {risk_metrics.get('avg_quality_score', 0):.3f}")
                
                # 保存单个策略结果
                trades.to_csv(f"examples/out/trades_v7_{strategy_key}.csv", index=False)
                
                comparison_data[strategy_key] = {
                    "trade_count": len(trades),
                    "win_rate": risk_metrics.get('win_rate', 0),
                    "total_pnl": float(trades['pnl'].sum()),
                    "net_pnl": float(trades['pnl'].sum() - trades['fee'].sum()),
                    "sharpe_ratio": risk_metrics.get('sharpe_ratio', 0),
                    "max_drawdown": risk_metrics.get('max_drawdown', 0),
                    "profit_factor": risk_metrics.get('profit_factor', 0),
                    "avg_signal_strength": risk_metrics.get('avg_signal_strength', 0),
                    "avg_quality_score": risk_metrics.get('avg_quality_score', 0),
                    "max_consecutive_losses": risk_metrics.get('max_consecutive_losses', 0)
                }
            else:
                print(f"\n{strategy_key} 策略: 无交易")
                comparison_data[strategy_key] = {
                    "trade_count": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "net_pnl": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "profit_factor": 0,
                    "avg_signal_strength": 0,
                    "avg_quality_score": 0,
                    "max_consecutive_losses": 0
                }
        
        # 保存对比结果
        with open("examples/out/v7_multi_signal_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n多信号对比结果已保存到: examples/out/v7_multi_signal_comparison.json")
        
        # 找出最佳策略
        best_strategy = max(comparison_data.items(), key=lambda x: x[1]['net_pnl'])
        print(f"\n最佳策略: {best_strategy[0]}")
        print(f"   净PnL: ${best_strategy[1]['net_pnl']:.2f}")
        print(f"   胜率: {best_strategy[1]['win_rate']:.2%}")
        print(f"   平均信号强度: {best_strategy[1]['avg_signal_strength']:.3f}")
        
    else:
        # 单策略模式
        if args.advanced:
            trades = run_strategy_v7_advanced(df, params, args.signal_type)
        else:
            trades = run_strategy_v7(df, params, args.signal_type, args.strategy_mode)
        
        # 保存结果
        os.makedirs("examples/out", exist_ok=True)
        
        if not trades.empty:
            trades.to_csv("examples/out/trades_v7.csv", index=False)
            print(f"交易记录已保存到: examples/out/trades_v7.csv")
            
            # 计算详细统计
            risk_metrics = calculate_advanced_risk_metrics(trades)
            
            total_trades = len(trades)
            winning_trades = len(trades[trades["pnl"] > 0])
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            total_pnl = trades["pnl"].sum()
            total_fees = trades["fee"].sum()
            net_pnl = total_pnl - total_fees
            
            print(f"\n=== v7 回测结果 ===")
            print(f"总交易数: {total_trades}")
            print(f"盈利交易: {winning_trades}")
            print(f"胜率: {win_rate:.2f}%")
            print(f"总PnL: ${total_pnl:.2f}")
            print(f"总手续费: ${total_fees:.2f}")
            print(f"净PnL: ${net_pnl:.2f}")
            print(f"夏普比率: {risk_metrics.get('sharpe_ratio', 0):.3f}")
            print(f"最大回撤: ${risk_metrics.get('max_drawdown', 0):.2f}")
            print(f"盈亏比: {risk_metrics.get('profit_factor', 0):.3f}")
            print(f"最大连续亏损: {risk_metrics.get('max_consecutive_losses', 0)}")
            
            if total_trades > 0:
                avg_pnl = trades["pnl"].mean()
                avg_holding = trades["holding_sec"].mean()
                avg_signal_strength = trades["signal_strength"].mean()
                avg_quality_score = trades["quality_score"].mean() if "quality_score" in trades.columns else 0
                print(f"平均PnL: ${avg_pnl:.2f}")
                print(f"平均持仓时间: {avg_holding:.1f}秒")
                print(f"平均信号强度: {avg_signal_strength:.3f}")
                print(f"平均质量评分: {avg_quality_score:.3f}")
                
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
                    "signal_type": args.signal_type,
                    "strategy_mode": args.strategy_mode,
                    "strategy_version": "advanced" if args.advanced else "standard",
                    "run_timestamp": datetime.now().isoformat(),
                    "data_path": params["data"]["path"]
                },
                "trade_metrics": risk_metrics,
                "performance_summary": {
                    "total_pnl": total_pnl,
                    "total_fees": total_fees,
                    "net_pnl": net_pnl,
                    "avg_pnl_per_trade": trades["pnl"].mean() if total_trades > 0 else 0,
                    "avg_holding_seconds": trades["holding_sec"].mean() if total_trades > 0 else 0,
                    "avg_signal_strength": trades["signal_strength"].mean() if total_trades > 0 else 0,
                    "avg_quality_score": trades["quality_score"].mean() if "quality_score" in trades.columns and total_trades > 0 else 0
                }
            }
            
            with open("examples/out/summary_v7.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"摘要已保存到: examples/out/summary_v7.json")
            
        else:
            print("没有生成任何交易")
    
    print("\n回测完成!")

if __name__ == "__main__":
    main()
