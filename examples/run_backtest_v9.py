import pandas as pd
import yaml
import os
import sys
import json
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import load_csv
from features import add_feature_block
from strategy_v9 import (
    run_strategy_v9_ml_enhanced, run_strategy_v9_advanced, 
    run_strategy_v9_multi_signal, run_strategy_v9_with_monitoring
)
from backtest import calculate_trade_metrics
from risk_v8 import calculate_profitability_metrics

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run v9 ML-enhanced strategy backtest.")
    parser.add_argument("--config", type=str, default="config/params_v9_ml_integration.yaml",
                        help="Path to the YAML configuration file.")
    parser.add_argument("--signal-type", type=str, default="ml_enhanced",
                        choices=["ml_enhanced", "real_time_optimized"],
                        help="Type of signal to use.")
    parser.add_argument("--strategy-mode", type=str, default="ultra_profitable",
                        choices=["ultra_profitable", "ml_optimized"],
                        help="Strategy mode.")
    parser.add_argument("--multi-signal", action="store_true",
                        help="Run multi-signal comparison.")
    parser.add_argument("--with-monitoring", action="store_true",
                        help="Run with real-time monitoring.")
    args = parser.parse_args()

    print(f"=== v9 机器学习集成回测 ===")
    print(f"加载文件: {args.config}")
    print(f"信号类型: {args.signal_type}")
    print(f"策略模式: {args.strategy_mode}")
    print(f"多信号模式: {args.multi_signal}")
    print(f"实时监控: {args.with_monitoring}")
    print()

    # Load parameters
    with open(args.config, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Load data
    print("加载数据...")
    df = load_csv(params["data"]["path"])
    print(f"数据长度: {len(df)} 条")

    # Add features
    print("生成特征...")
    df = add_feature_block(df, params)

    output_dir = "examples/out"
    os.makedirs(output_dir, exist_ok=True)

    if args.with_monitoring:
        print("运行带监控的策略...")
        results = run_strategy_v9_with_monitoring(df, params)
        
        trades = results['trades']
        performance_report = results['performance_report']
        dashboard_data = results['dashboard_data']
        optimization_suggestions = results['optimization_suggestions']
        performance_trend = results['performance_trend']
        
        # Save trades
        trades_path = os.path.join(output_dir, "trades_v9_monitored.csv")
        trades.to_csv(trades_path, index=False)
        print(f"交易记录已保存到: {trades_path}")
        
        # Save performance report
        report_path = os.path.join(output_dir, "v9_performance_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(performance_report, f, indent=4, ensure_ascii=False)
        print(f"性能报告已保存到: {report_path}")
        
        # Save dashboard data
        dashboard_path = os.path.join(output_dir, "v9_dashboard_data.json")
        with open(dashboard_path, "w", encoding="utf-8") as f:
            json.dump(dashboard_data, f, indent=4, ensure_ascii=False)
        print(f"仪表板数据已保存到: {dashboard_path}")
        
        # Print optimization suggestions
        print("\n=== 优化建议 ===")
        for suggestion in optimization_suggestions:
            print(f"- {suggestion}")
        
        # Print performance trend
        print(f"\n=== 性能趋势 ===")
        print(f"胜率趋势: {performance_trend.get('win_rate_trend', 'unknown')}")
        print(f"收益趋势: {performance_trend.get('pnl_trend', 'unknown')}")
        
    elif args.multi_signal:
        print("运行多信号对比...")
        results = run_strategy_v9_multi_signal(df, params)
        
        comparison_data = {}
        print("\n=== v9 多信号对比结果 ===")
        for name, trades in results.items():
            if not trades.empty:
                trade_metrics = calculate_trade_metrics(trades)
                profitability_metrics = calculate_profitability_metrics(trades)
                
                comparison_data[name] = {
                    "total_trades": trade_metrics.get("total_trades", 0),
                    "win_rate": trade_metrics.get("win_rate", 0.0),
                    "total_pnl": trades["pnl"].sum(),
                    "net_pnl": profitability_metrics.get("net_pnl", 0.0),
                    "cost_efficiency": profitability_metrics.get("cost_efficiency", 0.0),
                    "profitability_score": profitability_metrics.get("profitability_score", 0.0),
                    "profit_factor": profitability_metrics.get("profit_factor", 0.0),
                    "avg_signal_strength": trades["signal_strength"].mean() if "signal_strength" in trades.columns else 0.0,
                    "avg_quality_score": trades["quality_score"].mean() if "quality_score" in trades.columns else 0.0,
                    "avg_ml_prediction": trades["ml_prediction"].mean() if "ml_prediction" in trades.columns else 0.0,
                    "avg_ml_confidence": trades["ml_confidence"].mean() if "ml_confidence" in trades.columns else 0.0
                }
                print(f"  {name} 结果:")
                print(f"    交易数: {comparison_data[name]['total_trades']}")
                print(f"    胜率: {comparison_data[name]['win_rate']:.2%}")
                print(f"    净PnL: ${comparison_data[name]['net_pnl']:.2f}")
                print(f"    成本效率: {comparison_data[name]['cost_efficiency']:.2f}")
                print(f"    盈利能力评分: {comparison_data[name]['profitability_score']:.3f}")
                print(f"    盈利因子: {comparison_data[name]['profit_factor']:.3f}")
                print(f"    平均信号强度: {comparison_data[name]['avg_signal_strength']:.3f}")
                print(f"    平均质量评分: {comparison_data[name]['avg_quality_score']:.3f}")
                print(f"    平均ML预测: {comparison_data[name]['avg_ml_prediction']:.3f}")
                print(f"    平均ML置信度: {comparison_data[name]['avg_ml_confidence']:.3f}")
            else:
                comparison_data[name] = {
                    "total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "net_pnl": 0.0,
                    "cost_efficiency": 0.0, "profitability_score": 0.0, "profit_factor": 0.0,
                    "avg_signal_strength": 0.0, "avg_quality_score": 0.0, "avg_ml_prediction": 0.0, "avg_ml_confidence": 0.0
                }
                print(f"  {name} 结果: 无交易")

        comparison_path = os.path.join(output_dir, "v9_multi_signal_comparison.json")
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        print(f"\n多信号对比结果已保存到: {comparison_path}")
        
        # 找出最佳策略 (基于盈利能力评分)
        best_strategy_name = max(comparison_data, key=lambda k: comparison_data[k]['profitability_score'])
        best_strategy_data = comparison_data[best_strategy_name]
        print(f"\n最佳策略: {best_strategy_name}")
        print(f"   盈利能力评分: {best_strategy_data['profitability_score']:.3f}")
        print(f"   净PnL: ${best_strategy_data['net_pnl']:.2f}")
        print(f"   成本效率: {best_strategy_data['cost_efficiency']:.2f}")

    else:
        # Run single strategy
        print("运行策略...")
        trades = run_strategy_v9_ml_enhanced(df, params, args.signal_type, args.strategy_mode)

        # Save trades
        trades_path = os.path.join(output_dir, "trades_v9.csv")
        trades.to_csv(trades_path, index=False)
        print(f"交易记录已保存到: {trades_path}")

        # Calculate and save summary metrics
        summary = {
            "backtest_info": {
                "config_path": args.config,
                "signal_type": args.signal_type,
                "strategy_mode": args.strategy_mode,
                "run_timestamp": datetime.now().isoformat(),
                "data_path": params["data"]["path"],
                "initial_equity": params["backtest"]["initial_equity_usd"]
            }
        }

        if not trades.empty:
            trade_metrics = calculate_trade_metrics(trades)
            profitability_metrics = calculate_profitability_metrics(trades)
            
            summary["trade_metrics"] = trade_metrics
            summary["profitability_metrics"] = profitability_metrics
            summary["performance_summary"] = {
                "total_trades": trade_metrics.get("total_trades", 0),
                "win_rate": trade_metrics.get("win_rate", 0.0),
                "total_pnl": trades["pnl"].sum(),
                "net_pnl": profitability_metrics.get("net_pnl", 0.0),
                "avg_pnl_per_trade": trades["pnl"].mean(),
                "max_win": trades[trades["pnl"] > 0]["pnl"].max() if not trades[trades["pnl"] > 0].empty else 0.0,
                "max_loss": trades[trades["pnl"] < 0]["pnl"].min() if not trades[trades["pnl"] < 0].empty else 0.0,
                "avg_holding_seconds": trade_metrics.get("avg_holding_seconds", 0.0),
                "avg_signal_strength": trades["signal_strength"].mean() if "signal_strength" in trades.columns else 0.0,
                "avg_quality_score": trades["quality_score"].mean() if "quality_score" in trades.columns else 0.0,
                "avg_ml_prediction": trades["ml_prediction"].mean() if "ml_prediction" in trades.columns else 0.0,
                "avg_ml_confidence": trades["ml_confidence"].mean() if "ml_confidence" in trades.columns else 0.0,
                "cost_efficiency": profitability_metrics.get("cost_efficiency", 0.0),
                "profitability_score": profitability_metrics.get("profitability_score", 0.0),
                "profit_factor": profitability_metrics.get("profit_factor", 0.0)
            }
        else:
            summary["trade_metrics"] = {"total_trades": 0, "win_rate": 0.0}
            summary["profitability_metrics"] = {"net_pnl": 0.0, "cost_efficiency": 0.0, "profitability_score": 0.0}
            summary["performance_summary"] = {"total_trades": 0, "win_rate": 0.0, "net_pnl": 0.0}

        summary_path = os.path.join(output_dir, "summary_v9.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        print(f"\n=== v9 回测总结 ===")
        print(f"总交易数: {summary['trade_metrics'].get('total_trades', 0)}")
        print(f"盈利交易: {summary['trade_metrics'].get('winning_trades', 0)}")
        print(f"胜率: {summary['trade_metrics'].get('win_rate', 0.0):.2%}")
        print(f"总PnL: ${summary['performance_summary'].get('total_pnl', 0.0):.2f}")
        print(f"净PnL: ${summary['performance_summary'].get('net_pnl', 0.0):.2f}")
        print(f"平均PnL: ${summary['performance_summary'].get('avg_pnl_per_trade', 0.0):.2f}")
        print(f"平均持仓时间: {summary['performance_summary'].get('avg_holding_seconds', 0.0):.1f}秒")
        print(f"平均信号强度: {summary['performance_summary'].get('avg_signal_strength', 0.0):.3f}")
        print(f"平均质量评分: {summary['performance_summary'].get('avg_quality_score', 0.0):.3f}")
        print(f"平均ML预测: {summary['performance_summary'].get('avg_ml_prediction', 0.0):.3f}")
        print(f"平均ML置信度: {summary['performance_summary'].get('avg_ml_confidence', 0.0):.3f}")
        print(f"成本效率: {summary['performance_summary'].get('cost_efficiency', 0.0):.2f}")
        print(f"盈利能力评分: {summary['performance_summary'].get('profitability_score', 0.0):.3f}")
        print(f"盈利因子: {summary['performance_summary'].get('profit_factor', 0.0):.3f}")
        print(f"总结已保存到: {summary_path}")
        print("回测完成!")

if __name__ == "__main__":
    main()
