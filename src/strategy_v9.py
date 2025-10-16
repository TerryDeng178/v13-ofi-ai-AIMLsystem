import numpy as np
import pandas as pd
from typing import Dict
try:
    from .risk_v8 import (
        compute_ultra_profitable_levels, compute_profit_optimized_position_sizing,
        calculate_profitability_metrics, check_profitability_limits
    )
    from .exec import SimBroker
    from .signals_v9_ml import (
        gen_signals_v9_ml_enhanced, gen_signals_v9_real_time_optimized
    )
    from .monitoring_v9 import RealTimeMonitor, PerformanceTracker
except ImportError:
    # For testing when running as standalone
    from risk_v8 import (
        compute_ultra_profitable_levels, compute_profit_optimized_position_sizing,
        calculate_profitability_metrics, check_profitability_limits
    )
    from exec import SimBroker
    from signals_v9_ml import (
        gen_signals_v9_ml_enhanced, gen_signals_v9_real_time_optimized
    )
    from monitoring_v9 import RealTimeMonitor, PerformanceTracker

def run_strategy_v9_ml_enhanced(df: pd.DataFrame, params: dict, signal_type: str = "ml_enhanced", 
                               strategy_mode: str = "ultra_profitable") -> pd.DataFrame:
    """
    v9 机器学习增强策略运行器
    """
    broker = SimBroker(fee_bps=params["risk"]["fee_bps"], slip_bps_budget_frac=params["risk"]["slip_bps_budget_frac"])
    trades = []
    equity = params["backtest"]["initial_equity_usd"]
    open_pos = None
    
    # 初始化实时监控
    monitor = RealTimeMonitor(params.get("monitoring", {}))
    performance_tracker = PerformanceTracker()
    
    # 启动监控
    monitor.start_monitoring()
    
    # 选择信号生成函数
    signal_func_map = {
        "ml_enhanced": gen_signals_v9_ml_enhanced,
        "real_time_optimized": gen_signals_v9_real_time_optimized
    }
    
    signal_func = signal_func_map.get(signal_type, gen_signals_v9_ml_enhanced)
    
    # 生成信号
    df_with_signals = signal_func(df, params)
    
    # 预计算median_depth
    median_depth = df_with_signals["bid1_size"].rolling(60, min_periods=30).median()
    
    # 性能跟踪
    recent_performance = {"win_rate": 0.5, "avg_pnl": 0.0, "profit_factor": 1.0}
    daily_pnl = 0.0
    consecutive_losses = 0
    
    for i, row in df_with_signals.iterrows():
        # 流动性前置检查
        spread_bps = (row["ask1"] - row["bid1"]) / row["price"] * 1e4
        depth_now = row["bid1_size"] + row["ask1_size"]
        depth_med = median_depth.iloc[i] if i < len(median_depth) else depth_now
        thin_book_spread_max = params["signals"]["ml_enhanced"].get("thin_book_spread_bps_max", 5.0)
        
        if not (spread_bps <= thin_book_spread_max and depth_now >= depth_med):
            continue
        
        # 检查信号
        if pd.isna(row["sig_side"]) or row["sig_side"] == 0:
            continue
        
        side = int(row["sig_side"])
        signal_strength = row.get("signal_strength", 1.0)
        quality_score = row.get("quality_score", 1.0)
        ml_prediction = row.get("ml_prediction", 1.0)
        ml_confidence = row.get("ml_confidence", 1.0)
        
        # 开仓逻辑
        if open_pos is None:
            # 智能仓位管理
            if strategy_mode == "ultra_profitable":
                _, qty_usd = compute_profit_optimized_position_sizing(
                    row, params, signal_strength, quality_score, ml_confidence, recent_performance)
            else:
                # 默认模式
                from risk import position_size
                _, qty_usd = position_size(row, params)
            
            if qty_usd <= 0:
                continue
            
            # 智能止盈止损计算
            if strategy_mode == "ultra_profitable":
                sl, tp = compute_ultra_profitable_levels(row, params, signal_strength, quality_score, ml_confidence)
            else:
                # 默认模式
                from risk import compute_levels
                sl, tp = compute_levels(row, params)
            
            # 模拟执行
            entry_price = row["price"]
            exit_price = row["price"] * (1 + 0.0001 * side)  # 简化滑点模拟
            fee = qty_usd * params["risk"]["fee_bps"] / 10000
            
            # 计算PnL
            pnl = qty_usd * (exit_price - entry_price) / entry_price * side - fee
            
            # 盈利能力检查
            if check_profitability_limits(equity, params, pnl, daily_pnl, consecutive_losses):
                # 记录交易
                trade = {
                    "entry_ts": row["ts"],
                    "exit_ts": row["ts"] + pd.Timedelta(seconds=30),  # 固定30秒持仓
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "qty_usd": qty_usd,
                    "pnl": pnl,
                    "fee": fee,
                    "slippage": 0.0,  # 简化处理
                    "holding_sec": 30,
                    "signal_strength": signal_strength,
                    "quality_score": quality_score,
                    "ml_prediction": ml_prediction,
                    "ml_confidence": ml_confidence,
                    "sl": sl,
                    "tp": tp,
                    "strategy_mode": strategy_mode
                }
                trades.append(trade)
                
                # 添加到监控系统
                monitor.add_trade(trade)
                
                # 更新权益和性能跟踪
                equity += pnl
                daily_pnl += pnl
                
                # 更新连续亏损计数
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                
                # 更新性能跟踪
                if len(trades) >= 10:
                    recent_trades = pd.DataFrame(trades[-10:])
                    recent_performance = {
                        "win_rate": len(recent_trades[recent_trades["pnl"] > 0]) / len(recent_trades),
                        "avg_pnl": recent_trades["pnl"].mean(),
                        "profit_factor": calculate_profit_factor(recent_trades)
                    }
                    
                    # 添加到性能跟踪器
                    performance_tracker.add_performance_data(recent_performance)
    
    # 停止监控
    monitor.stop_monitoring()
    
    return pd.DataFrame(trades) if trades else pd.DataFrame()

def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
    """
    计算盈利因子
    """
    if trades_df.empty:
        return 0.0
    
    winning_pnl = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
    losing_pnl = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
    
    return winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')

def run_strategy_v9_advanced(df: pd.DataFrame, params: dict, signal_type: str = "ml_enhanced") -> pd.DataFrame:
    """
    v9 高级策略运行器 - 包含完整的ML集成和实时监控
    """
    trades = run_strategy_v9_ml_enhanced(df, params, signal_type, "ultra_profitable")
    
    # 计算盈利能力指标
    profitability_metrics = calculate_profitability_metrics(trades)
    
    # 保存盈利能力指标到交易记录
    if not trades.empty and profitability_metrics:
        trades["win_rate"] = profitability_metrics.get("win_rate", 0.0)
        trades["profit_factor"] = profitability_metrics.get("profit_factor", 0.0)
        trades["cost_efficiency"] = profitability_metrics.get("cost_efficiency", 0.0)
        trades["profitability_score"] = profitability_metrics.get("profitability_score", 0.0)
    
    return trades

def run_strategy_v9_multi_signal(df: pd.DataFrame, params: dict) -> Dict[str, pd.DataFrame]:
    """
    v9 多信号策略运行器 - 同时运行多种ML增强信号逻辑
    """
    signal_types = ["ml_enhanced", "real_time_optimized"]
    strategy_modes = ["ultra_profitable", "ml_optimized"]
    results = {}
    
    for signal_type in signal_types:
        for strategy_mode in strategy_modes:
            key = f"{signal_type}_{strategy_mode}"
            print(f"运行 {key} 策略...")
            trades = run_strategy_v9_ml_enhanced(df, params, signal_type, strategy_mode)
            results[key] = trades
            
            if not trades.empty:
                profitability_metrics = calculate_profitability_metrics(trades)
                print(f"  {key}: 交易数={len(trades)}, 胜率={profitability_metrics.get('win_rate', 0):.2%}, "
                      f"净PnL=${profitability_metrics.get('net_pnl', 0):.2f}, "
                      f"成本效率={profitability_metrics.get('cost_efficiency', 0):.2f}, "
                      f"盈利能力评分={profitability_metrics.get('profitability_score', 0):.3f}")
                
                # 输出ML相关指标
                if "ml_prediction" in trades.columns:
                    avg_ml_prediction = trades["ml_prediction"].mean()
                    avg_ml_confidence = trades["ml_confidence"].mean()
                    print(f"    平均ML预测: {avg_ml_prediction:.3f}, 平均ML置信度: {avg_ml_confidence:.3f}")
    
    return results

def run_strategy_v9_with_monitoring(df: pd.DataFrame, params: dict) -> Dict[str, any]:
    """
    v9 带监控的策略运行器 - 包含完整的监控和报告生成
    """
    # 运行策略
    trades = run_strategy_v9_ml_enhanced(df, params, "ml_enhanced", "ultra_profitable")
    
    # 初始化监控系统
    monitor = RealTimeMonitor(params.get("monitoring", {}))
    
    # 添加所有交易到监控系统
    for trade in trades:
        monitor.add_trade(trade)
    
    # 生成性能报告
    performance_report = monitor.generate_performance_report()
    
    # 获取仪表板数据
    dashboard_data = monitor.get_dashboard_data()
    
    # 生成优化建议
    performance_tracker = PerformanceTracker()
    if not trades.empty:
        recent_performance = {
            "win_rate": len(trades[trades["pnl"] > 0]) / len(trades),
            "avg_pnl": trades["pnl"].mean(),
            "total_pnl": trades["pnl"].sum(),
            "max_drawdown": trades["pnl"].cumsum().expanding().max() - trades["pnl"].cumsum()
        }
        performance_tracker.add_performance_data(recent_performance)
    
    optimization_suggestions = performance_tracker.get_optimization_suggestions()
    performance_trend = performance_tracker.get_performance_trend()
    
    return {
        'trades': trades,
        'performance_report': performance_report,
        'dashboard_data': dashboard_data,
        'optimization_suggestions': optimization_suggestions,
        'performance_trend': performance_trend
    }
