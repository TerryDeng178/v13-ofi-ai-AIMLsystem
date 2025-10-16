import numpy as np
import pandas as pd
from typing import Dict
try:
    from .risk_v8 import (
        compute_ultra_profitable_levels, compute_profit_optimized_position_sizing,
        compute_cost_optimized_levels, compute_frequency_optimized_levels,
        calculate_profitability_metrics, check_profitability_limits
    )
    from .exec import SimBroker
    from .signals_v8 import (
        gen_signals_v8_intelligent_filter, gen_signals_v8_frequency_optimized
    )
except ImportError:
    # For testing when running as standalone
    from risk_v8 import (
        compute_ultra_profitable_levels, compute_profit_optimized_position_sizing,
        compute_cost_optimized_levels, compute_frequency_optimized_levels,
        calculate_profitability_metrics, check_profitability_limits
    )
    from exec import SimBroker
    from signals_v8 import (
        gen_signals_v8_intelligent_filter, gen_signals_v8_frequency_optimized
    )

def run_strategy_v8_intelligent(df: pd.DataFrame, params: dict, signal_type: str = "intelligent_filter", 
                               strategy_mode: str = "ultra_profitable") -> pd.DataFrame:
    """
    v8 智能策略运行器 - 基于多维度智能评分的策略执行
    """
    broker = SimBroker(fee_bps=params["risk"]["fee_bps"], slip_bps_budget_frac=params["risk"]["slip_bps_budget_frac"])
    trades = []
    equity = params["backtest"]["initial_equity_usd"]
    open_pos = None
    
    # 选择信号生成函数
    signal_func_map = {
        "intelligent_filter": gen_signals_v8_intelligent_filter,
        "frequency_optimized": gen_signals_v8_frequency_optimized
    }
    
    signal_func = signal_func_map.get(signal_type, gen_signals_v8_intelligent_filter)
    
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
        thin_book_spread_max = params["signals"]["momentum"].get("thin_book_spread_bps_max", 6.0)
        
        if not (spread_bps <= thin_book_spread_max and depth_now >= depth_med):
            continue
        
        # 检查信号
        if pd.isna(row["sig_side"]) or row["sig_side"] == 0:
            continue
        
        side = int(row["sig_side"])
        signal_strength = row.get("signal_strength", 1.0)
        quality_score = row.get("quality_score", 1.0)
        intelligence_score = row.get("intelligence_score", 1.0)
        frequency_score = row.get("frequency_score", 1.0)
        
        # 开仓逻辑
        if open_pos is None:
            # 智能仓位管理
            if strategy_mode == "ultra_profitable":
                _, qty_usd = compute_profit_optimized_position_sizing(
                    row, params, signal_strength, quality_score, intelligence_score, recent_performance)
            elif strategy_mode == "frequency_optimized":
                _, qty_usd = compute_profit_optimized_position_sizing(
                    row, params, signal_strength, quality_score, intelligence_score, recent_performance)
            else:
                # 默认模式
                from risk import position_size
                _, qty_usd = position_size(row, params)
            
            if qty_usd <= 0:
                continue
            
            # 智能止盈止损计算
            if strategy_mode == "ultra_profitable":
                sl, tp = compute_ultra_profitable_levels(row, params, signal_strength, quality_score, intelligence_score)
            elif strategy_mode == "frequency_optimized":
                sl, tp = compute_frequency_optimized_levels(row, params, frequency_score)
            else:
                # 默认模式
                from risk import compute_levels
                sl, tp = compute_levels(row, params)
            
            # 成本优化检查
            expected_costs = {
                "fee_bps": params["risk"]["fee_bps"],
                "slippage_bps": 2.0  # 估算滑点
            }
            cost_optimized_sl, cost_optimized_tp = compute_cost_optimized_levels(row, params, expected_costs)
            
            # 选择最优止盈止损
            if strategy_mode == "ultra_profitable":
                sl, tp = cost_optimized_sl, cost_optimized_tp
            
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
                    "holding_sec": 30,
                    "signal_strength": signal_strength,
                    "quality_score": quality_score,
                    "intelligence_score": intelligence_score,
                    "frequency_score": frequency_score,
                    "sl": sl,
                    "tp": tp,
                    "strategy_mode": strategy_mode
                }
                trades.append(trade)
                
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

def run_strategy_v8_advanced(df: pd.DataFrame, params: dict, signal_type: str = "intelligent_filter") -> pd.DataFrame:
    """
    v8 高级策略运行器 - 包含完整的盈利能力控制和性能跟踪
    """
    trades = run_strategy_v8_intelligent(df, params, signal_type, "ultra_profitable")
    
    # 计算盈利能力指标
    profitability_metrics = calculate_profitability_metrics(trades)
    
    # 保存盈利能力指标到交易记录
    if not trades.empty and profitability_metrics:
        trades["win_rate"] = profitability_metrics.get("win_rate", 0.0)
        trades["profit_factor"] = profitability_metrics.get("profit_factor", 0.0)
        trades["cost_efficiency"] = profitability_metrics.get("cost_efficiency", 0.0)
        trades["profitability_score"] = profitability_metrics.get("profitability_score", 0.0)
    
    return trades

def run_strategy_v8_multi_signal(df: pd.DataFrame, params: dict) -> Dict[str, pd.DataFrame]:
    """
    v8 多信号策略运行器 - 同时运行多种信号逻辑并比较盈利能力
    """
    signal_types = ["intelligent_filter", "frequency_optimized"]
    strategy_modes = ["ultra_profitable", "frequency_optimized"]
    results = {}
    
    for signal_type in signal_types:
        for strategy_mode in strategy_modes:
            key = f"{signal_type}_{strategy_mode}"
            print(f"运行 {key} 策略...")
            trades = run_strategy_v8_intelligent(df, params, signal_type, strategy_mode)
            results[key] = trades
            
            if not trades.empty:
                profitability_metrics = calculate_profitability_metrics(trades)
                print(f"  {key}: 交易数={len(trades)}, 胜率={profitability_metrics.get('win_rate', 0):.2%}, "
                      f"净PnL=${profitability_metrics.get('net_pnl', 0):.2f}, "
                      f"成本效率={profitability_metrics.get('cost_efficiency', 0):.2f}, "
                      f"盈利能力评分={profitability_metrics.get('profitability_score', 0):.3f}")
    
    return results
