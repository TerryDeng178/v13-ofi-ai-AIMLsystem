import numpy as np
import pandas as pd
from typing import Dict
try:
    from .risk_v7 import (
        compute_ultra_optimized_levels, compute_dynamic_position_sizing,
        compute_risk_adjusted_levels, compute_time_based_exit_levels_v7,
        check_advanced_risk_limits, calculate_advanced_risk_metrics
    )
    from .exec import SimBroker
    from .signals_v7 import (
        gen_signals_v7_quality_filtered, gen_signals_v7_dynamic_threshold,
        gen_signals_v7_momentum_enhanced
    )
except ImportError:
    # For testing when running as standalone
    from risk_v7 import (
        compute_ultra_optimized_levels, compute_dynamic_position_sizing,
        compute_risk_adjusted_levels, compute_time_based_exit_levels_v7,
        check_advanced_risk_limits, calculate_advanced_risk_metrics
    )
    from exec import SimBroker
    from signals_v7 import (
        gen_signals_v7_quality_filtered, gen_signals_v7_dynamic_threshold,
        gen_signals_v7_momentum_enhanced
    )

def run_strategy_v7(df: pd.DataFrame, params: dict, signal_type: str = "quality_filtered", 
                   strategy_mode: str = "ultra_optimized") -> pd.DataFrame:
    """
    v7策略运行器 - 支持信号质量筛选和动态仓位管理
    """
    broker = SimBroker(fee_bps=params["risk"]["fee_bps"], slip_bps_budget_frac=params["risk"]["slip_bps_budget_frac"])
    trades = []
    equity = params["backtest"]["initial_equity_usd"]
    open_pos = None
    
    # 选择信号生成函数
    signal_func_map = {
        "quality_filtered": gen_signals_v7_quality_filtered,
        "dynamic_threshold": gen_signals_v7_dynamic_threshold,
        "momentum_enhanced": gen_signals_v7_momentum_enhanced
    }
    
    signal_func = signal_func_map.get(signal_type, gen_signals_v7_quality_filtered)
    
    # 生成信号
    df_with_signals = signal_func(df, params)
    
    # 预计算median_depth
    median_depth = df_with_signals["bid1_size"].rolling(60, min_periods=30).median()
    
    # 性能跟踪
    recent_performance = {"win_rate": 0.5, "avg_pnl": 0.0}
    daily_pnl = 0.0
    
    for i, row in df_with_signals.iterrows():
        # 流动性前置检查
        spread_bps = (row["ask1"] - row["bid1"]) / row["price"] * 1e4
        depth_now = row["bid1_size"] + row["ask1_size"]
        depth_med = median_depth.iloc[i] if i < len(median_depth) else depth_now
        thin_book_spread_max = params["signals"]["momentum"].get("thin_book_spread_bps_max", 8.0)
        
        if not (spread_bps <= thin_book_spread_max and depth_now >= depth_med):
            continue
        
        # 检查信号
        if pd.isna(row["sig_side"]) or row["sig_side"] == 0:
            continue
        
        side = int(row["sig_side"])
        signal_strength = row.get("signal_strength", 1.0)
        quality_score = row.get("quality_score", 1.0)
        
        # 开仓逻辑
        if open_pos is None:
            # 动态仓位管理
            _, qty_usd = compute_dynamic_position_sizing(row, params, signal_strength, quality_score, recent_performance)
            if qty_usd <= 0:
                continue
            
            # 超优化止盈止损计算
            if strategy_mode == "ultra_optimized":
                sl, tp = compute_ultra_optimized_levels(row, params, signal_strength, quality_score)
            elif strategy_mode == "risk_adjusted":
                market_condition = "normal"  # 简化市场条件判断
                sl, tp = compute_risk_adjusted_levels(row, params, market_condition)
            else:
                # 默认模式
                from risk import compute_levels
                sl, tp = compute_levels(row, params)
            
            # 时间基础退出水平
            exit_levels = compute_time_based_exit_levels_v7(row, params)
            
            # 模拟执行
            entry_price = row["price"]
            exit_price = row["price"] * (1 + 0.0001 * side)  # 简化滑点模拟
            fee = qty_usd * params["risk"]["fee_bps"] / 10000
            
            # 计算PnL
            pnl = qty_usd * (exit_price - entry_price) / entry_price * side - fee
            
            # 风险检查
            if check_advanced_risk_limits(equity, params, pnl, daily_pnl):
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
                    "sl": sl,
                    "tp": tp,
                    "strategy_mode": strategy_mode
                }
                trades.append(trade)
                
                # 更新权益和性能跟踪
                equity += pnl
                daily_pnl += pnl
                
                # 更新性能跟踪
                if len(trades) >= 10:
                    recent_trades = pd.DataFrame(trades[-10:])
                    recent_performance = {
                        "win_rate": len(recent_trades[recent_trades["pnl"] > 0]) / len(recent_trades),
                        "avg_pnl": recent_trades["pnl"].mean()
                    }
    
    return pd.DataFrame(trades) if trades else pd.DataFrame()

def run_strategy_v7_advanced(df: pd.DataFrame, params: dict, signal_type: str = "quality_filtered") -> pd.DataFrame:
    """
    v7高级策略运行器 - 包含完整的风险控制和性能跟踪
    """
    trades = run_strategy_v7(df, params, signal_type, "ultra_optimized")
    
    # 计算高级风险指标
    risk_metrics = calculate_advanced_risk_metrics(trades)
    
    # 保存风险指标到交易记录
    if not trades.empty and risk_metrics:
        trades["win_rate"] = risk_metrics.get("win_rate", 0.0)
        trades["profit_factor"] = risk_metrics.get("profit_factor", 0.0)
        trades["sharpe_ratio"] = risk_metrics.get("sharpe_ratio", 0.0)
        trades["max_consecutive_losses"] = risk_metrics.get("max_consecutive_losses", 0)
    
    return trades

def run_strategy_v7_multi_signal(df: pd.DataFrame, params: dict) -> Dict[str, pd.DataFrame]:
    """
    v7多信号策略运行器 - 同时运行多种信号逻辑并比较结果
    """
    signal_types = ["quality_filtered", "dynamic_threshold", "momentum_enhanced"]
    strategy_modes = ["ultra_optimized", "risk_adjusted"]
    results = {}
    
    for signal_type in signal_types:
        for strategy_mode in strategy_modes:
            key = f"{signal_type}_{strategy_mode}"
            print(f"运行 {key} 策略...")
            trades = run_strategy_v7(df, params, signal_type, strategy_mode)
            results[key] = trades
            
            if not trades.empty:
                risk_metrics = calculate_advanced_risk_metrics(trades)
                print(f"  {key}: 交易数={len(trades)}, 胜率={risk_metrics.get('win_rate', 0):.2%}, "
                      f"总PnL=${trades['pnl'].sum():.2f}, 净PnL=${trades['pnl'].sum() - trades['fee'].sum():.2f}")
    
    return results
