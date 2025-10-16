import numpy as np
import pandas as pd
from typing import Dict
try:
    from .risk_v6 import (
        compute_dynamic_levels, compute_adaptive_levels, 
        compute_risk_adjusted_position_size, compute_time_based_exit_levels,
        check_risk_limits, calculate_risk_metrics
    )
    from .exec import SimBroker
    from .signals_v6 import (
        gen_signals_v6_quality, gen_signals_v6_momentum_enhanced,
        gen_signals_v6_reversal_enhanced, gen_signals_v6_adaptive
    )
    from .signals_v6_simple import (
        gen_signals_v6_ultra_simple, gen_signals_v6_minimal_enhancement
    )
except ImportError:
    # For testing when running as standalone
    from risk_v6 import (
        compute_dynamic_levels, compute_adaptive_levels,
        compute_risk_adjusted_position_size, compute_time_based_exit_levels,
        check_risk_limits, calculate_risk_metrics
    )
    from exec import SimBroker
    from signals_v6 import (
        gen_signals_v6_quality, gen_signals_v6_momentum_enhanced,
        gen_signals_v6_reversal_enhanced, gen_signals_v6_adaptive
    )
    from signals_v6_simple import (
        gen_signals_v6_ultra_simple, gen_signals_v6_minimal_enhancement
    )

def run_strategy_v6(df: pd.DataFrame, params: dict, signal_type: str = "quality", 
                   strategy_mode: str = "dynamic") -> pd.DataFrame:
    """
    v6策略运行器 - 支持多种信号逻辑和策略模式
    """
    broker = SimBroker(fee_bps=params["risk"]["fee_bps"], slip_bps_budget_frac=params["risk"]["slip_bps_budget_frac"])
    trades = []
    equity = params["backtest"]["initial_equity_usd"]
    open_pos = None
    
    # 选择信号生成函数
    signal_func_map = {
        "quality": gen_signals_v6_quality,
        "momentum_enhanced": gen_signals_v6_momentum_enhanced,
        "reversal_enhanced": gen_signals_v6_reversal_enhanced,
        "adaptive": gen_signals_v6_adaptive,
        "ultra_simple": gen_signals_v6_ultra_simple,
        "minimal_enhancement": gen_signals_v6_minimal_enhancement
    }
    
    signal_func = signal_func_map.get(signal_type, gen_signals_v6_quality)
    
    # 生成信号
    df_with_signals = signal_func(df, params)
    
    # 预计算median_depth
    median_depth = df_with_signals["bid1_size"].rolling(60, min_periods=30).median()
    
    # 性能跟踪
    recent_performance = {"win_rate": 0.5, "avg_pnl": 0.0}
    
    for i, row in df_with_signals.iterrows():
        # 流动性前置检查
        spread_bps = (row["ask1"] - row["bid1"]) / row["price"] * 1e4
        depth_now = row["bid1_size"] + row["ask1_size"]
        depth_med = median_depth.iloc[i] if i < len(median_depth) else depth_now
        thin_book_spread_max = params["signals"]["momentum"].get("thin_book_spread_bps_max", 10.0)
        
        if not (spread_bps <= thin_book_spread_max and depth_now >= depth_med):
            continue
        
        # 检查信号
        if pd.isna(row["sig_side"]) or row["sig_side"] == 0:
            continue
        
        side = int(row["sig_side"])
        signal_strength = row.get("signal_strength", 1.0)
        
        # 开仓逻辑
        if open_pos is None:
            # 风险调整仓位大小
            _, qty_usd = compute_risk_adjusted_position_size(row, params, recent_performance)
            if qty_usd <= 0:
                continue
            
            # 动态止盈止损计算
            if strategy_mode == "dynamic":
                sl, tp = compute_dynamic_levels(row, params, signal_strength)
            elif strategy_mode == "adaptive":
                market_vol = row.get("atr", 0.001) / row["price"]  # 简化的市场波动率
                sl, tp = compute_adaptive_levels(row, params, market_vol)
            else:
                # 默认模式
                from risk import compute_levels
                sl, tp = compute_levels(row, params)
            
            # 时间基础退出水平
            exit_levels = compute_time_based_exit_levels(row, params)
            
            # 模拟执行
            entry_result = broker.simulate_fill(side, qty_usd, row["price"], row["atr"], 0.0001)
            
            if entry_result.status == "filled":
                # 记录开仓
                open_pos = {
                    "entry": entry_result.fill_price,
                    "side": side,
                    "qty_usd": qty_usd,
                    "sl": sl,
                    "tp": tp,
                    "entry_ts": row["ts"],
                    "fee": entry_result.fee,
                    "signal_strength": signal_strength,
                    "exit_levels": exit_levels
                }
        else:
            # 平仓逻辑
            current_price = row["price"]
            current_time = row["ts"]
            holding_time = (current_time - open_pos["entry_ts"]).total_seconds()
            
            should_exit = False
            exit_reason = ""
            exit_price = current_price
            
            # 止损/止盈检查
            if side > 0:  # 多头
                if current_price <= open_pos["sl"]:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif current_price >= open_pos["tp"]:
                    should_exit = True
                    exit_reason = "take_profit"
            else:  # 空头
                if current_price >= open_pos["sl"]:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif current_price <= open_pos["tp"]:
                    should_exit = True
                    exit_reason = "take_profit"
            
            # 时间基础退出检查
            if not should_exit and open_pos["exit_levels"]:
                for level_name, level_info in open_pos["exit_levels"].items():
                    if holding_time >= level_info["time_seconds"]:
                        if side > 0 and current_price >= level_info["price"]:
                            should_exit = True
                            exit_reason = f"time_{level_name}"
                            exit_price = level_info["price"]
                            break
                        elif side < 0 and current_price <= level_info["price"]:
                            should_exit = True
                            exit_reason = f"time_{level_name}"
                            exit_price = level_info["price"]
                            break
            
            # 最大持仓时间检查
            max_holding_time = params["risk"].get("time_exit_seconds_max", 900)
            if not should_exit and holding_time >= max_holding_time:
                should_exit = True
                exit_reason = "max_time"
            
            # 执行平仓
            if should_exit:
                exit_result = broker.simulate_fill(-side, open_pos["qty_usd"], exit_price, row["atr"], 0.0001)
                
                if exit_result.status == "filled":
                    # 计算PnL
                    pnl = (exit_result.fill_price - open_pos["entry"]) / open_pos["entry"] * open_pos["side"] * open_pos["qty_usd"]
                    total_fee = open_pos["fee"] + exit_result.fee
                    net_pnl = pnl - total_fee
                    
                    # 风险检查
                    if check_risk_limits(equity, params, net_pnl):
                        # 记录交易
                        trade = {
                            "entry_ts": open_pos["entry_ts"],
                            "exit_ts": current_time,
                            "side": open_pos["side"],
                            "entry_price": open_pos["entry"],
                            "exit_price": exit_result.fill_price,
                            "qty_usd": open_pos["qty_usd"],
                            "pnl": net_pnl,
                            "fee": total_fee,
                            "holding_sec": holding_time,
                            "exit_reason": exit_reason,
                            "signal_strength": open_pos["signal_strength"],
                            "sl": open_pos["sl"],
                            "tp": open_pos["tp"]
                        }
                        trades.append(trade)
                        
                        # 更新权益
                        equity += net_pnl
                        
                        # 更新性能跟踪
                        if len(trades) >= 10:
                            recent_trades = pd.DataFrame(trades[-10:])
                            recent_performance = {
                                "win_rate": len(recent_trades[recent_trades["pnl"] > 0]) / len(recent_trades),
                                "avg_pnl": recent_trades["pnl"].mean()
                            }
                    
                    # 重置开仓
                    open_pos = None
    
    return pd.DataFrame(trades) if trades else pd.DataFrame()

def run_strategy_v6_advanced(df: pd.DataFrame, params: dict, signal_type: str = "quality") -> pd.DataFrame:
    """
    v6高级策略运行器 - 包含更复杂的风险控制和性能跟踪
    """
    trades = run_strategy_v6(df, params, signal_type, "dynamic")
    
    # 计算风险指标
    risk_metrics = calculate_risk_metrics(trades)
    
    # 保存风险指标到交易记录
    if not trades.empty and risk_metrics:
        trades["win_rate"] = risk_metrics.get("win_rate", 0.0)
        trades["profit_factor"] = risk_metrics.get("profit_factor", 0.0)
        trades["sharpe_ratio"] = risk_metrics.get("sharpe_ratio", 0.0)
    
    return trades

def run_strategy_v6_multi_signal(df: pd.DataFrame, params: dict) -> Dict[str, pd.DataFrame]:
    """
    v6多信号策略运行器 - 同时运行多种信号逻辑并比较结果
    """
    signal_types = ["quality", "momentum_enhanced", "reversal_enhanced", "adaptive"]
    results = {}
    
    for signal_type in signal_types:
        print(f"运行 {signal_type} 信号策略...")
        trades = run_strategy_v6_advanced(df, params, signal_type)
        results[signal_type] = trades
        
        if not trades.empty:
            risk_metrics = calculate_risk_metrics(trades)
            print(f"  {signal_type}: 交易数={len(trades)}, 胜率={risk_metrics.get('win_rate', 0):.2%}, "
                  f"总PnL=${trades['pnl'].sum():.2f}")
    
    return results
