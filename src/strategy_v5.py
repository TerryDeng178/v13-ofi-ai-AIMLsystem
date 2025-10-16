import numpy as np
import pandas as pd
try:
    from .risk import position_size, compute_levels
    from .exec import SimBroker
    from .signals_v5 import gen_signals_v5_ultra_simple, gen_signals_v5_reversal, gen_signals_v5_momentum_breakout
except ImportError:
    # For testing when running as standalone
    from risk import position_size, compute_levels
    from exec import SimBroker
    from signals_v5 import gen_signals_v5_ultra_simple, gen_signals_v5_reversal, gen_signals_v5_momentum_breakout

def run_strategy_v5(df: pd.DataFrame, params: dict, signal_type: str = "ultra_simple"):
    """
    v5策略运行器，支持不同的信号逻辑
    """
    broker = SimBroker(fee_bps=params["risk"]["fee_bps"], slip_bps_budget_frac=params["risk"]["slip_bps_budget_frac"])
    trades = []
    equity = params["backtest"]["initial_equity_usd"]
    open_pos = None
    
    # 选择信号生成函数
    if signal_type == "ultra_simple":
        signal_func = gen_signals_v5_ultra_simple
    elif signal_type == "reversal":
        signal_func = gen_signals_v5_reversal
    elif signal_type == "momentum_breakout":
        signal_func = gen_signals_v5_momentum_breakout
    else:
        signal_func = gen_signals_v5_ultra_simple
    
    # 生成信号
    df_with_signals = signal_func(df, params)
    
    for i, row in df_with_signals.iterrows():
        # 预计算median_depth以避免在循环中重复计算
        if i == 0:
            median_depth = df_with_signals["bid1_size"].rolling(60, min_periods=30).median()
        
        # 流动性前置检查 (简化版)
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
        
        # 简化版策略：直接开仓，不做复杂检查
        if open_pos is None:
            # 计算仓位大小
            _, qty_usd = position_size(row, params)
            if qty_usd <= 0:
                continue
            
            # 计算止盈止损
            sl, tp = compute_levels(row, params)
            
            # 简化执行：直接成交
            entry_price = row["price"]
            exit_price = row["price"] * (1 + 0.0001 * side)  # 简化滑点模拟
            fee = qty_usd * params["risk"]["fee_bps"] / 10000
            
            # 记录交易
            trade = {
                "entry_ts": row["ts"],
                "exit_ts": row["ts"] + pd.Timedelta(seconds=30),  # 简化：固定30秒持仓
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "qty_usd": qty_usd,
                "pnl": qty_usd * (exit_price - entry_price) / entry_price * side - fee,
                "fee": fee,
                "holding_sec": 30,
                "sl": sl,
                "tp": tp
            }
            trades.append(trade)
            
            # 更新权益
            equity += trade["pnl"]
    
    return pd.DataFrame(trades) if trades else pd.DataFrame()

def run_strategy_v5_advanced(df: pd.DataFrame, params: dict, signal_type: str = "ultra_simple"):
    """
    v5高级策略运行器，包含更复杂的逻辑
    """
    broker = SimBroker(fee_bps=params["risk"]["fee_bps"], slip_bps_budget_frac=params["risk"]["slip_bps_budget_frac"])
    trades = []
    equity = params["backtest"]["initial_equity_usd"]
    open_pos = None
    
    # 选择信号生成函数
    if signal_type == "ultra_simple":
        signal_func = gen_signals_v5_ultra_simple
    elif signal_type == "reversal":
        signal_func = gen_signals_v5_reversal
    elif signal_type == "momentum_breakout":
        signal_func = gen_signals_v5_momentum_breakout
    else:
        signal_func = gen_signals_v5_ultra_simple
    
    # 生成信号
    df_with_signals = signal_func(df, params)
    
    # 预计算median_depth
    median_depth = df_with_signals["bid1_size"].rolling(60, min_periods=30).median()
    
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
        
        # 高级策略：包含更多检查
        if open_pos is None:
            # 计算仓位大小
            _, qty_usd = position_size(row, params)
            if qty_usd <= 0:
                continue
            
            # 计算止盈止损
            sl, tp = compute_levels(row, params)
            
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
                    "fee": entry_result.fee
                }
        else:
            # 检查平仓条件
            current_price = row["price"]
            
            # 止损/止盈检查
            if side > 0:  # 多头
                if current_price <= open_pos["sl"] or current_price >= open_pos["tp"]:
                    # 平仓
                    exit_result = broker.simulate_fill(-side, open_pos["qty_usd"], current_price, row["atr"], 0.0001)
                    
                    if exit_result.status == "filled":
                        # 计算PnL
                        pnl = (exit_result.fill_price - open_pos["entry"]) / open_pos["entry"] * open_pos["side"] * open_pos["qty_usd"]
                        total_fee = open_pos["fee"] + exit_result.fee
                        net_pnl = pnl - total_fee
                        
                        # 记录交易
                        trade = {
                            "entry_ts": open_pos["entry_ts"],
                            "exit_ts": row["ts"],
                            "side": open_pos["side"],
                            "entry_price": open_pos["entry"],
                            "exit_price": exit_result.fill_price,
                            "qty_usd": open_pos["qty_usd"],
                            "pnl": net_pnl,
                            "fee": total_fee,
                            "holding_sec": (row["ts"] - open_pos["entry_ts"]).total_seconds(),
                            "sl": open_pos["sl"],
                            "tp": open_pos["tp"]
                        }
                        trades.append(trade)
                        
                        # 更新权益
                        equity += net_pnl
                        
                        # 重置开仓
                        open_pos = None
    
    return pd.DataFrame(trades) if trades else pd.DataFrame()
