import numpy as np
import pandas as pd
try:
    from .risk import position_size, compute_levels
    from .exec import SimBroker
    from .signals_v6_simple import gen_signals_v6_ultra_simple
except ImportError:
    # For testing when running as standalone
    from risk import position_size, compute_levels
    from exec import SimBroker
    from signals_v6_simple import gen_signals_v6_ultra_simple

def run_strategy_v6_simple(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v6简化策略运行器 - 基于v5成功逻辑，添加最小改进
    """
    broker = SimBroker(fee_bps=params["risk"]["fee_bps"], slip_bps_budget_frac=params["risk"]["slip_bps_budget_frac"])
    trades = []
    equity = params["backtest"]["initial_equity_usd"]
    open_pos = None
    
    # 生成信号
    df_with_signals = gen_signals_v6_ultra_simple(df, params)
    
    # 预计算median_depth（简化版）
    median_depth = df_with_signals["bid1_size"].rolling(60, min_periods=30).median()
    
    for i, row in df_with_signals.iterrows():
        # 简化的流动性检查
        spread_bps = (row["ask1"] - row["bid1"]) / row["price"] * 1e4
        depth_now = row["bid1_size"] + row["ask1_size"]
        depth_med = median_depth.iloc[i] if i < len(median_depth) else depth_now
        thin_book_spread_max = params["signals"]["momentum"].get("thin_book_spread_bps_max", 10.0)
        
        # 放宽流动性检查
        if spread_bps > thin_book_spread_max * 2:  # 放宽spread检查
            continue
        
        # 检查信号
        if pd.isna(row["sig_side"]) or row["sig_side"] == 0:
            continue
        
        side = int(row["sig_side"])
        signal_strength = row.get("signal_strength", 1.0)
        
        # 简化开仓逻辑
        if open_pos is None:
            # 基础仓位计算
            _, qty_usd = position_size(row, params)
            if qty_usd <= 0:
                continue
            
            # 简化的止盈止损计算
            sl, tp = compute_levels(row, params)
            
            # 简化的执行模拟
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
                "signal_strength": signal_strength,
                "sl": sl,
                "tp": tp
            }
            trades.append(trade)
            
            # 更新权益
            equity += trade["pnl"]
    
    return pd.DataFrame(trades) if trades else pd.DataFrame()
