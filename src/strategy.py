
import numpy as np
import pandas as pd
try:
    from .risk import position_size, compute_levels
    from .exec import SimBroker
except ImportError:
    # For testing when running as standalone
    from risk import position_size, compute_levels
    from exec import SimBroker

def run_strategy(df: pd.DataFrame, params: dict):
    broker = SimBroker(fee_bps=params["risk"]["fee_bps"], slip_bps_budget_frac=params["risk"]["slip_bps_budget_frac"])
    trades = []
    equity = params["backtest"]["initial_equity_usd"]
    open_pos = None  # dict with entry, side, qty_usd, sl, tp, entry_ts

    for i, row in df.iterrows():
        # 预计算median_depth以避免在循环中重复计算
        if i == 0:
            median_depth = (df["bid1_size"] + df["ask1_size"]).rolling(300).median().bfill()
        if open_pos is not None:
            # check stops / targets / time exit
            price = row["price"]
            elapsed = (row["ts"] - open_pos["entry_ts"]).total_seconds()
            time_lo = params["risk"]["time_exit_seconds_min"]
            time_hi = params["risk"]["time_exit_seconds_max"]
            hit = False
            if open_pos["side"] > 0:
                if price <= open_pos["sl"] or price >= open_pos["tp"]:
                    hit = True
            else:
                if price >= open_pos["sl"] or price <= open_pos["tp"]:
                    hit = True
            if hit or elapsed >= time_hi:
                # close at market
                exit_result = broker.simulate_fill(-open_pos["side"], open_pos["qty_usd"], price, row["atr"], 0)
                if exit_result.status.value == "FILLED":
                    exit_price = exit_result.fill_price
                    fee2 = exit_result.fee
                    slip2 = exit_result.slippage_bps
                    pnl = (exit_price - open_pos["entry_px"]) * open_pos["side"] * (open_pos["qty_usd"]/open_pos["entry_px"]) - (open_pos["fee"] + fee2)
                else:
                    # If exit order fails, use market price and continue
                    exit_price = price
                    fee2 = 0
                    slip2 = 0
                    pnl = (exit_price - open_pos["entry_px"]) * open_pos["side"] * (open_pos["qty_usd"]/open_pos["entry_px"]) - open_pos["fee"]
                trades.append({
                    "entry_ts": open_pos["entry_ts"],
                    "exit_ts": row["ts"],
                    "side": open_pos["side"],
                    "entry_px": open_pos["entry_px"],
                    "exit_px": exit_price,
                    "qty_usd": open_pos["qty_usd"],
                    "pnl": pnl,
                    "fee": open_pos["fee"] + fee2
                })
                equity += pnl
                open_pos = None

        # open new position if no open and signal present
        if open_pos is None and row.get("sig_side", 0) != 0:
            # 硬约束1: 流动性前置检查 (v3)
            spread_bps = (row["ask1"] - row["bid1"]) / row["price"] * 1e4
            depth_now = row["bid1_size"] + row["ask1_size"]  # 可替换为 top-5 合计
            depth_med = median_depth.iloc[i] if i < len(median_depth) else depth_now
            if not (spread_bps <= params["signals"]["momentum"]["thin_book_spread_bps_max"] and depth_now >= depth_med):
                continue

            # 硬约束2: 会话窗过滤（仅对背离信号）
            minute = row["ts"].minute; hour = row["ts"].hour
            session_window = params["execution"].get("session_window_minutes", 15)
            def in_window(h,m):
                return (h==8 and m<=session_window) or (h==13 and m<=session_window) or (h==20 and m<=session_window) or (h in [7,12,19] and m>=(60-session_window))
            if row.get("sig_type") == "divergence" and not in_window(hour, minute):
                continue
                
            side, qty_usd = position_size(row, params)
            
            # 硬约束3: 滑点预算→拒单 (v3)
            exp_reward = max(abs(row["vwap"] - row["price"]), 0.5 * row["atr"])
            entry_result = broker.simulate_fill(side, qty_usd, row["price"], row["atr"], exp_reward)
            budget_bps = min(params["execution"]["max_slippage_bps"],
                           params["risk"]["slip_bps_budget_frac"] * (exp_reward / row["price"]) * 1e4)
            if entry_result.slippage_bps > budget_bps:
                continue
                
            if entry_result.status.value == "FILLED":
                entry_px = entry_result.fill_price
                fee = entry_result.fee
                slip = entry_result.slippage_bps
                sl, tp = compute_levels(row, params)
                open_pos = {
                    "entry_ts": row["ts"],
                    "side": side,
                    "qty_usd": qty_usd,
                    "entry_px": entry_px,
                    "sl": sl,
                    "tp": tp,
                    "fee": fee
                }

    return pd.DataFrame(trades)
