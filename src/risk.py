
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any

class TakeProfitLevel:
    def __init__(self, price: float, time_seconds: int, level_type: str, description: str):
        self.price = price
        self.time_seconds = time_seconds
        self.level_type = level_type  # 'vwap', '1.5r', etc.
        self.description = description

def position_size(row, params):
    k = params["signals"]["sizing"]["k_ofi"]
    size_max = params["signals"]["sizing"]["size_max_usd"]
    # scale by ofi_z magnitude
    scale = min(1.0, max(0.1, abs(row.get("ofi_z", 0.0)) * k))
    notional = scale * size_max
    side = row["sig_side"]
    return side, notional

def compute_levels(row, params):
    atr = float(row["atr"]); price = float(row["price"])
    lo = params["risk"]["atr_stop_lo"]
    tick = max(float(row.get("ask1", price) - row.get("bid1", price)), 1e-2)
    min_sl = max(params["risk"].get("min_tick_sl_mult", 6) * tick, 1e-2)
    atr_sl = max(lo * atr, min_sl)
    if row["sig_side"] > 0:
        sl = price - atr_sl; tp = max(row.get("vwap", price), price + 0.5*atr)
    else:
        sl = price + atr_sl; tp = min(row.get("vwap", price), price - 0.5*atr)
    return sl, tp

def compute_time_based_tp_levels(row, params, entry_time) -> list:
    """
    Compute time-based take profit levels.
    Returns list of TakeProfitLevel objects in chronological order.
    """
    if not params["risk"].get("time_tp_enabled", False):
        return []
    
    atr = row["atr"]
    price = row["price"]
    side = row["sig_side"]
    vwap = row.get("vwap", price)
    
    tp_levels = []
    risk_params = params["risk"]
    
    # First TP: VWAP after specified time
    vwap_seconds = risk_params.get("time_tp_vwap_seconds", 300)
    if side > 0:  # Long position
        vwap_tp = max(vwap, price)  # Take profit at VWAP or higher
        tp_levels.append(TakeProfitLevel(
            price=vwap_tp,
            time_seconds=vwap_seconds,
            level_type="vwap",
            description=f"VWAP TP after {vwap_seconds}s"
        ))
    else:  # Short position
        vwap_tp = min(vwap, price)  # Take profit at VWAP or lower
        tp_levels.append(TakeProfitLevel(
            price=vwap_tp,
            time_seconds=vwap_seconds,
            level_type="vwap",
            description=f"VWAP TP after {vwap_seconds}s"
        ))
    
    # Second TP: 1.5R after specified time
    r_multiplier = risk_params.get("time_tp_15r_multiplier", 1.5)
    r_seconds = risk_params.get("time_tp_15r_seconds", 600)
    
    # Calculate R (risk) based on stop loss
    stop_loss = price - max(risk_params["atr_stop_lo"] * atr, 1e-8) if side > 0 else price + max(risk_params["atr_stop_lo"] * atr, 1e-8)
    r_amount = abs(price - stop_loss)
    
    if side > 0:  # Long position
        r_tp = price + (r_multiplier * r_amount)
        tp_levels.append(TakeProfitLevel(
            price=r_tp,
            time_seconds=r_seconds,
            level_type="1.5r",
            description=f"{r_multiplier}R TP after {r_seconds}s"
        ))
    else:  # Short position
        r_tp = price - (r_multiplier * r_amount)
        tp_levels.append(TakeProfitLevel(
            price=r_tp,
            time_seconds=r_seconds,
            level_type="1.5r",
            description=f"{r_multiplier}R TP after {r_seconds}s"
        ))
    
    return tp_levels

def check_time_based_exit(current_time, entry_time, tp_levels, current_price, side) -> Optional[Tuple[str, float]]:
    """
    Check if any time-based take profit level should be triggered.
    Returns (exit_reason, exit_price) if exit should occur, None otherwise.
    """
    holding_seconds = (current_time - entry_time).total_seconds()
    
    for tp_level in tp_levels:
        if holding_seconds >= tp_level.time_seconds:
            # Check if price has reached the TP level
            if side > 0 and current_price >= tp_level.price:
                return f"time_tp_{tp_level.level_type}", tp_level.price
            elif side < 0 and current_price <= tp_level.price:
                return f"time_tp_{tp_level.level_type}", tp_level.price
    
    return None

class CircuitBreaker:
    def __init__(self, equity0, risk_params):
        self.equity0 = equity0
        self.risk_params = risk_params
        self.daily_pnl = 0.0

    def update(self, pnl):
        self.daily_pnl += pnl

    def halted(self):
        limit = - self.risk_params["daily_drawdown_stop_pct"] * self.equity0
        return self.daily_pnl <= limit
