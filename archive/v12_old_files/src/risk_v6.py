import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any

def compute_dynamic_levels(row: pd.Series, params: dict, signal_strength: float = 1.0) -> Tuple[float, float]:
    """
    动态止盈止损计算 - 根据信号强度调整
    """
    atr = float(row["atr"])
    price = float(row["price"])
    
    base_stop = params["risk"]["atr_stop_lo"]
    base_take = params["risk"]["atr_stop_hi"]
    
    # 根据信号强度调整
    strength_multiplier = min(2.0, max(0.5, signal_strength / 1.5))
    
    # 强信号用更紧止损，更高止盈
    dynamic_stop = base_stop / strength_multiplier
    dynamic_take = base_take * strength_multiplier
    
    # 最小tick止损保护
    tick = max(float(row.get("ask1", price) - row.get("bid1", price)), 1e-2)
    min_tick_sl_mult = params["risk"].get("min_tick_sl_mult", 2)
    min_sl = max(min_tick_sl_mult * tick, 1e-2)
    
    atr_sl = max(dynamic_stop * atr, min_sl)
    
    if row["sig_side"] > 0:
        sl = price - atr_sl
        tp = price + dynamic_take * atr
    else:
        sl = price + atr_sl
        tp = price - dynamic_take * atr
    
    return sl, tp

def compute_adaptive_levels(row: pd.Series, params: dict, market_volatility: float = 1.0) -> Tuple[float, float]:
    """
    自适应止盈止损计算 - 根据市场波动率调整
    """
    atr = float(row["atr"])
    price = float(row["price"])
    
    base_stop = params["risk"]["atr_stop_lo"]
    base_take = params["risk"]["atr_stop_hi"]
    
    # 根据市场波动率调整
    vol_multiplier = min(1.5, max(0.7, market_volatility))
    
    # 高波动率市场用更宽止损，更高止盈
    adaptive_stop = base_stop * vol_multiplier
    adaptive_take = base_take * vol_multiplier
    
    # 最小tick止损保护
    tick = max(float(row.get("ask1", price) - row.get("bid1", price)), 1e-2)
    min_tick_sl_mult = params["risk"].get("min_tick_sl_mult", 2)
    min_sl = max(min_tick_sl_mult * tick, 1e-2)
    
    atr_sl = max(adaptive_stop * atr, min_sl)
    
    if row["sig_side"] > 0:
        sl = price - atr_sl
        tp = price + adaptive_take * atr
    else:
        sl = price + atr_sl
        tp = price - adaptive_take * atr
    
    return sl, tp

def compute_risk_adjusted_position_size(row: pd.Series, params: dict, recent_performance: Dict = None) -> Tuple[int, float]:
    """
    风险调整仓位大小计算
    """
    k = params["signals"]["sizing"]["k_ofi"]
    size_max = params["signals"]["sizing"]["size_max_usd"]
    
    # 基础仓位计算
    ofi_z = abs(row.get("ofi_z", 0.0))
    base_scale = min(1.0, max(0.1, ofi_z * k))
    
    # 风险调整
    risk_multiplier = 1.0
    if recent_performance:
        win_rate = recent_performance.get("win_rate", 0.5)
        avg_pnl = recent_performance.get("avg_pnl", 0.0)
        
        # 基于胜率调整
        if win_rate > 0.6:
            risk_multiplier *= 1.2  # 高胜率时增加仓位
        elif win_rate < 0.4:
            risk_multiplier *= 0.8  # 低胜率时减少仓位
        
        # 基于平均PnL调整
        if avg_pnl > 0:
            risk_multiplier *= 1.1  # 盈利时增加仓位
        else:
            risk_multiplier *= 0.9  # 亏损时减少仓位
    
    # 最终仓位
    adjusted_scale = base_scale * risk_multiplier
    notional = adjusted_scale * size_max
    side = row["sig_side"]
    
    return side, notional

def compute_time_based_exit_levels(row: pd.Series, params: dict) -> Dict[str, Any]:
    """
    时间基础退出水平计算
    """
    price = float(row["price"])
    atr = float(row["atr"])
    
    # 时间止盈配置
    time_tp_enabled = params["risk"].get("time_tp_enabled", True)
    time_tp_vwap_seconds = params["risk"].get("time_tp_vwap_seconds", 60)
    time_tp_15r_seconds = params["risk"].get("time_tp_15r_seconds", 120)
    time_tp_15r_multiplier = params["risk"].get("time_tp_15r_multiplier", 1.2)
    
    exit_levels = {}
    
    if time_tp_enabled:
        # VWAP止盈
        vwap_price = row.get("vwap", price)
        if row["sig_side"] > 0:
            vwap_tp = max(vwap_price, price + 0.3 * atr)
        else:
            vwap_tp = min(vwap_price, price - 0.3 * atr)
        
        exit_levels["vwap_tp"] = {
            "price": vwap_tp,
            "time_seconds": time_tp_vwap_seconds,
            "level_type": "vwap"
        }
        
        # 1.5R止盈
        r_multiplier = time_tp_15r_multiplier
        if row["sig_side"] > 0:
            r_tp = price + r_multiplier * atr
        else:
            r_tp = price - r_multiplier * atr
        
        exit_levels["r_tp"] = {
            "price": r_tp,
            "time_seconds": time_tp_15r_seconds,
            "level_type": "r_multiplier"
        }
    
    return exit_levels

def check_risk_limits(current_equity: float, params: dict, trade_pnl: float = 0.0) -> bool:
    """
    检查风险限制
    """
    # 单笔交易风险限制
    max_trade_risk = params["risk"].get("max_trade_risk_pct", 0.01)
    max_trade_loss = current_equity * max_trade_risk
    
    if abs(trade_pnl) > max_trade_loss:
        return False
    
    # 日回撤限制
    daily_drawdown_limit = params["risk"].get("daily_drawdown_stop_pct", 0.08)
    max_daily_loss = current_equity * daily_drawdown_limit
    
    # 这里需要跟踪日收益，简化处理
    if trade_pnl < -max_daily_loss:
        return False
    
    return True

def calculate_risk_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    计算风险指标
    """
    if trades_df.empty:
        return {}
    
    # 基本统计
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df["pnl"] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # 收益统计
    total_pnl = trades_df["pnl"].sum()
    avg_pnl = trades_df["pnl"].mean()
    
    # 风险调整收益
    if total_trades > 1:
        pnl_std = trades_df["pnl"].std()
        sharpe_ratio = avg_pnl / pnl_std if pnl_std > 0 else 0.0
    else:
        sharpe_ratio = 0.0
    
    # 最大回撤
    cumulative_pnl = trades_df["pnl"].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max
    max_drawdown = drawdown.min()
    
    # 盈亏比
    winning_pnl = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
    losing_pnl = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
    profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor
    }
