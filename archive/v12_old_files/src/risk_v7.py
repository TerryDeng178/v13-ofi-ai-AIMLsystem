import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def compute_ultra_optimized_levels(row: pd.Series, params: dict, signal_strength: float = 1.0, quality_score: float = 1.0) -> Tuple[float, float]:
    """
    超优化止盈止损计算 - 目标1:10盈亏比
    """
    atr = float(row["atr"])
    price = float(row["price"])
    
    base_stop = params["risk"]["atr_stop_lo"]
    base_take = params["risk"]["atr_stop_hi"]
    
    # 根据信号强度和质量的综合调整
    strength_multiplier = min(2.0, max(0.5, signal_strength / 2.0))
    quality_multiplier = min(1.5, max(0.8, quality_score))
    
    # 综合调整因子
    combined_multiplier = (strength_multiplier + quality_multiplier) / 2
    
    # 超优化：更紧的止损，更高的止盈
    ultra_stop = base_stop / combined_multiplier
    ultra_take = base_take * combined_multiplier
    
    # 最小tick止损保护
    tick = max(float(row.get("ask1", price) - row.get("bid1", price)), 1e-2)
    min_tick_sl_mult = params["risk"].get("min_tick_sl_mult", 2)
    min_sl = max(min_tick_sl_mult * tick, 1e-2)
    
    atr_sl = max(ultra_stop * atr, min_sl)
    
    if row["sig_side"] > 0:
        sl = price - atr_sl
        tp = price + ultra_take * atr
    else:
        sl = price + atr_sl
        tp = price - ultra_take * atr
    
    return sl, tp

def compute_dynamic_position_sizing(row: pd.Series, params: dict, signal_strength: float = 1.0, quality_score: float = 1.0, recent_performance: Dict = None) -> Tuple[int, float]:
    """
    动态仓位管理 - 根据信号强度和质量调整仓位
    """
    k = params["signals"]["sizing"]["k_ofi"]
    size_max = params["signals"]["sizing"]["size_max_usd"]
    
    # 基础仓位计算
    ofi_z = abs(row.get("ofi_z", 0.0))
    base_scale = min(1.0, max(0.1, ofi_z * k))
    
    # 信号强度调整
    strength_multiplier = min(2.0, max(0.5, signal_strength / 2.0))
    
    # 信号质量调整
    quality_multiplier = min(1.5, max(0.8, quality_score))
    
    # 历史表现调整
    performance_multiplier = 1.0
    if recent_performance:
        win_rate = recent_performance.get("win_rate", 0.5)
        avg_pnl = recent_performance.get("avg_pnl", 0.0)
        
        # 基于胜率调整
        if win_rate > 0.6:
            performance_multiplier *= 1.3  # 高胜率时大幅增加仓位
        elif win_rate > 0.5:
            performance_multiplier *= 1.1  # 中等胜率时适度增加仓位
        elif win_rate < 0.4:
            performance_multiplier *= 0.7  # 低胜率时减少仓位
        
        # 基于平均PnL调整
        if avg_pnl > 0:
            performance_multiplier *= 1.2  # 盈利时增加仓位
        else:
            performance_multiplier *= 0.8  # 亏损时减少仓位
    
    # 最终仓位计算
    final_multiplier = strength_multiplier * quality_multiplier * performance_multiplier
    adjusted_scale = base_scale * final_multiplier
    notional = adjusted_scale * size_max
    side = row["sig_side"]
    
    return side, notional

def compute_risk_adjusted_levels(row: pd.Series, params: dict, market_condition: str = "normal") -> Tuple[float, float]:
    """
    风险调整止盈止损 - 根据市场条件调整
    """
    atr = float(row["atr"])
    price = float(row["price"])
    
    base_stop = params["risk"]["atr_stop_lo"]
    base_take = params["risk"]["atr_stop_hi"]
    
    # 市场条件调整
    if market_condition == "high_volatility":
        # 高波动率市场：更宽止损，更高止盈
        stop_multiplier = 1.3
        take_multiplier = 1.2
    elif market_condition == "low_volatility":
        # 低波动率市场：更紧止损，适中止盈
        stop_multiplier = 0.8
        take_multiplier = 0.9
    else:
        # 正常市场
        stop_multiplier = 1.0
        take_multiplier = 1.0
    
    # 计算调整后的止盈止损
    adjusted_stop = base_stop * stop_multiplier
    adjusted_take = base_take * take_multiplier
    
    # 最小tick止损保护
    tick = max(float(row.get("ask1", price) - row.get("bid1", price)), 1e-2)
    min_tick_sl_mult = params["risk"].get("min_tick_sl_mult", 2)
    min_sl = max(min_tick_sl_mult * tick, 1e-2)
    
    atr_sl = max(adjusted_stop * atr, min_sl)
    
    if row["sig_side"] > 0:
        sl = price - atr_sl
        tp = price + adjusted_take * atr
    else:
        sl = price + atr_sl
        tp = price - adjusted_take * atr
    
    return sl, tp

def compute_time_based_exit_levels_v7(row: pd.Series, params: dict) -> Dict[str, Any]:
    """
    v7时间基础退出水平计算 - 更精细的时间管理
    """
    price = float(row["price"])
    atr = float(row["atr"])
    
    # 时间止盈配置
    time_tp_enabled = params["risk"].get("time_tp_enabled", True)
    time_tp_vwap_seconds = params["risk"].get("time_tp_vwap_seconds", 60)
    time_tp_15r_seconds = params["risk"].get("time_tp_15r_seconds", 120)
    time_tp_15r_multiplier = params["risk"].get("time_tp_15r_multiplier", 1.3)
    
    exit_levels = {}
    
    if time_tp_enabled:
        # VWAP止盈
        vwap_price = row.get("vwap", price)
        if row["sig_side"] > 0:
            vwap_tp = max(vwap_price, price + 0.2 * atr)  # 更保守的VWAP止盈
        else:
            vwap_tp = min(vwap_price, price - 0.2 * atr)
        
        exit_levels["vwap_tp"] = {
            "price": vwap_tp,
            "time_seconds": time_tp_vwap_seconds,
            "level_type": "vwap"
        }
        
        # 1.3R止盈
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

def check_advanced_risk_limits(current_equity: float, params: dict, trade_pnl: float = 0.0, daily_pnl: float = 0.0) -> bool:
    """
    高级风险限制检查
    """
    # 单笔交易风险限制
    max_trade_risk = params["risk"].get("max_trade_risk_pct", 0.01)
    max_trade_loss = current_equity * max_trade_risk
    
    if abs(trade_pnl) > max_trade_loss:
        return False
    
    # 日回撤限制
    daily_drawdown_limit = params["risk"].get("daily_drawdown_stop_pct", 0.08)
    max_daily_loss = current_equity * daily_drawdown_limit
    
    if daily_pnl < -max_daily_loss:
        return False
    
    # 连续亏损限制
    max_consecutive_losses = params["risk"].get("max_consecutive_losses", 5)
    # 这里需要跟踪连续亏损次数，简化处理
    
    return True

def calculate_advanced_risk_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    计算高级风险指标
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
    
    # 连续亏损分析
    consecutive_losses = 0
    max_consecutive_losses = 0
    for pnl in trades_df["pnl"]:
        if pnl < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    
    # 信号质量分析
    avg_signal_strength = trades_df["signal_strength"].mean() if "signal_strength" in trades_df.columns else 0
    avg_quality_score = trades_df["quality_score"].mean() if "quality_score" in trades_df.columns else 0
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "max_consecutive_losses": max_consecutive_losses,
        "avg_signal_strength": avg_signal_strength,
        "avg_quality_score": avg_quality_score
    }
