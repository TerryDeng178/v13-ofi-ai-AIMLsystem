import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def compute_ultra_profitable_levels(row: pd.Series, params: dict, signal_strength: float = 1.0, 
                                   quality_score: float = 1.0, intelligence_score: float = 1.0) -> Tuple[float, float]:
    """
    v8 超盈利止盈止损计算 - 目标实现净盈利
    """
    atr = float(row["atr"])
    price = float(row["price"])
    
    base_stop = params["risk"]["atr_stop_lo"]
    base_take = params["risk"]["atr_stop_hi"]
    
    # 根据信号强度、质量和智能评分的综合调整
    strength_multiplier = min(2.0, max(0.5, signal_strength / 2.0))
    quality_multiplier = min(1.5, max(0.8, quality_score))
    intelligence_multiplier = min(1.3, max(0.9, intelligence_score))
    
    # 综合调整因子 - 更激进的止盈止损设置
    combined_multiplier = (strength_multiplier + quality_multiplier + intelligence_multiplier) / 3
    
    # 超盈利设置：更紧的止损，更高的止盈
    ultra_stop = base_stop / combined_multiplier
    ultra_take = base_take * combined_multiplier * 1.2  # 额外20%止盈提升
    
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

def compute_profit_optimized_position_sizing(row: pd.Series, params: dict, signal_strength: float = 1.0, 
                                           quality_score: float = 1.0, intelligence_score: float = 1.0, 
                                           recent_performance: Dict = None) -> Tuple[int, float]:
    """
    v8 盈利优化仓位管理 - 基于多维度评分动态调整仓位
    """
    k = params["sizing"]["k_ofi"]
    size_max = params["sizing"]["size_max_usd"]
    
    # 基础仓位计算
    ofi_z = abs(row.get("ofi_z", 0.0))
    base_scale = min(1.0, max(0.1, ofi_z * k))
    
    # 信号强度调整
    strength_multiplier = min(2.0, max(0.5, signal_strength / 2.0))
    
    # 信号质量调整
    quality_multiplier = min(1.5, max(0.8, quality_score))
    
    # 智能评分调整
    intelligence_multiplier = min(1.3, max(0.9, intelligence_score))
    
    # 历史表现调整
    performance_multiplier = 1.0
    if recent_performance:
        win_rate = recent_performance.get("win_rate", 0.5)
        avg_pnl = recent_performance.get("avg_pnl", 0.0)
        profit_factor = recent_performance.get("profit_factor", 1.0)
        
        # 基于胜率调整
        if win_rate > 0.7:
            performance_multiplier *= 1.4  # 高胜率时大幅增加仓位
        elif win_rate > 0.6:
            performance_multiplier *= 1.2  # 中等胜率时适度增加仓位
        elif win_rate < 0.4:
            performance_multiplier *= 0.6  # 低胜率时大幅减少仓位
        
        # 基于平均PnL调整
        if avg_pnl > 0:
            performance_multiplier *= 1.3  # 盈利时增加仓位
        else:
            performance_multiplier *= 0.7  # 亏损时减少仓位
        
        # 基于盈利因子调整
        if profit_factor > 2.0:
            performance_multiplier *= 1.2  # 高盈利因子时增加仓位
        elif profit_factor < 1.0:
            performance_multiplier *= 0.8  # 低盈利因子时减少仓位
    
    # 最终仓位计算
    final_multiplier = strength_multiplier * quality_multiplier * intelligence_multiplier * performance_multiplier
    adjusted_scale = base_scale * final_multiplier
    notional = adjusted_scale * size_max
    side = row["sig_side"]
    
    return side, notional

def compute_cost_optimized_levels(row: pd.Series, params: dict, expected_costs: Dict = None) -> Tuple[float, float]:
    """
    v8 成本优化止盈止损 - 考虑交易成本优化止盈止损设置
    """
    atr = float(row["atr"])
    price = float(row["price"])
    
    base_stop = params["risk"]["atr_stop_lo"]
    base_take = params["risk"]["atr_stop_hi"]
    
    # 成本调整因子
    cost_adjustment = 1.0
    if expected_costs:
        fee_bps = expected_costs.get("fee_bps", 0.5)
        slippage_bps = expected_costs.get("slippage_bps", 2.0)
        total_cost_bps = fee_bps + slippage_bps
        
        # 根据成本调整止盈止损
        if total_cost_bps > 5.0:  # 高成本
            cost_adjustment = 1.3  # 提高止盈止损要求
        elif total_cost_bps < 2.0:  # 低成本
            cost_adjustment = 0.9  # 降低止盈止损要求
    
    # 计算调整后的止盈止损
    adjusted_stop = base_stop * cost_adjustment
    adjusted_take = base_take * cost_adjustment
    
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

def compute_frequency_optimized_levels(row: pd.Series, params: dict, frequency_score: float = 1.0) -> Tuple[float, float]:
    """
    v8 频率优化止盈止损 - 平衡交易频率和盈利能力
    """
    atr = float(row["atr"])
    price = float(row["price"])
    
    base_stop = params["risk"]["atr_stop_lo"]
    base_take = params["risk"]["atr_stop_hi"]
    
    # 频率调整因子
    frequency_multiplier = min(1.2, max(0.8, frequency_score))
    
    # 计算调整后的止盈止损
    adjusted_stop = base_stop / frequency_multiplier  # 高频率时更紧止损
    adjusted_take = base_take * frequency_multiplier  # 高频率时更高止盈
    
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

def calculate_profitability_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    计算v8盈利能力指标
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
    
    # 成本统计
    total_fees = trades_df["fee"].sum() if "fee" in trades_df.columns else 0.0
    total_slippage = trades_df["slippage"].sum() if "slippage" in trades_df.columns else 0.0
    total_costs = total_fees + total_slippage
    
    # 净收益
    net_pnl = total_pnl - total_costs
    
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
    
    # 成本效率
    cost_efficiency = net_pnl / total_costs if total_costs > 0 else float('inf')
    
    # 盈利能力评分
    profitability_score = calculate_profitability_score(win_rate, profit_factor, cost_efficiency, sharpe_ratio)
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "net_pnl": net_pnl,
        "avg_pnl": avg_pnl,
        "total_costs": total_costs,
        "cost_efficiency": cost_efficiency,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "profitability_score": profitability_score
    }

def calculate_profitability_score(win_rate: float, profit_factor: float, cost_efficiency: float, sharpe_ratio: float) -> float:
    """
    计算盈利能力综合评分
    """
    # 胜率评分 (0-0.3)
    win_rate_score = min(1.0, win_rate / 0.6) * 0.3
    
    # 盈利因子评分 (0-0.3)
    profit_factor_score = min(1.0, profit_factor / 2.0) * 0.3
    
    # 成本效率评分 (0-0.2)
    cost_efficiency_score = min(1.0, cost_efficiency / 5.0) * 0.2
    
    # 夏普比率评分 (0-0.2)
    sharpe_score = min(1.0, max(0, sharpe_ratio) / 2.0) * 0.2
    
    total_score = win_rate_score + profit_factor_score + cost_efficiency_score + sharpe_score
    
    return total_score

def check_profitability_limits(current_equity: float, params: dict, trade_pnl: float = 0.0, 
                              daily_pnl: float = 0.0, consecutive_losses: int = 0) -> bool:
    """
    检查盈利能力限制
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
    if consecutive_losses >= max_consecutive_losses:
        return False
    
    return True
