
import yaml
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
try:
    from .data import load_csv, resample_to_seconds
    from .features import add_feature_block
    from .regimes import classify_regime
    from .signals import gen_signals
    from .strategy import run_strategy
except ImportError:
    # For testing when running as standalone
    from data import load_csv, resample_to_seconds
    from features import add_feature_block
    from regimes import classify_regime
    from signals import gen_signals
    from strategy import run_strategy

def calculate_risk_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> dict:
    """
    Calculate comprehensive risk metrics including IR, Sharpe ratio, and MDD.
    """
    if len(returns) == 0:
        return {
            "information_ratio": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "volatility": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0
        }
    
    # Convert annual risk-free rate to daily (assuming daily returns)
    daily_rf = risk_free_rate / 252
    
    # Basic metrics
    mean_return = returns.mean()
    volatility = returns.std()
    
    # Sharpe ratio
    sharpe_ratio = (mean_return - daily_rf) / volatility if volatility > 0 else 0.0
    
    # Information ratio (using returns as both strategy and benchmark)
    # In practice, you'd compare against a benchmark
    information_ratio = mean_return / volatility if volatility > 0 else 0.0
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Maximum Drawdown Duration
    drawdown_periods = (drawdown < 0).astype(int)
    drawdown_durations = []
    current_duration = 0
    for is_drawdown in drawdown_periods:
        if is_drawdown:
            current_duration += 1
        else:
            if current_duration > 0:
                drawdown_durations.append(current_duration)
                current_duration = 0
    if current_duration > 0:
        drawdown_durations.append(current_duration)
    
    max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0
    
    # Higher moments
    skewness = returns.skew() if len(returns) > 2 else 0.0
    kurtosis = returns.kurtosis() if len(returns) > 2 else 0.0
    
    return {
        "information_ratio": information_ratio,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
        "volatility": volatility,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "mean_return": mean_return
    }

def calculate_trade_metrics(trades: pd.DataFrame) -> dict:
    """
    Calculate detailed trade-level metrics.
    """
    if len(trades) == 0:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_holding_seconds": 0.0,
            "max_holding_seconds": 0.0,
            "min_holding_seconds": 0.0
        }
    
    winning_trades = trades[trades["pnl"] > 0]
    losing_trades = trades[trades["pnl"] < 0]
    
    total_trades = len(trades)
    winning_count = len(winning_trades)
    losing_count = len(losing_trades)
    win_rate = winning_count / total_trades if total_trades > 0 else 0.0
    
    avg_win = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0.0
    
    total_wins = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0.0
    total_losses = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
    
    # Holding time metrics
    holding_seconds = (trades["exit_ts"] - trades["entry_ts"]).dt.total_seconds()
    avg_holding_seconds = holding_seconds.mean()
    max_holding_seconds = holding_seconds.max()
    min_holding_seconds = holding_seconds.min()
    
    return {
        "total_trades": total_trades,
        "winning_trades": winning_count,
        "losing_trades": losing_count,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_holding_seconds": avg_holding_seconds,
        "max_holding_seconds": max_holding_seconds,
        "min_holding_seconds": min_holding_seconds
    }

def run(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    df = load_csv(params["data"]["path"])
    df = resample_to_seconds(df, 1)
    df = add_feature_block(df, params)
    df["regime"] = classify_regime(df)
    df = gen_signals(df, params)

    trades = run_strategy(df, params)

    # Calculate comprehensive metrics
    if len(trades) > 0:
        trades["holding_sec"] = (trades["exit_ts"] - trades["entry_ts"]).dt.total_seconds()
    else:
        trades["holding_sec"] = pd.Series(dtype=float)
    
    # Trade-level metrics
    trade_metrics = calculate_trade_metrics(trades)
    
    # Risk metrics (using daily returns approximation)
    if len(trades) > 0:
        # Create a simple returns series from trade PnL
        # This is a simplified approach - in practice you'd track portfolio value over time
        daily_pnl = trades.groupby(trades["exit_ts"].dt.date)["pnl"].sum()
        daily_returns = daily_pnl / params["backtest"]["initial_equity_usd"]
        risk_metrics = calculate_risk_metrics(daily_returns)
    else:
        risk_metrics = calculate_risk_metrics(pd.Series(dtype=float))
    
    # Comprehensive summary
    summary = {
        "backtest_info": {
            "config_path": config_path,
            "run_timestamp": datetime.now().isoformat(),
            "data_path": params["data"]["path"],
            "initial_equity": params["backtest"]["initial_equity_usd"]
        },
        "trade_metrics": trade_metrics,
        "risk_metrics": risk_metrics,
        "performance_summary": {
            "total_pnl": trades["pnl"].sum() if len(trades) > 0 else 0.0,
            "total_fees": trades["fee"].sum() if len(trades) > 0 else 0.0,
            "net_pnl": (trades["pnl"] - trades["fee"]).sum() if len(trades) > 0 else 0.0,
            "avg_pnl_per_trade": trades["pnl"].mean() if len(trades) > 0 else 0.0,
            "max_win": trades["pnl"].max() if len(trades) > 0 else 0.0,
            "max_loss": trades["pnl"].min() if len(trades) > 0 else 0.0
        }
    }
    
    # Save detailed results to JSON
    output_dir = "examples/out"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert datetime columns to strings for JSON serialization
    trades_json = trades.copy()
    for col in trades_json.select_dtypes(include=['datetime64']).columns:
        trades_json[col] = trades_json[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save trades data
    trades_json.to_json(f"{output_dir}/trades.json", orient="records", indent=2)
    
    # 扩展回测指标 (v3)
    if len(trades) > 1:
        rets = trades["pnl"] / trades["qty_usd"].replace(0, np.nan).abs()
        rets = rets.dropna().values
        sharpe = float((np.mean(rets) / (np.std(rets)+1e-9)) * np.sqrt(365*24*60*60)) if len(rets)>1 else 0.0
    else:
        sharpe = 0.0
    curve = trades["pnl"].cumsum() if len(trades) else pd.Series(dtype=float)
    peak = np.maximum.accumulate(curve.fillna(0)) if len(curve) else curve
    mdd = float((curve - peak).min()) if len(curve) else 0.0
    cost = float(abs(trades.get("pnl",0)).sum() - trades["pnl"].sum()) if len(trades) else 0.0  # 近似
    summary.update({"sharpe_approx": sharpe, "mdd": mdd, "cost_est": cost})

    # Save summary
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Backtest completed. Results saved to {output_dir}/")
    print(f"Total trades: {trade_metrics['total_trades']}")
    print(f"Win rate: {trade_metrics['win_rate']:.2%}")
    print(f"Total PnL: ${summary['performance_summary']['total_pnl']:.2f}")
    print(f"Sharpe ratio: {risk_metrics['sharpe_ratio']:.3f}")
    print(f"Max drawdown: {risk_metrics['max_drawdown']:.2%}")
    print(f"Approximate Sharpe: {sharpe:.3f}")
    print(f"Maximum Drawdown: {mdd:.2f}")
    
    return trades, summary
