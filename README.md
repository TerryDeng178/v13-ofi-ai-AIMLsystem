# OFI/CVD Microstructure Strategy Framework

This repository contains a **production-ready skeleton** for microstructure-driven short-term alpha on crypto perpetuals,
covering **OFI/CVD momentum & divergence**, **sweep & reclaim**, **liquidation harvesting**, and **spot-perp dislocation**.
It includes data adapters, feature engineering, signal generation, risk, execution (simulated broker), and an event-driven backtester.

> Language: Python 3.10+

## Quick Start

1) Install deps:
```bash
pip install -r requirements.txt
```

2) Run the example backtest using synthetic data:
```bash
python examples/run_backtest.py --config config/params.yaml
```

3) Inspect results in `examples/out/`.

4) Run unit tests:
```bash
python tests/run_tests.py
```

## Modules

- `src/data.py` — Data adapters (CSV + placeholder realtime)
- `src/features.py` — **Multi-level OFI (5-level weighted)**, CVD/BP/VWAP/ATR and z-scores
- `src/regimes.py` — Regime classification (trend/chop/high-vol)
- `src/signals.py` — **Adaptive quantile-based thresholds** for momentum & divergence
- `src/risk.py` — **Time-based take profit levels**, sizing, stops, targets, circuit breakers
- `src/exec.py` — **Enhanced broker with IOC/FOK logic** and slippage budget checking
- `src/strategy.py` — Orchestrator from features → signals → orders
- `src/backtest.py` — **Comprehensive metrics** (IR, Sharpe, MDD) with JSON output
- `src/utils.py` — Utility functions (rolling z-score, EWMA, VWAP, etc.)
- `config/params.yaml` — **Extended strategy parameters** (adaptive thresholds, time TP, execution)
- `examples/run_backtest.py` — Demo runner
- `examples/sample_data.csv` — 1s synthetic tape+L2 for demo
- `examples/out/` — **Backtest results** (trades.json, summary.json)
- `tests/` — **Unit tests** covering utils and features modules
- `prompts/cursor_system.md` — Cursor system prompt (中文)
- `prompts/cursor_task.md` — Cursor task prompt (中文)

## New Features

### 1. Multi-Level OFI Calculation
- **5-level weighted OFI**: Uses top 5 order book levels with decreasing weights
- **Configurable levels**: Adjust `ofi_levels` in config for different market depths
- **Enhanced signal quality**: Better captures order flow dynamics

### 2. Adaptive Quantile-Based Thresholds
- **Dynamic thresholds**: Uses rolling quantiles instead of fixed z-score values
- **Market regime adaptation**: Thresholds adjust to current market conditions
- **Configurable windows**: Set `quantile_window` for lookback period

### 3. Enhanced Execution System
- **IOC/FOK support**: Immediate-or-Cancel and Fill-or-Kill order types
- **Slippage budget checking**: Rejects orders exceeding slippage limits
- **Risk-aware execution**: Prevents trades with excessive execution costs

### 4. Time-Based Take Profit
- **Two-stage TP**: First at VWAP after 5min, then at 1.5R after 10min
- **Configurable timing**: Adjust TP levels and timing in config
- **Risk-adjusted targets**: TP levels scale with trade risk (R-multiples)

### 5. Comprehensive Backtesting
- **Advanced metrics**: Information Ratio, Sharpe ratio, Maximum Drawdown
- **JSON output**: Detailed results saved to `examples/out/`
- **Performance tracking**: Trade-level and portfolio-level analytics

### 6. Unit Testing
- **Full coverage**: Tests for utils and features modules
- **Quality assurance**: Ensures code reliability and correctness
- **Easy validation**: Run `python tests/run_tests.py`

## vNext Strong Constraints (v3.0)

### 7. Hard Constraints Suite
- **Liquidity Pre-check**: Spread ≤ threshold AND depth ≥ rolling median
- **Two-stage Reclaim**: Breakout → next-bar reclaim confirmation for divergence
- **Slippage Budget Control**: Reject orders exceeding slippage budget
- **Min Tick Stop Loss**: Prevent tick-level stop-outs with configurable multiplier

### 8. A/B Shadow Testing
- **A Group (Strong Constraints)**: All vNext constraints enabled
- **B Group (Relaxed)**: Baseline comparison with relaxed parameters
- **KPI Targets**: Win rate ≥30-40% (divergence), ≥45% (momentum), IR ≥ 0

### 9. Bucket Attribution Analysis
- **Multi-dimensional Analysis**: Signal type × OFI z-score × Spread × Depth × Session
- **Positive IR Buckets**: Enable only buckets with positive information ratio
- **Parameter Scaling**: Scale parameters only for positive IR buckets

### 10. Configuration Files
- **params_vnext.yaml**: Strong constraints configuration
- **params_relaxed.yaml**: Relaxed baseline configuration
- **UTF-8 Encoding**: Fixed encoding issues for international compatibility

## Notes

- Backtests include **fees + slippage** and **time-barrier exits**.
- All thresholds expressed in **z-scores or quantiles** for portability.
- **Strategy-data decoupling**: All parameters configurable via YAML.
- Replace CSV adapter with your exchange WS feed to go live.

## License

MIT
