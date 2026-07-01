
"""
utils.py
Utility functions for VectorBT backtesting framework.
Compatible with config.py and strategies.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import vectorbt as vbt

try:
    from .config import (
        DATASETS,
        INITIAL_CASH,
        FEES,
        SLIPPAGE,
        FREQ,
        POSITION_SIZE,
        REPORT_DIR,
        PLOT_DIR,
    )
except ImportError:
    from config import (
        DATASETS,
        INITIAL_CASH,
        FEES,
        SLIPPAGE,
        FREQ,
        POSITION_SIZE,
        REPORT_DIR,
        PLOT_DIR,
    )

def load_data(mode="train"):
    if mode not in DATASETS:
        raise ValueError(f"Unknown mode: {mode}")
    path = DATASETS[mode]
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def run_backtest(close, entries, exits):
    return vbt.Portfolio.from_signals(
        close,
        entries=entries.fillna(False),
        exits=exits.fillna(False),
        init_cash=INITIAL_CASH,
        fees=FEES,
        slippage=SLIPPAGE,
        size=POSITION_SIZE,
        size_type="percent",
        freq=FREQ,
    )


def calculate_metrics(name, portfolio):
    trades = portfolio.trades.records_readable

    try:
        profit_factor = portfolio.trades.profit_factor()
    except Exception:
        profit_factor = np.nan

    try:
        win_rate = portfolio.trades.win_rate() * 100
    except Exception:
        if len(trades):
            win_rate = (trades["PnL"] > 0).mean() * 100
        else:
            win_rate = np.nan

    try:
        sortino = portfolio.sortino_ratio()
    except Exception:
        sortino = np.nan

    try:
        calmar = portfolio.calmar_ratio()
    except Exception:
        calmar = np.nan

    try:
        cagr = portfolio.annualized_return() * 100
    except Exception:
        cagr = np.nan

    try:
        volatility = portfolio.annualized_volatility() * 100
    except Exception:
        volatility = np.nan

    return {
        "Strategy": name,
        "Total Return [%]": portfolio.total_return() * 100,
        "CAGR [%]": cagr,
        "Sharpe Ratio": portfolio.sharpe_ratio(),
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Annual Volatility [%]": volatility,
        "Max Drawdown [%]": abs(portfolio.max_drawdown() * 100),
        "Profit Factor": profit_factor,
        "Win Rate [%]": win_rate,
        "# Trades": len(trades),
        "Final Value": portfolio.final_value(),
    }


def save_strategy_comparison(results):
    df = pd.DataFrame(results).replace([np.inf, -np.inf], np.nan).set_index("Strategy").round(3)
    outfile = REPORT_DIR / "strategy_comparison.csv"
    df.to_csv(outfile)
    print(f"Saved: {outfile}")
    return df


def save_trade_log(portfolio, strategy_name):
    outfile = REPORT_DIR / f"{strategy_name}_trade_log.csv"
    portfolio.trades.records_readable.to_csv(outfile, index=False)
    print(f"Saved: {outfile}")


def save_monthly_returns(portfolio, strategy_name):
    returns = portfolio.returns()
    monthly = (1 + returns).resample("ME").prod() - 1
    outfile = REPORT_DIR / f"{strategy_name}_monthly_returns.csv"
    monthly.to_csv(outfile, header=["Monthly Return"])
    print(f"Saved: {outfile}")


def plot_equity_curves(curves):
    df = pd.DataFrame(curves)
    fig = df.vbt.plot(title="Equity Curves")
    outfile = PLOT_DIR / "equity_curves.html"
    fig.write_html(str(outfile))
    print(f"Saved: {outfile}")


def plot_drawdown(portfolio, strategy_name):
    dd = portfolio.drawdown()
    fig = dd.vbt.plot(title=f"{strategy_name} Drawdown")
    outfile = PLOT_DIR / f"{strategy_name}_drawdown.html"
    fig.write_html(str(outfile))
    print(f"Saved: {outfile}")


def save_all_outputs(portfolios, metrics):
    comparison = save_strategy_comparison(metrics)

    curves = {}

    for name, pf in portfolios.items():
        curves[name] = pf.value()
        save_trade_log(pf, name)
        save_monthly_returns(pf, name)
        plot_drawdown(pf, name)

    plot_equity_curves(curves)

    return comparison
