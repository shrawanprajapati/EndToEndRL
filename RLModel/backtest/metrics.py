"""
backtest/metrics.py
Performance metrics calculations for the trading agent.
"""

import numpy as np

HOURS_PER_YEAR = 24 * 365

def returns_from_equity(equity_curve):
    """Hourly simple returns from an equity curve."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) < 2:
        return np.array([])
    return (eq[1:] - eq[:-1]) / (eq[:-1] + 1e-9)

def cumulative_return(equity_curve):
    """Total return over the whole period, e.g. 0.0982 = +9.82%."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) < 2:
        return 0.0
    return float((eq[-1] - eq[0]) / (eq[0] + 1e-9))

def cagr(equity_curve, periods_per_year=HOURS_PER_YEAR):
    """Compound annual growth rate, annualised from hourly equity curve."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    n = len(eq)
    if n < 2 or eq[0] <= 0:
        return 0.0
    total_return = eq[-1] / eq[0]
    years = n / periods_per_year
    if years <= 0:
        return 0.0
    if total_return <= 0:
        return -1.0
    return float(total_return ** (1.0 / years) - 1.0)

def sharpe_ratio(equity_curve, periods_per_year=HOURS_PER_YEAR, risk_free=0.0):
    """Annualised Sharpe ratio from hourly returns."""
    rets = returns_from_equity(equity_curve)
    if len(rets) < 2:
        return 0.0
    excess = rets - (risk_free / periods_per_year)
    std = np.std(excess, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))

def sortino_ratio(equity_curve, periods_per_year=HOURS_PER_YEAR, risk_free=0.0):
    """Annualised Sortino ratio — only penalises downside volatility."""
    rets = returns_from_equity(equity_curve)
    if len(rets) < 2:
        return 0.0
    excess = rets - (risk_free / periods_per_year)
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std < 1e-12:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))

def max_drawdown(equity_curve):
    """Maximum peak-to-trough decline, e.g. 0.25 = -25% drawdown."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / (peak + 1e-9)
    return float(np.max(dd))

def calmar_ratio(equity_curve, periods_per_year=HOURS_PER_YEAR):
    """CAGR / max drawdown — return per unit of worst-case pain."""
    mdd = max_drawdown(equity_curve)
    if mdd < 1e-9:
        return 0.0
    return float(cagr(equity_curve, periods_per_year) / mdd)

def win_rate(trade_log):
    """Fraction of round-trip trades that were profitable."""
    if not trade_log:
        return 0.0
    wins, total_round_trips = 0, 0
    entry_value = None
    for t in trade_log:
        if t.get("action", 0) > 0:
            if entry_value is None:
                entry_value = t.get("portfolio_value", 10000.0)
        elif t.get("action", 0) < 0:
            if entry_value is not None:
                total_round_trips += 1
                if t.get("portfolio_value", 10000.0) >= entry_value:
                    wins += 1
                entry_value = None
    if total_round_trips == 0:
        return 0.0
    return float(wins / total_round_trips)

def num_trades(trade_log):
    return int(len(trade_log)) if trade_log else 0

def avg_trade_cost(trade_log):
    if not trade_log:
        return 0.0
    costs = [t.get("cost", 0.0) for t in trade_log if "cost" in t]
    return float(np.mean(costs)) if costs else 0.0

def alpha(agent_equity_curve, buy_hold_equity_curve):
    agent_ret = cumulative_return(agent_equity_curve)
    bh_ret = cumulative_return(buy_hold_equity_curve)
    return float(agent_ret - bh_ret)

def full_report(equity_curve, trade_log, buy_hold_equity_curve=None):
    report = {
        "cumulative_return": cumulative_return(equity_curve),
        "cagr":              cagr(equity_curve),
        "sharpe":            sharpe_ratio(equity_curve),
        "sortino":           sortino_ratio(equity_curve),
        "max_drawdown":      max_drawdown(equity_curve),
        "calmar":            calmar_ratio(equity_curve),
        "win_rate":          win_rate(trade_log),
        "num_trades":        num_trades(trade_log),
        "avg_trade_cost":    avg_trade_cost(trade_log),
    }
    if buy_hold_equity_curve is not None:
        bh_ret = cumulative_return(buy_hold_equity_curve)
        report["buy_hold_return"] = bh_ret
        report["alpha"] = report["cumulative_return"] - bh_ret
    return report
