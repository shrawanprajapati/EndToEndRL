"""
tests/test_backtest.py
Test cases for backtesting metrics calculations.
"""

import numpy as np
import pytest
from backtest import metrics

def test_returns_from_equity():
    equity = [100.0, 110.0, 99.0]
    expected_rets = [0.10, -0.10]
    rets = metrics.returns_from_equity(equity)
    assert len(rets) == 2
    np.testing.assert_allclose(rets, expected_rets, rtol=1e-4)

def test_cumulative_return():
    equity = [100.0, 120.0, 150.0]
    assert pytest.approx(metrics.cumulative_return(equity)) == 0.50

def test_max_drawdown():
    # Peak is 150, drops to 120 -> drawdown is (150-120)/150 = 0.20 (20%)
    equity = [100.0, 150.0, 120.0, 130.0]
    assert pytest.approx(metrics.max_drawdown(equity)) == 0.20

def test_win_rate():
    trade_log = [
        {"action": 1.0, "portfolio_value": 10000.0}, # buy
        {"action": -1.0, "portfolio_value": 11000.0}, # sell (win)
        {"action": 1.0, "portfolio_value": 11000.0}, # buy
        {"action": -1.0, "portfolio_value": 10000.0}, # sell (loss)
    ]
    assert metrics.win_rate(trade_log) == 0.50
