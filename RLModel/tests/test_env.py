"""
tests/test_env.py  |  Day 8  |  M4 — Quant Engineer
──────────────────────────────────────────────────────
PURPOSE : Verify critical environment behaviour before RL training begins.
          Most important test: stop-loss triggers at -5%.
RUNS    : python -m pytest tests/test_env.py -v

FIX NOTE (vs original): the mock CSV previously only included a subset of the
required market feature columns, and OBS_SIZE was hardcoded to an outdated
value. trading_env.py now requires all 22 FEATURE_COLS and exposes OBS_SIZE=28
(22 market + 6 portfolio). The mock CSV below matches that contract.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.trading_env import TradingEnvironment, OBS_SIZE
from env.portfolio   import Portfolio
from env.risk_layer  import (
    STOP_LOSS_THRESHOLD,
    TAKE_PROFIT_THRESHOLD,
    atr_position_cap,
    check_stop_loss,
    check_take_profit,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_mock_csv(n: int = 3000, start_price: float = 30_000.0) -> str:
    """
    Write a minimal featured CSV to a temp file.
    Includes ALL columns trading_env.FEATURE_COLS dynamically, plus OHLCV.
    """
    from env.trading_env import FEATURE_COLS
    np.random.seed(0)
    prices = start_price + np.cumsum(np.random.randn(n) * 50)
    df_dict = {
        "timestamp":        pd.date_range("2020-01-01", periods=n, freq="1h"),
        "open":             prices,
        "high":             prices * 1.001,
        "low":              prices * 0.999,
        "close":            prices,
        "volume":           np.abs(np.random.randn(n) * 1000 + 5000),
    }
    # Dynamically fill all FEATURE_COLS to prevent test failures on shape/column checks
    for col in FEATURE_COLS:
        df_dict[col] = np.random.uniform(-1.0, 1.0, size=n)

    df = pd.DataFrame(df_dict)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    return tmp.name


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_observation_shape():
    """reset() must return obs of shape (OBS_SIZE,) = (28,) — 22 market + 6 portfolio."""
    path = make_mock_csv()
    env  = TradingEnvironment(data_path=path)
    obs, _ = env.reset()
    assert obs.shape == (OBS_SIZE,), f"Obs shape {obs.shape} != ({OBS_SIZE},)"
    os.unlink(path)


def test_step_returns_correct_tuple():
    """step() must return (obs, reward, terminated, truncated, info)."""
    path = make_mock_csv()
    env  = TradingEnvironment(data_path=path)
    env.reset()
    result = env.step(np.array([0.5]))
    assert len(result) == 5, "step() must return 5-tuple"
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (OBS_SIZE,)
    assert isinstance(reward, float)
    assert isinstance(terminated, (bool, np.bool_))
    os.unlink(path)


def test_buy_increases_btc():
    """Action=1.0 should increase BTC allocation toward 100%."""
    p = Portfolio(10_000)
    p.rebalance(1.0, price=30_000)
    assert p.btc > 0, "BTC should be > 0 after buy action"
    assert p.cash < 10_000, "Cash should decrease after buying BTC"


def test_sell_clears_btc():
    """Action=0.0 should return allocation to 100% cash."""
    p = Portfolio(10_000)
    p.rebalance(0.8, price=30_000)    # buy
    p.rebalance(0.0, price=30_000)    # sell everything
    assert p.btc < 1e-4, "BTC should be ~0 after full sell"


def test_transaction_cost_deducted():
    """rebalance() must return a positive cost and reduce portfolio value."""
    p     = Portfolio(10_000)
    cost  = p.rebalance(0.8, price=30_000)
    assert cost > 0, "Transaction cost should be positive"
    assert p.value(30_000) < 10_000, "Portfolio value should decrease by transaction cost"


def test_stop_loss_triggers_at_minus_6_percent():
    """THE critical test. A loss beyond STOP_LOSS_THRESHOLD must trigger."""
    entry_price   = 30_000.0
    current_price = entry_price * (1 + STOP_LOSS_THRESHOLD - 0.01)
    assert check_stop_loss(current_price, entry_price) is True, \
        f"Stop-loss must trigger below {STOP_LOSS_THRESHOLD:.0%}"


def test_stop_loss_does_not_trigger_at_minus_3_percent():
    """Losses smaller than STOP_LOSS_THRESHOLD must not trigger the stop-loss."""
    entry_price   = 30_000.0
    current_price = entry_price * (1 + STOP_LOSS_THRESHOLD / 2)
    assert check_stop_loss(current_price, entry_price) is False, \
        f"Stop-loss must not trigger above {STOP_LOSS_THRESHOLD:.0%}"


def test_take_profit_triggers_at_plus_11_percent():
    """Gains beyond TAKE_PROFIT_THRESHOLD must trigger the take-profit."""
    entry_price   = 30_000.0
    current_price = entry_price * (1 + TAKE_PROFIT_THRESHOLD + 0.01)
    assert check_take_profit(current_price, entry_price) is True, \
        f"Take-profit must trigger above {TAKE_PROFIT_THRESHOLD:.0%}"


def test_stop_loss_none_entry_does_not_crash():
    """When no position is held (entry_price=None), stop_loss must return False."""
    assert check_stop_loss(25_000, None) is False


def test_atr_cap_reduces_large_position():
    """High ATR should reduce a large allocation. Uses real atr_position_cap(action, atr_value, capital_limit)."""
    capped = atr_position_cap(action=0.95, atr_value=0.05)
    assert capped < 0.95, "ATR cap should reduce position during high volatility"


def test_episode_terminates():
    """An episode must eventually terminate (done=True)."""
    path = make_mock_csv(n=3000)
    env  = TradingEnvironment(data_path=path)
    env.reset()
    terminated = False
    for _ in range(3000):
        _, _, terminated, _, _ = env.step(np.array([0.5]))
        if terminated:
            break
    assert terminated, "Episode never terminated"
    os.unlink(path)
