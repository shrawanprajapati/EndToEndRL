"""
env/risk_layer.py  |  Days 7-8  |  M2 — Env Engineer
Three pure functions — no classes, no state.
Called inside trading_env.step() BEFORE the agent's action is executed.
Imported by: trading_env.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STOP_LOSS_THRESHOLD   = -0.05  # force exit if position loses 5%
TAKE_PROFIT_THRESHOLD =  0.10  # force exit if position gains 10%


def check_stop_loss(price, entry_price, threshold=STOP_LOSS_THRESHOLD):
    """
    Returns True if the open position has lost >= threshold% since entry.
    When True, trading_env forces action=0.0 (full exit).
    entry_price is None when no BTC is held → always returns False.
    """
    if entry_price is None or entry_price <= 0:
        return False
    pnl = (price - entry_price) / entry_price  # e.g. -0.06 means -6% loss
    return pnl <= threshold


def check_take_profit(price, entry_price, threshold=TAKE_PROFIT_THRESHOLD):
    """
    Returns True if the open position has gained >= threshold% since entry.
    When True, trading_env forces action=0.0 (lock in profit).
    """
    if entry_price is None or entry_price <= 0:
        return False
    pnl = (price - entry_price) / entry_price  # e.g. 0.12 means +12% gain
    return pnl >= threshold


def atr_position_cap(action, atr_value, capital_limit=0.95):
    """
    Reduces the agent's target allocation when market volatility is high.
    atr_value is the normalised ATR column from featured_data.csv.

    Logic: higher ATR → smaller max_allowed → agent cannot go all-in during chaos.
    Example: ATR=0.02 → max_allowed = 0.95 / (1 + 0.2) = 0.79 (capped at 79%)
             ATR=0.05 → max_allowed = 0.95 / (1 + 0.5) = 0.63 (capped at 63%)
    """
    max_allowed = capital_limit / (1 + 10 * abs(atr_value))
    return float(min(action, max_allowed))