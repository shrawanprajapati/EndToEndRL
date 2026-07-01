"""
env/risk_layer.py  |  Days 7-8  |  M2 — Env Engineer
Three pure functions — no classes, no state.
Called inside trading_env.step() BEFORE the agent's action is executed.
Imported by: trading_env.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STOP_LOSS_THRESHOLD = -0.08
TAKE_PROFIT_THRESHOLD = 0.25


def check_stop_loss(price, entry_price, is_short=False, threshold=STOP_LOSS_THRESHOLD):
    """
    Returns True if the position has lost >= threshold% since entry.
    For long: loss occurs if price drops below entry_price.
    For short: loss occurs if price rises above entry_price.
    """
    if entry_price is None or entry_price <= 0:
        return False
    if is_short:
        # short loses when price rises
        pnl = (entry_price - price) / entry_price
    else:
        # long loses when price drops
        pnl = (price - entry_price) / entry_price
    return pnl <= threshold

def check_take_profit(price, entry_price, is_short=False, threshold=TAKE_PROFIT_THRESHOLD):
    """
    Returns True if the position has gained >= threshold% since entry.
    """
    if entry_price is None or entry_price <= 0:
        return False
    if is_short:
        pnl = (entry_price - price) / entry_price
    else:
        pnl = (price - entry_price) / entry_price
    return pnl >= threshold

def check_trailing_stop_loss(price, highest_price, is_short=False, threshold=STOP_LOSS_THRESHOLD):
    """
    Returns True if the price has dropped >= threshold% from its highest watermark since entry.
    For long: triggers if price drops below (highest_price * (1 + threshold)).
    For short: triggers if price rises above (lowest_price * (1 - threshold)). (highest_price is lowest for shorts).
    """
    if highest_price is None or highest_price <= 0:
        return False
    if is_short:
        # For shorts, 'highest_price' stores the lowest price seen
        drawdown = (highest_price - price) / highest_price
    else:
        drawdown = (price - highest_price) / highest_price
    # threshold is negative (e.g. -0.08)
    return drawdown <= threshold


def atr_position_cap(action, atr_value, capital_limit=0.95):
    """
    Reduces the agent's target allocation when market volatility is high.
    Caps both positive (long) and negative (short) allocations.
    """
    max_allowed = capital_limit / (1 + 10 * abs(atr_value))
    if action < 0:
        return float(max(action, -max_allowed))
    else:
        return float(min(action, max_allowed))