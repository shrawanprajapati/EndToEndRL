"""
env/reward.py  |  Day 10  |  M3 — RL Engineer
Single function. Called inside trading_env.step() after every trade.
The reward design is THE most important factor in agent learning quality.
Imported by: trading_env.py

FORMULA:
    reward = step_return
             - lambda_dd  * drawdown         ← penalise losses from peak
             - lambda_tc  * transaction_cost  ← penalise overtrading
             - lambda_vol * rolling_vol       ← penalise high volatility swings

TUNING GUIDE (used by tune_hyperparams.py):
    agent never trades    → lambda_tc too high, reduce it
    agent overtrades      → lambda_tc too low,  increase it
    agent ignores losses  → lambda_dd too low,  increase it
    reward always near 0  → step_return scale too small, check portfolio math
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# default weights — Optuna will find better values in tune_hyperparams.py
LAMBDA_DD  = 0.5   # drawdown penalty
LAMBDA_TC  = 2.0   # transaction cost penalty
LAMBDA_VOL = 0.1   # volatility penalty


def compute_reward(step_return, drawdown, transaction_cost, rolling_vol,
                   lambda_dd=LAMBDA_DD, lambda_tc=LAMBDA_TC, lambda_vol=LAMBDA_VOL):
    """
    Args:
        step_return      : (portfolio_now - portfolio_prev) / portfolio_prev
        drawdown         : (peak_value - portfolio_now) / peak_value  (always >= 0)
        transaction_cost : cost paid this step as fraction of portfolio value
        rolling_vol      : std of last 20 step returns (measures instability)

    Returns:
        reward clipped to [-1.0, 1.0] to prevent exploding PPO gradients
    """
    reward = (
        step_return
        - lambda_dd  * drawdown
        - lambda_tc  * transaction_cost
        - lambda_vol * rolling_vol
    )

    # clip so a single bad step can't destabilise the neural network weights
    return float(max(-1.0, min(1.0, reward)))