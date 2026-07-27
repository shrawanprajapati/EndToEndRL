
def compute_reward(step_return, drawdown,
                   transaction_cost, rolling_vol,
                   regime="neutral",
                   drawdown_penalty_multiplier=50.0,
                   bearish_bonus_value=0.1,
                   min_vol_floor=1e-4):
    """
    Computes a differential Sharpe-inspired reward signal for the PPO agent.
    
    Args:
        step_return (float): The log return of the portfolio this step.
        drawdown (float): The drawdown from the portfolio's peak value.
        transaction_cost (float): Fee cost as a fraction of portfolio value.
        rolling_vol (float): Volatility of recent negative returns (downside vol).
        regime (str): The current HMM market regime ("bearish" or "neutral").
        drawdown_penalty_multiplier (float): Multiplier for the drawdown penalty.
        bearish_bonus_value (float): The bonus applied for capital preservation in bearish regimes.
        min_vol_floor (float): Minimum value for rolling volatility to prevent division by zero.
        
    Returns:
        float: Bounded continuous reward.
    """
    import math

    # 1. Base Return (Net of costs)
    net_return = step_return - transaction_cost
    
    # 2. Differential Sharpe Base (Return / Downside Volatility)
    # We add a small epsilon to prevent division by zero, using an adjustable floor
    vol_penalty = max(rolling_vol, min_vol_floor)
    risk_adjusted_return = net_return / vol_penalty
    
    # 3. Asymmetric Drawdown Penalty
    # The penalty scales quadratically as drawdown deepens, with an adjustable multiplier
    drawdown_penalty = (drawdown ** 2) * drawdown_penalty_multiplier
    
    # 4. Regime Awareness Bonus
    regime_bonus = 0.0
    if regime == "bearish" and net_return > -1e-5:
        # Heavily reward capital preservation in bear markets with an adjustable bonus
        regime_bonus = bearish_bonus_value
        
    reward = risk_adjusted_return - drawdown_penalty + regime_bonus
    
    # Bound the reward to prevent exploding gradients in PPO
    return float(max(-5.0, min(5.0, reward)))