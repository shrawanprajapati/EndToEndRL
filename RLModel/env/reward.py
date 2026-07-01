"""
env/reward.py — Simplified Reward Function
Includes pure return signal, conditional drawdown thresholds,
transaction cost penalization, and a regime-adaptive cash holding bonus.
"""

def compute_reward(step_return, drawdown, 
                   transaction_cost, rolling_vol,
                   regime="neutral"):
    """
    Computes a differential Sharpe-inspired reward signal for the PPO agent.
    
    Args:
        step_return (float): The log return of the portfolio this step.
        drawdown (float): The drawdown from the portfolio's peak value.
        transaction_cost (float): Fee cost as a fraction of portfolio value.
        rolling_vol (float): Volatility of recent negative returns (downside vol).
        regime (str): The current HMM market regime ("bearish" or "neutral").
        
    Returns:
        float: Bounded continuous reward.
    """
    import math

    # 1. Base Return (Net of costs)
    net_return = step_return - transaction_cost
    
    # 2. Differential Sharpe Base (Return / Downside Volatility)
    # We add a small epsilon to prevent division by zero
    vol_penalty = max(rolling_vol, 1e-4)
    risk_adjusted_return = net_return / vol_penalty
    
    # 3. Asymmetric Drawdown Penalty
    # The penalty scales quadratically as drawdown deepens
    drawdown_penalty = (drawdown ** 2) * 50.0
    
    # 4. Regime Awareness Bonus
    regime_bonus = 0.0
    if regime == "bearish" and net_return > -1e-5:
        # Heavily reward capital preservation in bear markets
        regime_bonus = 0.1
        
    reward = risk_adjusted_return - drawdown_penalty + regime_bonus
    
    # Bound the reward to prevent exploding gradients in PPO
    return float(max(-5.0, min(5.0, reward)))