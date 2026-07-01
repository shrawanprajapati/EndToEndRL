"""
backtest/benchmark.py
Baseline trading strategies to compare the RL agent against.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from env.portfolio import Portfolio

def buy_and_hold(df, price_col="close", initial_cash=10_000.0):
    portfolio = Portfolio(initial_cash)
    prices = df[price_col].to_numpy(dtype=np.float64)
    equity_curve = []
    portfolio.rebalance(1.0, prices[0])
    for price in prices:
        equity_curve.append(portfolio.value(price))
    return equity_curve, portfolio.trade_log

def moving_average_crossover(df, price_col="close", fast=12, slow=48, initial_cash=10_000.0):
    portfolio = Portfolio(initial_cash)
    prices = df[price_col].to_numpy(dtype=np.float64)
    fast_ma = pd.Series(prices).rolling(fast, min_periods=1).mean().to_numpy()
    slow_ma = pd.Series(prices).rolling(slow, min_periods=1).mean().to_numpy()
    equity_curve = []
    for i, price in enumerate(prices):
        target = 1.0 if fast_ma[i] > slow_ma[i] else 0.0
        portfolio.rebalance(target, price)
        equity_curve.append(portfolio.value(price))
    return equity_curve, portfolio.trade_log

def random_policy(df, price_col="close", initial_cash=10_000.0, seed=42):
    rng = np.random.default_rng(seed)
    portfolio = Portfolio(initial_cash)
    prices = df[price_col].to_numpy(dtype=np.float64)
    equity_curve = []
    for price in prices:
        target = float(rng.uniform(0.0, 1.0))
        portfolio.rebalance(target, price)
        equity_curve.append(portfolio.value(price))
    return equity_curve, portfolio.trade_log

def all_cash(df, price_col="close", initial_cash=10_000.0):
    equity_curve = [initial_cash for _ in range(len(df))]
    return equity_curve, []

BASELINES = {
    "buy_and_hold": buy_and_hold,
    "ma_crossover": moving_average_crossover,
    "random":       random_policy,
    "all_cash":     all_cash,
}

def run_all_baselines(df, price_col="close", initial_cash=10_000.0):
    return {
        name: fn(df, price_col=price_col, initial_cash=initial_cash)
        for name, fn in BASELINES.items()
    }
