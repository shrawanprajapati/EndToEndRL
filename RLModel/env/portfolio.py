"""
env/portfolio.py  |  Day 6  |  M2 — Env Engineer
No Gymnasium dependency — pure math only.
Tracks cash (USDT) and BTC, executes trades with fee + slippage.
Imported by: trading_env.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FEE      = 0.001   # 0.1% Binance fee on every trade
SLIPPAGE = 0.0005  # 0.05% slippage on market orders


class Portfolio:
    def __init__(self, initial_cash=10_000.0):
        self.cash        = initial_cash  # USDT balance
        self.btc         = 0.0           # BTC held
        self.entry_price = None          # price when we last opened a BTC position
        self.trade_log   = []            # every trade stored here for run_backtest.py

    # ── read-only helpers ─────────────────────────────────────────────────────

    def value(self, price):
        # total portfolio value in USDT at current price
        return self.cash + self.btc * price

    def btc_fraction(self, price):
        # fraction of portfolio currently held as BTC (0.0 to 1.0)
        total = self.value(price)
        return (self.btc * price / total) if total > 0 else 0.0

    # ── core trade method ─────────────────────────────────────────────────────

    def rebalance(self, target_allocation, price):
        """
        Move portfolio to target_allocation fraction of BTC.
        target=0.8 means 80% BTC, 20% USDT.
        Returns: transaction cost paid in USDT (fed to reward function).
        """
        target_allocation = float(max(0.0, min(1.0, target_allocation)))

        total   = self.value(price)
        current = self.btc_fraction(price)
        delta   = target_allocation - current  # positive=buy BTC, negative=sell BTC

        # skip tiny trades under 0.1% — avoids paying fees for dust movements
        if abs(delta) < 0.001:
            return 0.0

        trade_value = abs(delta) * total       # USDT value being traded
        cost        = trade_value * (FEE + SLIPPAGE)
        net_value   = trade_value - cost       # what actually changes hands

        if delta > 0:  # BUY BTC with USDT
            effective_price  = price * (1 + SLIPPAGE)   # we pay slightly more when buying
            self.btc        += net_value / effective_price
            self.cash       -= (trade_value + cost)
            if self.entry_price is None:
                self.entry_price = effective_price       # record where we entered

        else:          # SELL BTC for USDT
            effective_price  = price * (1 - SLIPPAGE)   # we receive slightly less when selling
            btc_sold         = net_value / effective_price
            self.btc         = max(0.0, self.btc - btc_sold)
            self.cash       += (net_value - cost)
            if self.btc < 1e-6:
                self.entry_price = None                  # fully exited, clear entry price

        self.cash = max(0.0, self.cash)  # prevent floating-point negatives

        # log every trade for the backtest layer (run_backtest.py reads trade_log)
        self.trade_log.append({
            "price":           price,
            "action":          delta,
            "cost":            cost,
            "portfolio_value": self.value(price),
        })
        return cost

    def reset(self, initial_cash=10_000.0):
        # called by trading_env.reset() at the start of every episode
        self.cash        = initial_cash
        self.btc         = 0.0
        self.entry_price = None
        self.trade_log   = []