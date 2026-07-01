"""
env/portfolio.py  |  Day 6  |  M2 — Env Engineer
No Gymnasium dependency — pure math only.
Tracks cash (USDT) and BTC, executes trades with fee + slippage.
Imported by: trading_env.py

FIX (this version): the original rebalance() charged fee+slippage in dollars
via `cost`, then ALSO applied slippage a second time through `effective_price`,
then ALSO subtracted `cost` a second time from cash/proceeds. Net effect: every
trade lost roughly 2x its intended fee+slippage, silently, on both buy and sell.
This was the root cause of the consistently negative backtest returns — a
structural drag baked into every rebalance regardless of strategy quality.

Now: SLIPPAGE only ever shows up once, baked into `effective_price`. FEE is
the only thing taken out as a separate dollar `cost`. Cash/BTC movements are
computed once, cleanly, with no double subtraction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FEE      = 0.001   # 0.1% Binance fee on every trade
SLIPPAGE = 0.0005  # 0.05% slippage on market orders
MIN_TRADE_FRACTION = 0.02


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

    # ── core trade method ─────────────────────────────────────────────────────

    def rebalance(self, target_allocation, price, atr=0.0):
        """
        Move portfolio to target_allocation fraction of BTC.
        target=0.8 means 80% BTC, 20% USDT.
        target=-0.5 means 50% Short BTC, 50% USDT cash backing.
        Returns: transaction cost paid in USDT (fed to reward function).
        """
        target_allocation = float(max(-1.0, min(1.0, target_allocation)))

        if price <= 0:
            return 0.0

        total = self.value(price)
        if total <= 0:
            return 0.0

        current = self.btc_fraction(price)
        delta   = target_allocation - current  # positive=buy/cover BTC, negative=sell/short BTC

        # Skip tiny trades to reduce churn without making the policy unresponsive.
        if abs(delta) < MIN_TRADE_FRACTION:
            return 0.0

        # Calculate exact target BTC amount (negative for short)
        target_btc = (target_allocation * total) / price
        btc_to_trade = target_btc - self.btc

        dynamic_slippage = SLIPPAGE + (abs(atr) * 0.05)

        if btc_to_trade > 0:
            effective_price = price * (1 + dynamic_slippage)
            max_affordable_btc = self.cash / (effective_price * (1 + FEE))
            if btc_to_trade > max_affordable_btc:
                btc_to_trade = max(0.0, max_affordable_btc)
                if btc_to_trade <= 0:
                    return 0.0

        old_btc = self.btc
        trade_value = abs(btc_to_trade) * price
        fee_cost    = trade_value * FEE

        if btc_to_trade > 0:  # BUY BTC (Long entry, Long increase, or Short cover)
            effective_price = price * (1 + dynamic_slippage)
            cost_usdt = btc_to_trade * effective_price
            self.btc += btc_to_trade
            self.cash -= (cost_usdt + fee_cost)
            
            # Record trade log
            self.trade_log.append({
                "action": float(target_allocation),
                "price": float(effective_price),
                "cost": float(fee_cost + trade_value * dynamic_slippage),
                "portfolio_value": float(self.value(price))
            })
            if self.btc != 0 and (self.entry_price is None or old_btc <= 0 < self.btc):
                self.entry_price = effective_price

        elif btc_to_trade < 0:  # SELL BTC (Long reduction, Short entry, or Short increase)
            effective_price = price * (1 - dynamic_slippage)
            proceeds_usdt = abs(btc_to_trade) * effective_price
            self.btc += btc_to_trade  # negative values add down
            self.cash += (proceeds_usdt - fee_cost)
            
            # Record trade log
            self.trade_log.append({
                "action": float(target_allocation),
                "price": float(effective_price),
                "cost": float(fee_cost + trade_value * dynamic_slippage),
                "portfolio_value": float(self.value(price))
            })
            if self.btc != 0 and (self.entry_price is None or old_btc >= 0 > self.btc):
                self.entry_price = effective_price

        if abs(self.btc) < 1e-6:
            self.btc = 0.0
            self.entry_price = None

        self.cash = max(0.0, self.cash)
        return fee_cost + trade_value * dynamic_slippage

    def reset(self, initial_cash=10_000.0):
        # called by trading_env.reset() at the start of every episode
        self.cash        = initial_cash
        self.btc         = 0.0
        self.entry_price = None
        self.trade_log   = []
