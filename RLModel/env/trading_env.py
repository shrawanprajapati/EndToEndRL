"""
env/trading_env.py  |  Days 5-9  |  M2 — Env Engineer  [CENTRAL FILE]
Every other component feeds into this file.
Implements the full Gymnasium Env interface.

reset() → picks a random 90-day window from train.csv, resets portfolio
step()  → risk checks → rebalance → reward → next obs → termination check

STATE VECTOR (20 features):
  Market (17): returns_1h, rsi_14, macd_line, macd_signal, atr_14,
               bollinger_b, volume_change, log_volume, price_position,
               volatility_ratio, momentum_24h, ema_ratio, volume_spike,
               hour_sin, hour_cos, day_sin, day_cos
  Portfolio(3): current_allocation, unrealised_pnl, hours_since_trade

ACTION SPACE: Box([0], [1]) — target BTC allocation
              0.0 = 100% cash,  1.0 = 100% BTC
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from env.portfolio  import Portfolio
from env.risk_layer import check_stop_loss, check_take_profit, atr_position_cap
from env.reward     import compute_reward

# ── Episode constants ─────────────────────────────────────────────────────────
EPISODE_HOURS = 90 * 24    # each episode covers 90 days of hourly candles
INITIAL_CASH  = 10_000.0   # starting USDT balance
MIN_PORTFOLIO = 5_000.0    # episode ends early if portfolio drops below this

# ── Feature columns — MUST match featured_data.csv exactly ───────────────────
FEATURE_COLS = [
    "returns_1h",       # 1-hour log return — raw price movement signal
    "rsi_14",           # RSI 14 normalised to [0,1] — overbought/oversold
    "macd_line",        # MACD line (z-scored) — trend momentum
    "macd_signal",      # MACD signal line (z-scored) — smoothed trend
    "atr_14",           # ATR normalised by close — volatility in price units
    "bollinger_b",      # Bollinger %B — where price sits in the band [0=low, 1=high]
    "volume_change",    # volume % change (z-scored) — trading activity change
    "log_volume",       # log(1+volume) (z-scored) — reduces volume skew
    "price_position",   # price in 20-period high-low range [0=bottom, 1=top]
    "volatility_ratio", # short-term vol / long-term vol — regime detection
    "momentum_24h",     # 24-hour price momentum — medium-term trend strength
    "ema_ratio",        # price / EMA — how far price is above/below its trend
    "volume_spike",     # abnormal volume flag — signals unusual market activity
    "hour_sin",         # sin(hour/24 * 2π) — captures intraday time patterns
    "hour_cos",         # cos(hour/24 * 2π) — paired with hour_sin for cyclical encoding
    "day_sin",          # sin(day/7 * 2π) — captures day-of-week patterns
    "day_cos",          # cos(day/7 * 2π) — paired with day_sin
]

N_MARKET_FEATS    = len(FEATURE_COLS)              # 17
N_PORTFOLIO_FEATS = 3                              # allocation, pnl, time
OBS_SIZE          = N_MARKET_FEATS + N_PORTFOLIO_FEATS  # 20


class TradingEnvironment(gym.Env):
    """
    Gymnasium-compatible BTC/USDT trading environment.

    USAGE:
        env = TradingEnvironment()
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([0.8]))
    """
    metadata = {"render_modes": []}

    def __init__(self, data_path="data/processed/train.csv"):
        super().__init__()

        # load the CSV produced by feature_engineer.py
        self.df = pd.read_csv(data_path, parse_dates=["timestamp"])
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        # fail fast if feature_engineer.py changed column names
        missing = [c for c in FEATURE_COLS if c not in self.df.columns]
        assert not missing, f"Missing columns in {data_path}: {missing}"

        # observation space: 20 continuous values, roughly in [-3, 3]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        # action space: one continuous value — target BTC allocation [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # internal state — properly initialised in reset()
        self.portfolio         = Portfolio(INITIAL_CASH)
        self.current_step      = 0
        self.episode_end       = 0
        self.peak_value        = INITIAL_CASH
        self.prev_value        = INITIAL_CASH
        self.hours_since_trade = 0
        self.stop_loss_count   = 0
        self.take_profit_count = 0
        self._return_buffer    = []  # last 20 step returns for rolling volatility

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # randomly sample a 90-day window so agent sees different market regimes
        max_start         = len(self.df) - EPISODE_HOURS - 1
        episode_start     = int(self.np_random.integers(0, max_start))
        self.episode_end  = episode_start + EPISODE_HOURS
        self.current_step = episode_start

        self.portfolio.reset(INITIAL_CASH)
        self.peak_value        = INITIAL_CASH
        self.prev_value        = INITIAL_CASH
        self.hours_since_trade = 0
        self.stop_loss_count   = 0
        self.take_profit_count = 0
        self._return_buffer    = []

        return self._build_obs(), {}

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        """
        Execute one hourly candle step.
        ORDER IS STRICT — do not reorder these 5 stages.
        """
        action = float(np.clip(action[0], 0.0, 1.0))

        row   = self.df.iloc[self.current_step]
        price = float(row["close"])
        atr   = float(row["atr_14"])  # normalised ATR for position cap

        # ── Stage 1: Risk override — may force action to 0.0 ─────────────────
        if check_stop_loss(price, self.portfolio.entry_price):
            action = 0.0               # force full exit to stop the bleeding
            self.stop_loss_count += 1

        if check_take_profit(price, self.portfolio.entry_price):
            action = 0.0               # force exit to lock in profit
            self.take_profit_count += 1

        # ── Stage 2: Volatility-based position cap ────────────────────────────
        # high ATR means volatile market — reduce max allowed allocation
        action = atr_position_cap(action, atr)

        # ── Stage 3: Execute the trade ────────────────────────────────────────
        cost = self.portfolio.rebalance(action, price)
        if cost > 0:
            self.hours_since_trade = 0   # reset timer when a trade happens
        else:
            self.hours_since_trade += 1  # count hours of inactivity

        # ── Stage 4: Compute reward ───────────────────────────────────────────
        current_value   = self.portfolio.value(price)
        self.peak_value = max(self.peak_value, current_value)

        # step_return: how much did portfolio grow/shrink this hour
        step_return = (current_value - self.prev_value) / (self.prev_value + 1e-9)
        # drawdown: how far below peak are we right now
        drawdown    = (self.peak_value - current_value) / (self.peak_value + 1e-9)
        # cost as fraction of portfolio — penalises excessive trading
        cost_frac   = cost / (self.prev_value + 1e-9)

        # rolling volatility: std of last 20 returns (measures recent instability)
        self._return_buffer.append(step_return)
        if len(self._return_buffer) > 20:
            self._return_buffer.pop(0)
        rolling_vol = float(np.std(self._return_buffer)) if len(self._return_buffer) > 1 else 0.0

        reward = compute_reward(step_return, drawdown, cost_frac, rolling_vol)

        self.prev_value    = current_value
        self.current_step += 1

        # ── Stage 5: Check termination ────────────────────────────────────────
        terminated = (
            self.current_step >= self.episode_end or  # 90 days completed
            current_value < MIN_PORTFOLIO             # account blown
        )

        info = {
            "portfolio_value":  current_value,
            "btc_allocation":   self.portfolio.btc_fraction(price),
            "stop_loss_count":  self.stop_loss_count,
            "take_profit_count": self.take_profit_count,
        }

        return self._build_obs(), reward, terminated, False, info

    # ── observation builder ───────────────────────────────────────────────────

    def _build_obs(self):
        """
        Builds the 20-dimensional state vector for the PPO policy network.
        First 17: market features from CSV (already normalised by feature_engineer).
        Last  3:  live portfolio state (normalised inline).
        """
        row   = self.df.iloc[self.current_step]
        price = float(row["close"])

        # 17 market features — read directly from the normalised CSV
        market = np.array([float(row[c]) for c in FEATURE_COLS], dtype=np.float32)

        # 3 portfolio features — computed from live portfolio state
        portfolio = np.array([
            self.portfolio.btc_fraction(price),                            # [0, 1]
            (self.portfolio.value(price) - INITIAL_CASH) / INITIAL_CASH,  # unrealised P&L
            min(self.hours_since_trade / 100.0, 1.0),                     # hours idle, capped at 1
        ], dtype=np.float32)

        return np.concatenate([market, portfolio])  # shape: (20,)