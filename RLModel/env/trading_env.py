"""
env/trading_env.py  |  Days 5-9  |  M2 — Env Engineer  [CENTRAL FILE]
Every other component feeds into this file.
Implements the full Gymnasium Env interface.

reset() → picks a random 100-day window from train.csv, resets portfolio
step()  → risk checks → rebalance → reward → next obs → termination check

STATE VECTOR (28 features):
  Market (22): returns_1h, rsi_14, macd_line, macd_signal, atr_14,
               bollinger_b, volume_change, log_volume, price_position,
               volatility_ratio, momentum_24h, ema_ratio, golden_cross_ratio,
               volume_spike, hour_sin, hour_cos, day_sin, day_cos,
               frac_diff_0_5, hmm_regime, price_to_vwap, cvd_24h
  Portfolio(6): current_allocation, unrealised_pnl, hours_since_trade,
                drawdown, rolling_volatility, recent_return

ACTION SPACE: Box([0], [1]) — target BTC allocation
              0.0 = 100% cash,  1.0 = 100% BTC
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from env.portfolio  import Portfolio
from env.risk_layer import check_stop_loss, check_take_profit, check_trailing_stop_loss, atr_position_cap
from env.reward     import compute_reward

# ── Episode constants ─────────────────────────────────────────────────────────
EPISODE_HOURS = 100 * 24    # each episode covers 100 days of hourly candles
INITIAL_CASH  = 10_000.0   # starting USDT balance
MIN_PORTFOLIO = 7_000.0    # episode ends early if portfolio drops below this

# ── Feature columns — MUST match featured_data.csv exactly ───────────────────
TEMPLATE_FEATURE_COLS = [
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
    "golden_cross_ratio", # EMA50 / EMA200 — classic trend confirmation (ADDED)
    "volume_spike",     # abnormal volume flag — signals unusual market activity
    "hour_sin",         # sin(hour/24 * 2π) — captures intraday time patterns
    "hour_cos",         # cos(hour/24 * 2π) — paired with hour_sin for cyclical encoding
    "day_sin",          # sin(day/7 * 2π) — captures day-of-week patterns
    "day_cos",          # cos(day/7 * 2π) — paired with day_sin
    "frac_diff_0_5",    # Fractional differentiation (ADDED)
    "hmm_regime",       # Hidden Markov Model state (ADDED)
    "price_to_vwap",    # VWAP premium/discount (ADDED)
    "cvd_24h",          # Institutional buy/sell pressure proxy (ADDED)
    "skew_24h",         # return skewness
    "kurt_24h",         # return kurtosis
    "intraday_intensity",# Close position weighted by volume
    "vol_compression",  # ATR ratio
    "rvs",              # relative volume velocity
    "drawdown_72h",     # rolling drawdown context
    "garch_vol",        # GARCH conditional volatility forecast
    "hmm_prob_0",       # HMM transition probability state 0
    "hmm_prob_1",       # HMM transition probability state 1
    "hmm_prob_2"        # HMM transition probability state 2
]
FEATURE_COLS = TEMPLATE_FEATURE_COLS

N_MARKET_FEATS    = len(FEATURE_COLS)
N_PORTFOLIO_FEATS = 6                                   
OBS_SIZE          = N_MARKET_FEATS + N_PORTFOLIO_FEATS


class TradingEnvironment(gym.Env):
    """
    Gymnasium-compatible BTC/USDT trading environment.
    Updated for 38-dimensional state vectors.
    """
    metadata = {"render_modes": []}

    def __init__(self, data_path="data/processed/train.csv", domain_randomize: bool = False):
        super().__init__()

        self.data_path = data_path
        self.domain_randomize = domain_randomize

        # load the CSV produced by feature_engineer.py
        self.df = pd.read_csv(data_path, parse_dates=["timestamp"])
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        # fail fast if feature_engineer.py changed column names
        missing = [c for c in FEATURE_COLS if c not in self.df.columns]
        assert not missing, f"Missing columns in {data_path}: {missing}"

        # observation space: 38 continuous values, roughly in [-3, 3]
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
        self.watermark_price   = 0.0
        self._cached_limits = self._load_risk_limits()
        self._limits_loaded_at = time.time()

    def _get_limits(self):
        if time.time() - self._limits_loaded_at > 30:
            self._cached_limits = self._load_risk_limits()
            self._limits_loaded_at = time.time()
        return self._cached_limits

    def _load_risk_limits(self):
        limits = {
            "max_drawdown_pct": 15.0,
            "max_position_size": 0.8,
            "stop_loss_pct": 8.0,
            "take_profit_pct": 20.0
        }
        limits_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "risk_limits.json")
        if os.path.exists(limits_path):
            try:
                import json
                with open(limits_path, "r") as f:
                    file_limits = json.load(f)
                    limits.update(file_limits)
            except Exception:
                pass
        return limits

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly sample a window so the agent sees different market regimes.
        # Short fixture/test datasets run as one shorter episode instead of crashing.
        max_start = len(self.df) - EPISODE_HOURS - 1
        if max_start > 0:
            episode_start = int(self.np_random.integers(0, max_start))
            self.episode_end = episode_start + EPISODE_HOURS
        else:
            episode_start = 0
            self.episode_end = max(1, len(self.df) - 1)
        
        if self.episode_end > len(self.df) - 1:
            self.episode_end = len(self.df) - 1
            
        self.current_step = episode_start

        self.portfolio.reset(INITIAL_CASH)
        self.peak_value        = INITIAL_CASH
        self.prev_value        = INITIAL_CASH
        self.hours_since_trade = 0
        self.stop_loss_count   = 0
        self.take_profit_count = 0
        self.prev_action       = 0.0
        self.drawdown_duration = 0
        self.watermark_price   = 0.0
        self._return_buffer    = []
        self.recent_return     = 0.0

        return self._build_obs(), {}

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        """
        Execute one hourly candle step.
        ORDER IS STRICT — do not reorder these 5 stages.
        """
        action = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))

        row   = self.df.iloc[self.current_step]
        price = float(row["close"])
        
        # Domain Randomization (injecting ±0.01% price noise during training)
        if self.domain_randomize:
            price = price * (1.0 + self.np_random.normal(0.0, 0.0001))

        atr   = float(row["atr_14"])  # normalised ATR for position cap

        is_short = (self.portfolio.btc < 0)

        # Load dynamic risk limits
        limits = self._get_limits()
        max_position_size = float(limits.get("max_position_size", 0.8))
        sl_pct = float(limits.get("stop_loss_pct", 8.0))
        tp_pct = float(limits.get("take_profit_pct", 20.0))

        sl_threshold = -sl_pct / 100.0
        tp_threshold = tp_pct / 100.0

        # ── Stage 1: Risk override — may force action to 0.0 ─────────────────
        # Update watermark price for trailing stop loss
        if self.portfolio.btc > 0:
            self.watermark_price = max(self.watermark_price, price) if self.watermark_price > 0 else price
        elif self.portfolio.btc < 0:
            self.watermark_price = min(self.watermark_price, price) if self.watermark_price > 0 else price
        else:
            self.watermark_price = 0.0

        if check_trailing_stop_loss(price, self.watermark_price, is_short=is_short, threshold=sl_threshold):
            action = 0.0               # force full exit to stop the bleeding
            self.stop_loss_count += 1

        if check_take_profit(price, self.portfolio.entry_price, is_short=is_short, threshold=tp_threshold):
            action = 0.0               # force exit to lock in profit
            self.take_profit_count += 1

        # Portfolio Health Constraint (GARCH Volatility-Targeting Safety Layer)
        # If GARCH(1,1) forecast > 2.5% hourly volatility, restrict max exposure to 20%
        garch_vol_raw = float(row["garch_vol_raw"]) if "garch_vol_raw" in row.index else 0.0
        if garch_vol_raw > 0.025:
            action = float(np.clip(action, -0.2, 0.2))

        # ── Stage 2: Volatility-based position cap ────────────────────────────
        # high ATR means volatile market — reduce max allowed allocation
        action = atr_position_cap(action, atr)

        # Action clamp based on dynamic max position size limit [0.0, max_position_size]
        action = float(np.clip(action, 0.0, max_position_size))

        # ── Stage 3: Execute the trade ────────────────────────────────────────
        cost = self.portfolio.rebalance(action, price, atr=atr)
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

        # ── Asymmetric Risk: Downside Volatility (Sortino logic) ──────────────
        self._return_buffer.append(step_return)
        if len(self._return_buffer) > 20:
            self._return_buffer.pop(0)
            
        # Only calculate volatility of NEGATIVE returns (downside risk)
        negative_returns = [r for r in self._return_buffer if r < 0]
        downside_vol = float(np.std(negative_returns)) if len(negative_returns) > 1 else 0.0

        # Drawdown duration tracking
        if drawdown > 0.001:
            self.drawdown_duration += 1
        else:
            self.drawdown_duration = 0

        # Map HMM regime to "bearish" or "neutral"
        hmm_regime = float(row["hmm_regime"]) if "hmm_regime" in row.index else 0.0
        regime = "bearish" if hmm_regime < -0.5 else "neutral"

        # Quadratic Churn delta and track action history
        delta_action = float(action - self.prev_action)
        self.prev_action = action

        reward = compute_reward(
            step_return=step_return,
            drawdown=drawdown,
            transaction_cost=cost_frac,
            rolling_vol=downside_vol,
            regime=regime
        )

        self.recent_return  = step_return  
        self.prev_value     = current_value
        self.current_step  += 1            

        # ── Stage 5: Check termination ────────────────────────────────────────
        terminated = (
            self.current_step >= self.episode_end or  # 90 days completed
            current_value < MIN_PORTFOLIO             # account blown
        )

        info = {
            "portfolio_value":   current_value,
            "btc_allocation":    self.portfolio.btc_fraction(price),
            "stop_loss_count":   self.stop_loss_count,
            "take_profit_count": self.take_profit_count,
            "price":             price,   # close price this bar — used by evaluate.py for alpha vs buy-and-hold
            "drawdown":          drawdown,
            "action":            action,
            "position":          self.portfolio.btc_fraction(price),
            "unrealized_pnl":    current_value - 10000.0,
            "reward":            reward,
            "transaction_cost":   cost,
        }

        return self._build_obs(), reward, terminated, False, info

    # ── observation builder ───────────────────────────────────────────────────

    def _build_obs(self):
        """
        Builds the 38-dimensional state vector for the PPO policy network.
        First 32: market features from CSV (already normalised, with tanh compaction on volume).
        Last  6: live portfolio state (normalised inline).
        """
        row   = self.df.iloc[self.current_step]
        price = float(row["close"])

        # 32 market features with tanh compaction to squash large volume anomalies causal-style
        market = []
        for c in FEATURE_COLS:
            val = float(row[c])
            if c in ["volume_change", "log_volume", "intraday_intensity", "rvs"]:
                val = float(np.tanh(val))
            market.append(val)
        market = np.nan_to_num(
            np.array(market, dtype=np.float32),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        market = np.clip(market, -10.0, 10.0)

        # 6 portfolio features — computed from live portfolio state
        current_value = self.portfolio.value(price)

        drawdown = (
            self.peak_value - current_value
        ) / (self.peak_value + 1e-9)

        # Only show the AI its downside risk in the observation array
        negative_returns = [r for r in self._return_buffer if r < 0]
        downside_vol = (
            float(np.std(negative_returns))
            if len(negative_returns) > 1
            else 0.0
        )

        portfolio = np.array([

            # allocation
            self.portfolio.btc_fraction(price),

            # pnl
            (current_value - INITIAL_CASH)
            / INITIAL_CASH,

            # inactivity
            min(self.hours_since_trade / 100.0, 1.0),

            # drawdown
            min(drawdown, 1.0),

            # volatility
            min(downside_vol* 100.0, 1.0),

            # latest return
            np.clip(self.recent_return * 50.0, -1.0, 1.0)

        ], dtype=np.float32)

        return np.concatenate([market, portfolio])  # shape: (38,)
