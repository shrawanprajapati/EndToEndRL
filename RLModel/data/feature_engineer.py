# QuantPilot AI — feature_engineer.py
# Computes RSI, MACD, ATR, Bollinger %B and other features
# Input:  data/raw/btc_usdt_1h.csv
# Output: data/processed/featured_data.csv

import pandas as pd
import numpy as np
from ta.momentum   import RSIIndicator
from ta.trend      import MACD
from ta.volatility import BollingerBands, AverageTrueRange
import os

BASE   = os.path.dirname(os.path.abspath(__file__))
INPUT  = os.path.join(BASE, "raw",       "btc_usdt_1h.csv")
OUTPUT = os.path.join(BASE, "processed", "featured_data.csv")
os.makedirs(os.path.join(BASE, "processed"), exist_ok=True)

df = pd.read_csv(INPUT, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ── Features ──────────────────────────────────────────────────────────────────
df["returns_1h"]     = df["close"].pct_change()
df["rsi_14"]         = RSIIndicator(df["close"], 14).rsi() / 100

macd = MACD(df["close"])
df["macd_line"]      = macd.macd()
df["macd_signal"]    = macd.macd_signal()

df["atr_14"]         = AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()

bb = BollingerBands(df["close"], 20)
df["bollinger_b"]    = ((df["close"] - bb.bollinger_lband()) /
                        (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, np.nan)).clip(0, 1)

df["volume_change"]  = df["volume"].pct_change().clip(-5, 5)
df["log_volume"]     = np.log1p(df["volume"])

roll_high            = df["high"].rolling(480).max()
roll_low             = df["low"].rolling(480).min()
df["price_position"] = ((df["close"] - roll_low) /
                        (roll_high - roll_low).replace(0, np.nan)).clip(0, 1)

# ── Z-score normalise unbounded features ─────────────────────────────────────
for col in ["returns_1h","macd_line","macd_signal","atr_14","volume_change","log_volume"]:
    m = df[col].rolling(500, min_periods=50).mean()
    s = df[col].rolling(500, min_periods=50).std()
    df[col] = ((df[col] - m) / s.replace(0, np.nan)).clip(-3, 3)

# ── Drop warm-up NaN rows ─────────────────────────────────────────────────────
features = ["returns_1h","rsi_14","macd_line","macd_signal",
            "atr_14","bollinger_b","volume_change","log_volume","price_position"]
df = df.dropna(subset=features).reset_index(drop=True)

df.to_csv(OUTPUT, index=False)
print(f"Saved {len(df)} rows, {len(df.columns)} columns to {OUTPUT}")
print(df[features].describe().round(3))
