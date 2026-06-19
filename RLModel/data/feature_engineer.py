"""
data/feature_engineer.py  |  Day 3  |  M1 — Data Engineer
────────────────────────────────────────────────────────────
READS   : data/raw/btc_usdt_1h.csv
PRODUCES: data/processed/featured_data.csv  (23 columns, 0 NaN)

COLUMNS PRODUCED:
    timestamp, open, high, low, close, volume   ← original OHLCV
    returns_1h      ← 1-hour log return
    rsi_14          ← RSI normalised to [0, 1]
    macd_line       ← MACD line (z-scored)
    macd_signal     ← MACD signal line (z-scored)
    atr_14          ← ATR normalised by close price
    bollinger_b     ← Bollinger %B
    volume_change   ← volume % change (z-scored)
    log_volume      ← log(1+volume) (z-scored)
    price_position  ← price in 20-period high-low range [0,1]
    volatility_ratio← 7-period std / 30-period std (regime detection)
    momentum_24h    ← 24-hour log return (medium-term trend)
    ema_ratio       ← close / EMA50 (how far from trend)
    volume_spike    ← 1 if volume > 2x 20-period average, else 0
    hour_sin        ← sin encoding of hour-of-day
    hour_cos        ← cos encoding of hour-of-day
    day_sin         ← sin encoding of day-of-week
    day_cos         ← cos encoding of day-of-week

RUNS: python data/feature_engineer.py
LIBRARY: pip install ta  (NOT pandas_ta — incompatible with Python 3.14)
"""

import numpy as np
import pandas as pd
import ta                   # pip install ta

RAW_PATH  = "data/raw/btc_usdt_1h.csv"
OUT_PATH  = "data/processed/featured_data.csv"
Z_WINDOW  = 500             # rolling window for z-score normalisation


# ── Step 1: Compute all features ──────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 17 new columns to df.
    Every indicator uses only PAST data — no look-ahead bias.
    All ta library indicators are causal (use only closed candles).
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── 1. returns_1h ─────────────────────────────────────────────────────────
    # log return = log(close_t / close_{t-1})
    # log returns are additive and more stationary than price returns
    # row 0 will be NaN → dropped later in dropna()
    df["returns_1h"] = np.log(close / close.shift(1))

    # ── 2. rsi_14 ─────────────────────────────────────────────────────────────
    # RSI measures momentum: >0.7 = overbought, <0.3 = oversold
    # divided by 100 to normalise to [0, 1]
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi() / 100.0

    # ── 3. macd_line and 4. macd_signal ──────────────────────────────────────
    # MACD = EMA12 - EMA26 → captures trend changes
    # signal = EMA9 of MACD line → smoother version for crossover signals
    # these will be z-scored later (they live on price scale so need normalisation)
    macd_obj          = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd_line"]   = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()

    # ── 5. atr_14 ─────────────────────────────────────────────────────────────
    # Average True Range = average of (high-low, high-prev_close, low-prev_close)
    # divided by close to normalise to a fraction (e.g. 0.02 = 2% volatility)
    atr_raw      = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_14"] = atr_raw / (close + 1e-9)

    # ── 6. bollinger_b ────────────────────────────────────────────────────────
    # %B = (price - lower_band) / (upper_band - lower_band)
    # 0 = price at lower band, 1 = price at upper band, can exceed [0,1] in breakouts
    bb              = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bollinger_b"] = bb.bollinger_pband()

    # ── 7. volume_change ─────────────────────────────────────────────────────
    # % change in volume vs previous hour
    # captures sudden spikes in trading activity
    # will be z-scored later
    df["volume_change"] = vol.pct_change(periods=1)

    # ── 8. log_volume ─────────────────────────────────────────────────────────
    # log(1 + volume) compresses the large right-skew of raw volume
    # will be z-scored later
    df["log_volume"] = np.log1p(vol)

    # ── 9. price_position ────────────────────────────────────────────────────
    # where is current price within the 20-period high-low range?
    # 0 = at the bottom, 1 = at the top → useful range context for agent
    roll_high         = high.rolling(20).max()
    roll_low          = low.rolling(20).min()
    df["price_position"] = (close - roll_low) / (roll_high - roll_low + 1e-9)

    # ── 10. volatility_ratio ─────────────────────────────────────────────────
    # short-term volatility (7h) / long-term volatility (30h)
    # ratio > 1 means recent market is more volatile than usual → regime change signal
    # ratio < 1 means market is calming down
    short_vol            = close.pct_change().rolling(7).std()
    long_vol             = close.pct_change().rolling(30).std()
    df["volatility_ratio"] = short_vol / (long_vol + 1e-9)

    # ── 11. momentum_24h ─────────────────────────────────────────────────────
    # 24-hour log return — where is price compared to yesterday?
    # captures medium-term directional bias, not just 1-hour noise
    df["momentum_24h"] = np.log(close / close.shift(24))

    # ── 12. ema_ratio ────────────────────────────────────────────────────────
    # close / EMA50 — is price above or below its 50-hour trend?
    # > 1 means price is above trend (bullish), < 1 means below (bearish)
    ema50           = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["ema_ratio"] = close / (ema50 + 1e-9)

    # ── 13. volume_spike ─────────────────────────────────────────────────────
    # 1 if current volume > 2x the 20-period average volume, else 0
    # binary flag for unusual trading activity (often precedes big moves)
    vol_mean            = vol.rolling(20).mean()
    df["volume_spike"]  = (vol > 2 * vol_mean).astype(float)

    # ── 14-17. Cyclical time encodings ───────────────────────────────────────
    # Markets have strong intraday and day-of-week patterns (e.g. Asia open, weekend)
    # We use sin/cos pairs so the model understands hour 23 is close to hour 0
    # (a plain integer like "hour=23" wouldn't capture this circular relationship)
    hour                = df["timestamp"].dt.hour
    day_of_week         = df["timestamp"].dt.dayofweek  # 0=Monday, 6=Sunday

    df["hour_sin"] = np.sin(2 * np.pi * hour        / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour        / 24)
    df["day_sin"]  = np.sin(2 * np.pi * day_of_week / 7)
    df["day_cos"]  = np.cos(2 * np.pi * day_of_week / 7)

    return df


# ── Step 2: Z-score normalisation ────────────────────────────────────────────

def zscore_normalise(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Rolling z-score normalisation for each column.
    Uses only the past Z_WINDOW candles → no look-ahead bias.
    Result clipped to [-3, +3] to limit outlier impact on the neural network.

    Formula: z = (x - rolling_mean) / rolling_std  then clip(-3, 3)
    """
    for col in cols:
        roll_mean = df[col].rolling(Z_WINDOW, min_periods=1).mean()
        roll_std  = df[col].rolling(Z_WINDOW, min_periods=1).std().replace(0, 1e-9)
        df[col]   = ((df[col] - roll_mean) / roll_std).clip(-3, 3)
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # load raw OHLCV
    df = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {RAW_PATH}")

    # compute all 17 feature columns
    df = compute_features(df)

    # z-score normalise the columns that are on price/volume scale
    # (rsi, bollinger_b, price_position, volume_spike, hour/day sin/cos are
    #  already bounded — they don't need z-scoring)
    z_cols = [
        "macd_line", "macd_signal",   # on price scale → normalise
        "volume_change", "log_volume", # on volume scale → normalise
        "volatility_ratio",            # ratio, can range widely → normalise
        "momentum_24h",                # log return over 24h → normalise
        "ema_ratio",                   # ratio around 1.0 → normalise
    ]
    df = zscore_normalise(df, z_cols)

    # drop warm-up NaN rows (indicator lookback + z-score window)
    before = len(df)
    df     = df.dropna().reset_index(drop=True)
    print(f"Dropped {before - len(df)} warm-up rows")

    # final check — must be zero NaN before saving
    assert df.isnull().sum().sum() == 0, "NaN values remain — check indicator code"

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH}  |  shape={df.shape}")
    print(f"Columns: {list(df.columns)}")

    # ── optional: save correlation heatmap ────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        feature_cols = [
            "returns_1h", "rsi_14", "macd_line", "macd_signal", "atr_14",
            "bollinger_b", "volume_change", "log_volume", "price_position",
            "volatility_ratio", "momentum_24h", "ema_ratio", "volume_spike",
        ]
        corr = df[feature_cols].corr()

        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
        ax.set_xticks(range(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(feature_cols)))
        ax.set_yticklabels(feature_cols, fontsize=8)
        plt.colorbar(im)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("docs/feature_correlation.png", dpi=120)
        print("Saved → docs/feature_correlation.png")
    except Exception:
        pass   # matplotlib optional, don't block the pipeline