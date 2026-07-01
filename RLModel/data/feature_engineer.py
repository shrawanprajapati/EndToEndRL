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
import ta    
from hmmlearn.hmm import GaussianHMM          

RAW_PATH  = "data/raw/btc_usdt_1h.csv"
OUT_PATH  = "data/processed/featured_data.csv"
Z_WINDOW  = 500             # rolling window for z-score normalisation


def frac_diff_d45(series, d=0.45, lookback=50):
    """Computes fractional differentiation (d=0.45) with lookback window."""
    weights = [1.0]
    for k in range(1, lookback):
        weights.append(weights[-1] * (k - 1 - d) / k)
    weights = np.array(weights)
    
    diff_series = []
    arr = series.values
    for i in range(len(arr)):
        if i < lookback:
            diff_series.append(np.nan)
        else:
            val = np.sum(arr[i - lookback + 1 : i + 1][::-1] * weights)
            diff_series.append(val)
    return pd.Series(diff_series, index=series.index)


# ── Step 1: Compute all features ──────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds advanced quantitative features to the dataframe.
    Causal structures (.shift(1)) strictly maintained where appropriate.
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── 1. returns_1h ─────────────────────────────────────────────────────────
    df["returns_1h"] = np.log(close / close.shift(1))

    # ── 2. rsi_14 ─────────────────────────────────────────────────────────────
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi() / 100.0

    # ── 3. macd_line and 4. macd_signal ──────────────────────────────────────
    macd_obj          = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd_line"]   = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()

    # ── 5. atr_14 ─────────────────────────────────────────────────────────────
    atr_raw      = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_14"] = atr_raw / (close + 1e-9)

    # ── 6. bollinger_b ────────────────────────────────────────────────────────
    bb              = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bollinger_b"] = bb.bollinger_pband()

    # ── 7. volume_change ─────────────────────────────────────────────────────
    df["volume_change"] = vol.pct_change(periods=1)

    # ── 8. log_volume ─────────────────────────────────────────────────────────
    df["log_volume"] = np.log1p(vol)

    # ── 9. price_position ────────────────────────────────────────────────────
    roll_high         = high.rolling(20).max()
    roll_low          = low.rolling(20).min()
    df["price_position"] = (close - roll_low) / (roll_high - roll_low + 1e-9)

    # ── 10. volatility_ratio ─────────────────────────────────────────────────
    short_vol            = close.pct_change().rolling(7).std()
    long_vol             = close.pct_change().rolling(30).std()
    df["volatility_ratio"] = short_vol / (long_vol + 1e-9)

    # ── 11. momentum_24h ─────────────────────────────────────────────────────
    df["momentum_24h"] = np.log(close / close.shift(24))

    # ── 12. ema_ratio & 12b. golden_cross_ratio ──────────────────────────────
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["ema_ratio"] = close / (ema50 + 1e-9)
    
    ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    df["golden_cross_ratio"] = ema50 / (ema200 + 1e-9)

    # ── 13. volume_spike ─────────────────────────────────────────────────────
    vol_mean            = vol.rolling(20).mean()
    df["volume_spike"]  = (vol > 2 * vol_mean).astype(float)

    # ── 14-17. Cyclical time encodings ───────────────────────────────────────
    hour                = df["timestamp"].dt.hour
    day_of_week         = df["timestamp"].dt.dayofweek  # 0=Monday, 6=Sunday

    df["hour_sin"] = np.sin(2 * np.pi * hour        / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour        / 24)
    df["day_sin"]  = np.sin(2 * np.pi * day_of_week / 7)
    df["day_cos"]  = np.cos(2 * np.pi * day_of_week / 7)
    
    # ── 18. Fractional Differentiation (d=0.45) ──────────────────────────────
    print("Computing true fractional differentiation (d=0.45) with 50-lag memory...")
    df["frac_diff_0_5"] = frac_diff_d45(close, d=0.45, lookback=50)

    # ── 19. Hidden Markov Model (HMM) Market Regimes & Transition Probs ──────
    print("Fitting Gaussian HMM to detect regimes & transition probabilities...")
    temp_df = df[["returns_1h", "atr_14"]].dropna()
    hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
    hmm_model.fit(temp_df)
    regimes = hmm_model.predict(temp_df)
    df["hmm_regime"] = np.nan
    df.loc[temp_df.index, "hmm_regime"] = regimes
    df["hmm_regime"] = df["hmm_regime"] - 1.0

    # raw transition matrix mapping with mock model fallback
    transmat = getattr(hmm_model, "transmat_", np.eye(3))
    # Ensure transition states are clipped correctly within index range
    regime_probs = [transmat[int(r)] if int(r) >= 0 and int(r) < len(transmat) else transmat[0] for r in regimes]
    regime_probs = np.array(regime_probs)
    df["hmm_prob_0"] = np.nan
    df["hmm_prob_1"] = np.nan
    df["hmm_prob_2"] = np.nan
    df.loc[temp_df.index, "hmm_prob_0"] = regime_probs[:, 0]
    df.loc[temp_df.index, "hmm_prob_1"] = regime_probs[:, 1]
    df.loc[temp_df.index, "hmm_prob_2"] = regime_probs[:, 2]

    # ── 19b. GARCH(1,1) Volatility Forecast Injection ────────────────────────
    print("Fitting GARCH(1,1) conditional variance forecasting...")
    from arch import arch_model
    # returns scaled to % for numerical stability in GARCH fitting
    garch_rets = df["returns_1h"].fillna(0.0) * 100.0
    garch_model = arch_model(garch_rets, vol="Garch", p=1, q=1, dist="normal")
    garch_res = garch_model.fit(disp="off")
    # scale volatility forecast back to original return scale
    df["garch_vol"] = garch_res.conditional_volatility / 100.0
    df["garch_vol_raw"] = garch_res.conditional_volatility / 100.0

    # ── 20. Order Flow Proxies (VWAP & CVD) ──────────────────────────────────
    print("Calculating Institutional Order Flow Proxies...")
    typical_price = (high + low + close) / 3.0
    rolling_vol_24 = vol.rolling(window=24).sum() + 1e-9
    rolling_tp_vol = (typical_price * vol).rolling(window=24).sum()
    
    vwap_24h = rolling_tp_vol / rolling_vol_24
    df["price_to_vwap"] = close / vwap_24h

    shape_factor = ((close - low) - (high - close)) / (high - low + 1e-9)
    cvd_step = shape_factor * vol
    df["cvd_24h"] = cvd_step.rolling(window=24).sum() / rolling_vol_24

    print("Adding rolling third and fourth statistical moments (24h skewness/kurtosis with 5-period MA and shift(1))...")
    rets = close.pct_change()
    raw_skew = rets.rolling(24).skew()
    raw_kurt = rets.rolling(24).kurt()
    df["skew_24h"] = raw_skew.rolling(5).mean().shift(1)
    df["kurt_24h"] = raw_kurt.rolling(5).mean().shift(1)

    print("Computing Intraday Intensity Variance...")
    df["intraday_intensity"] = ((2 * close - high - low) / (high - low + 1e-9)) * vol

    print("Calculating Volatility Compression Ratios (4h ATR / 168h ATR)...")
    atr_4h = ta.volatility.AverageTrueRange(high, low, close, window=4).average_true_range() / (close + 1e-9)
    atr_168h = ta.volatility.AverageTrueRange(high, low, close, window=168).average_true_range() / (close + 1e-9)
    df["vol_compression"] = atr_4h / (atr_168h + 1e-9)

    print("Computing Relative Volume Velocity (RVS)...")
    median_vol_same_hour = []
    for i in range(len(df)):
        if i < 720:
            median_vol_same_hour.append(np.nan)
        else:
            current_hour = df["timestamp"].iloc[i].hour
            past_window = df.iloc[max(0, i-720):i]
            same_hour_vols = past_window[past_window["timestamp"].dt.hour == current_hour]["volume"]
            median_vol_same_hour.append(np.median(same_hour_vols) if len(same_hour_vols) > 0 else df["volume"].iloc[i])
    df["rvs"] = df["volume"] / (pd.Series(median_vol_same_hour, index=df.index) + 1e-9)

    print("Tracking rolling 72h max market drawdown...")
    roll_max_72h = close.rolling(72).max()
    df["drawdown_72h"] = (roll_max_72h - close) / (roll_max_72h + 1e-9)

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
        "golden_cross_ratio",          # ratio, can range widely → normalise
        "frac_diff_0_5",               # fractional diff proxy → normalise
        "price_to_vwap",               # ratio, can range widely → normalise
        "cvd_24h",                     # cumulative volume delta → normalise
        "skew_24h", "kurt_24h",        # moments
        "intraday_intensity",          # volume weighted
        "vol_compression", "rvs",      # ratios
        "drawdown_72h",                # rolling drawdown
        "garch_vol"                    # GARCH volatility forecast
    ]
    df = zscore_normalise(df, z_cols)

    # drop warm-up NaN rows (indicator lookback + z-score window)
    before = len(df)
    df     = df.dropna().reset_index(drop=True)
    print(f"Dropped {before - len(df)} warm-up rows")

    # final check — must be zero NaN before saving
    assert df.isnull().sum().sum() == 0, "NaN values remain — check indicator code"

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}  |  shape={df.shape}")
    print(f"Columns: {list(df.columns)}")

    # ── optional: save correlation heatmap ────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        feature_cols = [
            "returns_1h", "rsi_14", "macd_line", "macd_signal", "atr_14",
            "bollinger_b", "volume_change", "log_volume", "price_position",
            "volatility_ratio", "momentum_24h", "ema_ratio", "volume_spike", "golden_cross_ratio",
            "frac_diff_0_5", "hmm_regime", "price_to_vwap", "cvd_24h",
            "skew_24h", "kurt_24h", "intraday_intensity", "vol_compression", "rvs", "drawdown_72h",
            "garch_vol", "hmm_prob_0", "hmm_prob_1", "hmm_prob_2"
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
        print("Saved -> docs/feature_correlation.png")
    except Exception:
        pass   # matplotlib optional, don't block the pipeline
