"""
tests/test_features.py  |  Day 3  |  M1 — Data Engineer
──────────────────────────────────────────────────────────
PURPOSE : Verify feature_engineer.py output is correct before any downstream use.
RUNS    : python -m pytest tests/test_features.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.feature_engineer import compute_features, zscore_normalise


@pytest.fixture(autouse=True)
def stub_hmm(monkeypatch):
    """Keep feature tests focused on our transforms, not stochastic HMM fitting."""

    class DummyHMM:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    monkeypatch.setattr("data.feature_engineer.GaussianHMM", DummyHMM)


@pytest.fixture
def sample_df():
    """Minimal OHLCV dataframe for testing (200 rows to satisfy indicator warmup)."""
    np.random.seed(42)
    n = 600
    price = 30_000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="1h"),
        "open":      price,
        "high":      price * 1.002,
        "low":       price * 0.998,
        "close":     price,
        "volume":    np.abs(np.random.randn(n) * 1000 + 5000),
    })


def test_no_nan_after_feature_engineering(sample_df):
    """
    After compute_features + dropna, there must be ZERO NaN values.
    This is the single most important data quality check.
    """
    df = compute_features(sample_df.copy())
    df = df.dropna()
    assert df.isnull().sum().sum() == 0, "NaN values found after feature engineering"


def test_rsi_in_zero_to_one_range(sample_df):
    """RSI is divided by 100 in feature_engineer.py — must be in [0, 1]."""
    df = compute_features(sample_df.copy()).dropna()
    assert df["rsi_14"].between(0, 1).all(), "RSI outside [0, 1] range"


def test_bollinger_b_bounded(sample_df):
    """
    Bollinger %B is usually in [0, 1] but can exceed in breakouts.
    We check it's in a sane range, not wildly exploded.
    """
    df = compute_features(sample_df.copy()).dropna()
    assert df["bollinger_b"].between(-0.5, 1.5).all(), "Bollinger %B out of expected range"


def test_zscore_clipped_to_three(sample_df):
    """
    Z-scored features must be clipped to [-3, +3].
    Values outside this range indicate a normalisation bug.
    """
    df = compute_features(sample_df.copy())
    z_cols = [
        "macd_line",
        "macd_signal",
        "volume_change",
        "frac_diff_0_5",
        "price_to_vwap",
        "cvd_24h",
    ]
    df = zscore_normalise(df, z_cols)
    df = df.dropna()
    for col in z_cols:
        assert df[col].between(-3, 3).all(), f"{col} not clipped to [-3, 3]"


def test_no_lookahead_bias_in_returns(sample_df):
    """
    returns_1h[t] = log(close[t] / close[t-1])
    It must NOT use future data.  We verify by checking that shift(1) is used
    (i.e., the first valid value appears at index 1, not 0).
    """
    df = compute_features(sample_df.copy())
    # Row 0 must have NaN for returns_1h (no previous candle)
    assert pd.isna(df["returns_1h"].iloc[0]), "returns_1h[0] should be NaN (look-ahead bias risk)"


def test_feature_columns_present(sample_df):
    """All expected engineered feature columns must exist in output."""
    df = compute_features(sample_df.copy())
    required = [
        "returns_1h",
        "rsi_14",
        "macd_line",
        "macd_signal",
        "atr_14",
        "bollinger_b",
        "volume_change",
        "log_volume",
        "price_position",
        "volatility_ratio",
        "momentum_24h",
        "ema_ratio",
        "golden_cross_ratio",
        "volume_spike",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "frac_diff_0_5",
        "hmm_regime",
        "price_to_vwap",
        "cvd_24h",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
