import pandas as pd
import numpy as np

# Create copy
test_feat = test_df.copy()

# ==================================================
# 1. Price Features
# ==================================================

test_feat["Return"] = test_feat["Close"].pct_change()

test_feat["Log_Return"] = np.log(
    test_feat["Close"] /
    test_feat["Close"].shift(1)
)

test_feat["High_Low_Range"] = (
    test_feat["High"] -
    test_feat["Low"]
)

# ==================================================
# 2. Trend Features
# ==================================================

test_feat["SMA_20"] = (
    test_feat["Close"]
    .rolling(window=20)
    .mean()
)

test_feat["EMA_20"] = (
    test_feat["Close"]
    .ewm(span=20, adjust=False)
    .mean()
)

# ==================================================
# 3. RSI
# ==================================================

delta = test_feat["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

test_feat["RSI_14"] = (
    100 - (100 / (1 + rs))
)

# ==================================================
# 4. MACD
# ==================================================

ema12 = (
    test_feat["Close"]
    .ewm(span=12, adjust=False)
    .mean()
)

ema26 = (
    test_feat["Close"]
    .ewm(span=26, adjust=False)
    .mean()
)

test_feat["MACD"] = ema12 - ema26

test_feat["MACD_Signal"] = (
    test_feat["MACD"]
    .ewm(span=9, adjust=False)
    .mean()
)

# ==================================================
# 5. Volatility
# ==================================================

test_feat["Volatility_20"] = (
    test_feat["Return"]
    .rolling(20)
    .std()
)

# ==================================================
# 6. Volume Features
# ==================================================

test_feat["Volume_Change"] = (
    test_feat["Volume"]
    .pct_change()
)

test_feat["Volume_MA_20"] = (
    test_feat["Volume"]
    .rolling(20)
    .mean()
)

# ==================================================
# 7. Lag Features
# ==================================================

test_feat["Lag_Return_1"] = (
    test_feat["Return"]
    .shift(1)
)

test_feat["Lag_Return_2"] = (
    test_feat["Return"]
    .shift(2)
)

test_feat["Lag_Return_3"] = (
    test_feat["Return"]
    .shift(3)
)

# ==================================================
# Remove NaN Values
# ==================================================

test_feat = test_feat.dropna()

# ==================================================
# Final Check
# ==================================================

print("Shape:", test_feat.shape)

print("\nMissing Values:")
print(test_feat.isnull().sum())

print("\nColumns:")
print(test_feat.columns)

test_feat.head()

test_feat.to_csv("BTC_USDT_test_features.csv")
print("Test feature dataset saved successfully!")
