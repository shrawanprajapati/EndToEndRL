import pandas as pd
import numpy as np

# Make copy
train_feat = train_df.copy()

# ==================================================
# 1. Price Features
# ==================================================

train_feat["Return"] = train_feat["Close"].pct_change()

train_feat["Log_Return"] = np.log(
    train_feat["Close"] /
    train_feat["Close"].shift(1)
)

train_feat["High_Low_Range"] = (
    train_feat["High"] - train_feat["Low"]
)

train_feat["Open_Close_Diff"] = (
    train_feat["Close"] - train_feat["Open"]
)

# ==================================================
# 2. Moving Averages
# ==================================================

train_feat["SMA_10"] = (
    train_feat["Close"]
    .rolling(10)
    .mean()
)

train_feat["SMA_20"] = (
    train_feat["Close"]
    .rolling(20)
    .mean()
)

train_feat["SMA_50"] = (
    train_feat["Close"]
    .rolling(50)
    .mean()
)

train_feat["EMA_10"] = (
    train_feat["Close"]
    .ewm(span=10)
    .mean()
)

train_feat["EMA_20"] = (
    train_feat["Close"]
    .ewm(span=20)
    .mean()
)

# ==================================================
# 3. Volatility Features
# ==================================================

train_feat["Volatility_10"] = (
    train_feat["Return"]
    .rolling(10)
    .std()
)

train_feat["Volatility_20"] = (
    train_feat["Return"]
    .rolling(20)
    .std()
)

# ==================================================
# 4. RSI
# ==================================================

delta = train_feat["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

train_feat["RSI_14"] = (
    100 - (100 / (1 + rs))
)

# ==================================================
# 5. MACD
# ==================================================

ema12 = train_feat["Close"].ewm(
    span=12,
    adjust=False
).mean()

ema26 = train_feat["Close"].ewm(
    span=26,
    adjust=False
).mean()

train_feat["MACD"] = ema12 - ema26

train_feat["MACD_Signal"] = (
    train_feat["MACD"]
    .ewm(span=9, adjust=False)
    .mean()
)

train_feat["MACD_Hist"] = (
    train_feat["MACD"] -
    train_feat["MACD_Signal"]
)

# ==================================================
# 6. Bollinger Bands
# ==================================================

bb_mid = (
    train_feat["Close"]
    .rolling(20)
    .mean()
)

bb_std = (
    train_feat["Close"]
    .rolling(20)
    .std()
)

train_feat["BB_Middle"] = bb_mid

train_feat["BB_Upper"] = (
    bb_mid + 2 * bb_std
)

train_feat["BB_Lower"] = (
    bb_mid - 2 * bb_std
)

# ==================================================
# 7. Volume Features
# ==================================================

train_feat["Volume_Change"] = (
    train_feat["Volume"]
    .pct_change()
)

train_feat["Volume_MA_20"] = (
    train_feat["Volume"]
    .rolling(20)
    .mean()
)

# ==================================================
# 8. Lag Features
# ==================================================

train_feat["Lag_Return_1"] = (
    train_feat["Return"]
    .shift(1)
)

train_feat["Lag_Return_2"] = (
    train_feat["Return"]
    .shift(2)
)

train_feat["Lag_Return_3"] = (
    train_feat["Return"]
    .shift(3)
)

# ==================================================
# 9. Remove NaN rows
# ==================================================

train_feat = train_feat.dropna()

# ==================================================
# Final Check
# ==================================================

print("Shape:", train_feat.shape)

print("\nColumns:")
print(train_feat.columns)

train_feat.head()
