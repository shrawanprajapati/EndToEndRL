!pip install ccxt pandas -q
import ccxt
import pandas as pd
import time

# ===== CONFIG =====
exchange = ccxt.kraken()   # accessible from India, no API key needed
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
FILENAME = "BTC_USDT_1H_5years.csv"

YEARS = 5
TOTAL_HOURS = YEARS * 365 * 24

since_ms = exchange.parse8601("2021-01-01T00:00:00Z")
all_ohlcv = []

print("Downloading BTC/USDT hourly data from Kraken...")

while True:
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since_ms, limit=500)
    if not ohlcv:
        break
    all_ohlcv.extend(ohlcv)
    since_ms = ohlcv[-1][0] + 1
    print(f"{len(all_ohlcv):,} candles fetched", end="\r")
    time.sleep(0.5)
    if len(all_ohlcv) >= TOTAL_HOURS:
        break

print(f"\nTotal candles: {len(all_ohlcv):,}")

# Build DataFrame matching your file's exact column names
df = pd.DataFrame(
    all_ohlcv,
    columns=["Open time", "Open", "High", "Low", "Close", "Volume"]
)
df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
df = df.drop_duplicates(subset="Open time").sort_values("Open time")

df.to_csv(FILENAME, index=False)
print(f"Saved {len(df):,} rows to '{FILENAME}'")
print(df.head())
print(df.tail())
