# QuantPilot AI — data_collector.py
# Downloads 5 years BTC/USDT 1h OHLCV from Binance via CCXT
# Output: data/raw/btc_usdt_1h.csv

import ccxt
import pandas as pd
from datetime import datetime, timezone
import time, os

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, "raw"), exist_ok=True)
OUTPUT = os.path.join(BASE, "raw", "btc_usdt_1h.csv")

exchange = ccxt.binance({"enableRateLimit": True})

now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
since_ms = now_ms - (5 * 365 * 24 * 60 * 60 * 1000)

all_candles = []
since = since_ms

print("Downloading BTC/USDT 1h data from Binance...")

while True:
    candles = exchange.fetch_ohlcv("BTC/USDT", "1h", since=since, limit=1000)
    if not candles:
        break
    all_candles.extend(candles)
    since = candles[-1][0] + 1
    print(f"  Fetched {len(all_candles)} candles so far...", end="\r")
    time.sleep(0.3)
    if candles[-1][0] >= now_ms - (2 * 3600 * 1000):
        break

df = pd.DataFrame(all_candles, columns=["timestamp","open","high","low","close","volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

df.to_csv(OUTPUT, index=False)
print(f"\nSaved {len(df)} rows to {OUTPUT}")
