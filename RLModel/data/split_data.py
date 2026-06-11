# QuantPilot AI — split_data.py
# Splits featured data into train (4.5 years) + test (6 months)
# Input:  data/processed/featured_data.csv
# Output: data/processed/train.csv  +  data/processed/test.csv

import pandas as pd
import os

BASE   = os.path.dirname(os.path.abspath(__file__))
INPUT  = os.path.join(BASE, "processed", "featured_data.csv")
TRAIN  = os.path.join(BASE, "processed", "train.csv")
TEST   = os.path.join(BASE, "processed", "test.csv")

df = pd.read_csv(INPUT, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

split_date = df["timestamp"].max() - pd.DateOffset(months=6)

train = df[df["timestamp"] <  split_date].reset_index(drop=True)
test  = df[df["timestamp"] >= split_date].reset_index(drop=True)

train.to_csv(TRAIN, index=False)
test.to_csv(TEST,   index=False)

print(f"Train: {len(train)} rows  ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
print(f"Test:  {len(test)}  rows  ({test['timestamp'].min().date()}  → {test['timestamp'].max().date()})")
print(f"Files saved to: {os.path.join(BASE, 'processed')}")
print("⚠️  test.csv is now LOCKED — do not open until Day 14")
