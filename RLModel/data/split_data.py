"""
data/split_data.py  |  Day 4  |  M1 — Data Engineer
──────────────────────────────────────────────────────
READS   : data/processed/featured_data.csv
PRODUCES: data/processed/train.csv  (first ~54 months)
          data/processed/test.csv   (last 6 months — LOCKED until Day 14)

RUN ONCE: python data/split_data.py
After this runs, treat test.csv as if it does not exist until Day 14.
"""

import pandas as pd

IN_PATH    = "data/processed/featured_data.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH  = "data/processed/test.csv"

# 6 months × 30 days × 24 hours = 4,320 rows reserved for testing
TEST_HOURS = 6 * 30 * 24


if __name__ == "__main__":
    # load the full featured dataset
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {IN_PATH}")

    # calculate where training ends and testing begins
    # everything before cut → train, everything from cut onward → test
    cut   = len(df) - TEST_HOURS
    train = df.iloc[:cut].reset_index(drop=True)
    test  = df.iloc[cut:].reset_index(drop=True)

    # save both splits
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH,   index=False)

    # print summary so M1 can verify the split looks correct
    print(f"\nTrain : {len(train)} rows")
    print(f"  From : {train['timestamp'].min()}")
    print(f"  To   : {train['timestamp'].max()}")
    print(f"\nTest  : {len(test)} rows")
    print(f"  From : {test['timestamp'].min()}")
    print(f"  To   : {test['timestamp'].max()}")

    print()
    print("⚠  test.csv is now LOCKED.")
    print("   Do not open it for any reason until Day 14 (run_backtest.py).")
    print(f"   Saved → {TRAIN_PATH}")
    print(f"   Saved → {TEST_PATH}")