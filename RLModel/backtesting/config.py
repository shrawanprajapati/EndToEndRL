"""
config.py
Central configuration for the VectorBT backtesting framework.
"""

from pathlib import Path

# ===============================
# Project Paths
# ===============================
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "backtesting" / "reports"
PLOT_DIR = ROOT_DIR / "backtesting" / "plots"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "train": DATA_DIR / "train.csv",
    "test": DATA_DIR / "test.csv",
    "full": DATA_DIR / "featured_data.csv",
}

# ===============================
# Portfolio Settings
# ===============================
INITIAL_CASH = 10000

FEES = 0.001        # 0.10%

SLIPPAGE = 0.0005   # 0.05%

FREQ = "1h"

POSITION_SIZE = 0.95

# ===============================
# Strategy Parameters
# ===============================

SMA_FAST = 20
SMA_SLOW = 50

EMA_FAST = 12
EMA_SLOW = 26

RSI_LOW = 0.30
RSI_HIGH = 0.70

BB_WINDOW = 20
BB_STD = 2.0

# ===============================
# Optimization Grid
# ===============================

OPT_GRID = {

    "sma_fast": [10,20,30],

    "sma_slow":[50,100,200],

    "ema_fast":[10,12,20],

    "ema_slow":[26,50,100],

    "rsi_low":[0.20,0.25,0.30],

    "rsi_high":[0.70,0.75,0.80],

    "bb_window":[10,20,30],

    "bb_std":[1.5,2.0,2.5]
}