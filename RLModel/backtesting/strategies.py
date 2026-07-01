"""
strategies.py

All trading strategies live here.

Each strategy returns:

entries
exits

(boolean pandas Series)
"""

import pandas as pd
import vectorbt as vbt


# ======================================================
# SMA
# ======================================================

def sma_strategy(df, fast=20, slow=50):

    close = df["close"]

    fast_ma = vbt.MA.run(close, fast).ma

    slow_ma = vbt.MA.run(close, slow).ma

    entries = fast_ma.vbt.crossed_above(slow_ma)

    exits = fast_ma.vbt.crossed_below(slow_ma)

    return entries, exits


# ======================================================
# EMA
# ======================================================

def ema_strategy(df, fast=12, slow=26):

    close = df["close"]

    fast_ma = vbt.MA.run(
        close,
        fast,
        ewm=True
    ).ma

    slow_ma = vbt.MA.run(
        close,
        slow,
        ewm=True
    ).ma

    entries = fast_ma.vbt.crossed_above(slow_ma)

    exits = fast_ma.vbt.crossed_below(slow_ma)

    return entries, exits


# ======================================================
# RSI
# ======================================================

def rsi_strategy(df,
                 low=0.30,
                 high=0.70):

    rsi = df["rsi_14"]

    entries = (

        (rsi > low)

        &

        (rsi.shift(1) <= low)

    )

    exits = (

        (rsi < high)

        &

        (rsi.shift(1) >= high)

    )

    return entries.fillna(False), exits.fillna(False)


# ======================================================
# MACD
# ======================================================

def macd_strategy(df):

    macd = df["macd_line"]

    signal = df["macd_signal"]

    entries = macd.vbt.crossed_above(signal)

    exits = macd.vbt.crossed_below(signal)

    return entries, exits


# ======================================================
# Bollinger
# ======================================================

def bollinger_strategy(df,
                       window=20,
                       std=2):

    close = df["close"]

    bb = vbt.BBANDS.run(

        close,

        window=window,

        alpha=std

    )

    entries = close.vbt.crossed_above(

        bb.lower

    )

    exits = close.vbt.crossed_below(

        bb.upper

    )

    return entries, exits


# ======================================================
# Buy & Hold
# ======================================================

def buy_hold_strategy(df):

    entries = pd.Series(False, index=df.index)

    exits = pd.Series(False, index=df.index)

    entries.iloc[0] = True

    return entries, exits


# ======================================================
# Registry
# ======================================================

STRATEGIES = {

    "sma": sma_strategy,

    "ema": ema_strategy,

    "rsi": rsi_strategy,

    "macd": macd_strategy,

    "bollinger": bollinger_strategy,

    "buy_hold": buy_hold_strategy

}