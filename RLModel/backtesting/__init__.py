"""VectorBT strategy backtesting package."""

def get_strategies():
    from .strategies import STRATEGIES

    return STRATEGIES


def run_vectorbt_backtest(*args, **kwargs):
    from .backtest import run_vectorbt_backtest as _run_vectorbt_backtest

    return _run_vectorbt_backtest(*args, **kwargs)


__all__ = ["get_strategies", "run_vectorbt_backtest"]
