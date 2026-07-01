"""
backtest.py
Main entry point for the VectorBT backtesting framework + RL Model simulation.
"""
import argparse
import os
import sys
import numpy as np
from pathlib import Path
import pandas as pd
# --- ADDED MISSING IMPORT ---
from backtest import metrics 
# ----------------------------

try:
    from .config import REPORT_DIR, DATASETS
    from .strategies import STRATEGIES
    from .utils import load_data, run_backtest, calculate_metrics, save_all_outputs, plot_equity_curves
except ImportError:
    from config import REPORT_DIR, DATASETS
    from strategies import STRATEGIES
    from utils import load_data, run_backtest, calculate_metrics, save_all_outputs, plot_equity_curves

from stable_baselines3 import PPO
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env.trading_env import TradingEnvironment

def run_rl_model_backtest(mode="test"):
    print("DEBUG: Loading RL Model for backtest...")
    data_path = DATASETS[mode]
    model_path = str(Path(__file__).resolve().parent.parent / "models" / "saved" / "ppo_final")
    
    env = TradingEnvironment(data_path=str(data_path))
    model = PPO.load(model_path, env=env)
    
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]
        
    done = False
    portfolio_values = []
    
    # 3. RUN THE SIMULATION
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if isinstance(info, list): info = info[0]
        portfolio_values.append(float(info.get("portfolio_value", 10000)))
                    
    # 4. CALCULATE METRICS
    equity_curve = np.array(portfolio_values)
    report = metrics.full_report(equity_curve, env.portfolio.trade_log)

    # 5. FORMAT RESULT
    metrics_data = {
        "Strategy": "rl_model",
        "Total Return [%]": float(report.get("cumulative_return", 0.0) * 100),
        "CAGR [%]": float(report.get("cagr", 0.0) * 100),
        "Sharpe Ratio": float(report.get("sharpe", 0.0)),
        "Sortino Ratio": float(report.get("sortino", 0.0)),
        "Calmar Ratio": float(report.get("calmar", 0.0)),
        "Max Drawdown [%]": float(report.get("max_drawdown", 0.0) * 100),
        "Win Rate [%]": float(report.get("win_rate", 0.0)),
        "# Trades": int(report.get("num_trades", 0)),
        "Final Value": float(portfolio_values[-1])
    }

    plot_equity_curves({"rl_model": equity_curve})
    
    return {
        "mode": mode,
        "strategy": "rl_model",
        "strategies": ["rl_model"],
        "report_dir": str(REPORT_DIR),
        "comparison": [metrics_data],
    }

def run_vectorbt_backtest(mode="train", strategy="all"):
    if strategy == "rl_model":
        return run_rl_model_backtest(mode)
        
    df = load_data(mode)
    close = df["close"]
    portfolios = {}
    metrics_list = []

    strategy_names = list(STRATEGIES.keys()) if strategy == "all" else [strategy]

    for name in strategy_names:
        if name not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {name}")
        entries, exits = STRATEGIES[name](df)
        pf = run_backtest(close, entries, exits)
        portfolios[name] = pf
        metrics_list.append(calculate_metrics(name, pf))

    comparison = save_all_outputs(portfolios, metrics_list)
    comparison_list = comparison.reset_index().to_dict(orient="records")
    
    if strategy == "all":
        try:
            rl_result = run_rl_model_backtest(mode)
            comparison_list.extend(rl_result["comparison"])
            strategy_names.append("rl_model")
        except Exception as e:
            print(f"Error running RL model backtest: {e}")

    return {
        "mode": mode,
        "strategy": strategy,
        "strategies": strategy_names,
        "report_dir": str(REPORT_DIR),
        "comparison": comparison_list,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--strategy", default="all")
    args = parser.parse_args()
    result = run_vectorbt_backtest(args.mode, args.strategy)
    for row in result["comparison"]: print(row)

if __name__ == "__main__":
    main()