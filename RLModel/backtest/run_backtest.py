"""
backtest/run_backtest.py
The FINAL, one-shot backtest on the full locked test.csv (the last 6 months).
Collects advanced telemetry (equity, drawdowns, indicators, policy entropy) and saves as JSON.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import glob
import re
import matplotlib.pyplot as plt

# Force Python to recognize the main RLModel folder
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Try loading RecurrentPPO if sb3_contrib is available
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

from env.trading_env import TradingEnvironment, INITIAL_CASH
from backtest import metrics
from backtest.benchmark import run_all_baselines

TEST_DATA = os.path.join(ROOT_DIR, "data", "processed", "test.csv")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
SAVE_DIR = os.path.join(ROOT_DIR, "models", "saved")

def load_matching_model_and_env():
    """
    Finds a model checkpoint in models/saved/ that matches the TradingEnvironment's observation space.
    Prioritizes highest step checkpoints and RecurrentPPO if matching.
    """
    raw_env = TradingEnvironment(data_path=TEST_DATA)
    env_shape = raw_env.observation_space.shape[0]
    print(f"TradingEnvironment observation size: {env_shape} features.")

    # List all zip files in models/saved
    zip_files = glob.glob(os.path.join(SAVE_DIR, "*.zip"))
    
    # Sort zip files to put highest step checkpoints first
    def get_step_count(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'(\d+)_steps', basename)
        if match:
            return int(match.group(1))
        if "200k" in basename:
            return 200000
        if "final" in basename:
            return 9999999
        return 0

    zip_files.sort(key=get_step_count, reverse=True)

    selected_model_path = None
    selected_vecnorm_path = None
    is_recurrent = False
    model = None

    # Load Normalization Options
    vecnorm_options = [
        os.path.join(SAVE_DIR, "vec_normalize.pkl"),
        os.path.join(SAVE_DIR, "vec_normalize_200k.pkl")
    ]
    
    venv = DummyVecEnv([lambda: raw_env])

    for model_path in zip_files:
        basename = os.path.basename(model_path)
        print(f"Checking compatibility of: {basename} ...")
        
        # Test each vecnorm
        for vecnorm_path in vecnorm_options:
            if not os.path.exists(vecnorm_path):
                continue
                
            try:
                temp_venv = DummyVecEnv([lambda: raw_env])
                temp_venv = VecNormalize.load(vecnorm_path, temp_venv)
                temp_venv.training = False
                temp_venv.norm_reward = False
                
                # If vecnorm loads, try loading model
                # 1. Try RecurrentPPO first
                if RecurrentPPO is not None:
                    try:
                        temp_model = RecurrentPPO.load(model_path, env=temp_venv)
                        if temp_model.observation_space.shape[0] == env_shape:
                            model = temp_model
                            selected_model_path = model_path
                            selected_vecnorm_path = vecnorm_path
                            venv = temp_venv
                            is_recurrent = True
                            break
                    except Exception:
                        pass
                
                if model is None:
                    # 2. Try standard PPO
                    temp_model = PPO.load(model_path, env=temp_venv)
                    if temp_model.observation_space.shape[0] == env_shape:
                        model = temp_model
                        selected_model_path = model_path
                        selected_vecnorm_path = vecnorm_path
                        venv = temp_venv
                        is_recurrent = False
                        break
            except Exception as e:
                # Silently catch and try next Normalization or model
                continue
                
        if model is not None:
            break

    if model is None:
        raise ValueError(f"Could not find any saved model and normalization pair matching shape {env_shape}.")

    print(f"\n[LOAD SUCCESS]")
    print(f"Loaded model: {os.path.basename(selected_model_path)}")
    print(f"Loaded normalizer: {os.path.basename(selected_vecnorm_path) if selected_vecnorm_path else 'None'}")
    print(f"Model Type: {'RecurrentPPO' if is_recurrent else 'PPO'}")
    
    return model, venv, raw_env, is_recurrent

def run_agent_backtest(model, venv, raw_env, is_recurrent):
    """
    Runs the agent over the full test dataset, collecting detailed telemetry.
    """
    obs = venv.reset()
    
    # Telemetry storage
    equity_curve = [INITIAL_CASH]
    timestamps = [str(raw_env.df["timestamp"].iloc[0])]
    prices = [float(raw_env.df["close"].iloc[0])]
    actions = [0.0]
    positions = [0.0]
    entropy_curve = [0.0]
    
    # Custom indicator logging
    indicators = {
        "rsi_14": [float(raw_env.df["rsi_14"].iloc[0])],
        "atr_14": [float(raw_env.df["atr_14"].iloc[0])],
        "hmm_regime": [float(raw_env.df["hmm_regime"].iloc[0]) if "hmm_regime" in raw_env.df else 0.0],
        "macd_line": [float(raw_env.df["macd_line"].iloc[0])],
    }

    terminated = False
    step_count = 0

    # Recurrent LSTM states
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    accumulated_trade_log = []

    while not terminated:
        # Accumulate trade log before venv.step to preserve entries
        for trade in raw_env.portfolio.trade_log[len(accumulated_trade_log):]:
            accumulated_trade_log.append(trade)

        # Get action and calculate entropy
        with torch.no_grad():
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True
                )
                episode_starts = np.zeros((1,), dtype=bool)
                
                # Calculate policy entropy for RecurrentPPO
                try:
                    obs_tensor, _ = model.policy.obs_to_tensor(obs)
                    entropy_val = float(model.policy.get_distribution(obs_tensor, lstm_states).entropy().mean().item())
                except Exception:
                    entropy_val = 0.0
            else:
                action, _ = model.predict(obs, deterministic=True)
                
                # Calculate policy entropy for standard PPO
                try:
                    obs_tensor, _ = model.policy.obs_to_tensor(obs)
                    distribution = model.policy.get_distribution(obs_tensor)
                    entropy_val = float(distribution.entropy().mean().item())
                except Exception:
                    entropy_val = 0.0

        obs, reward, done, info = venv.step(action)
        terminated = bool(done[0])
        step_count += 1
        
        # Log telemetry
        portfolio_value = float(info[0]["portfolio_value"])
        equity_curve.append(portfolio_value)
        
        step_idx = min(raw_env.current_step, len(raw_env.df) - 1)
        timestamps.append(str(raw_env.df["timestamp"].iloc[step_idx]))
        prices.append(float(raw_env.df["close"].iloc[step_idx]))
        
        # Log active position and target action
        actions.append(float(action[0][0] if isinstance(action[0], (list, np.ndarray)) else action[0]))
        positions.append(float(raw_env.portfolio.btc_fraction(raw_env.df["close"].iloc[step_idx])))
        entropy_curve.append(entropy_val)
        
        # Indicators
        indicators["rsi_14"].append(float(raw_env.df["rsi_14"].iloc[step_idx]))
        indicators["atr_14"].append(float(raw_env.df["atr_14"].iloc[step_idx]))
        hmm_val = raw_env.df["hmm_regime"].iloc[step_idx] if "hmm_regime" in raw_env.df else 0.0
        indicators["hmm_regime"].append(float(hmm_val) if not pd.isna(hmm_val) else 0.0)
        indicators["macd_line"].append(float(raw_env.df["macd_line"].iloc[step_idx]))

    # Final check on trade log after the loop terminates
    for trade in raw_env.portfolio.trade_log[len(accumulated_trade_log):]:
        accumulated_trade_log.append(trade)
    
    # Build series dataframe to calculate drawdowns
    eq_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = ((peaks - eq_arr) / (peaks + 1e-9)).tolist()

    return {
        "timestamps": timestamps,
        "prices": prices,
        "equity_curve": equity_curve,
        "drawdowns": drawdowns,
        "actions": actions,
        "positions": positions,
        "entropy": entropy_curve,
        "indicators": indicators,
        "trade_log": accumulated_trade_log
    }

def main():
    print("Initializing environment and loading model...")
    try:
        model, venv, raw_env, is_recurrent = load_matching_model_and_env()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("Running simulation and collecting telemetry data...")
    agent_telemetry = run_agent_backtest(model, venv, raw_env, is_recurrent)

    print("Running benchmark strategies...")
    baselines = run_all_baselines(raw_env.df, initial_cash=INITIAL_CASH)

    # Compute metric reports
    agent_equity = agent_telemetry["equity_curve"]
    bh_equity, bh_trades = baselines["buy_and_hold"]
    
    # Calculate performance metrics
    agent_report = metrics.full_report(agent_equity, agent_telemetry["trade_log"], buy_hold_equity_curve=bh_equity)
    
    benchmark_reports = {}
    for name, (eq, trades) in baselines.items():
        benchmark_reports[name] = metrics.full_report(eq, trades, buy_hold_equity_curve=bh_equity)

    # Organize full report
    full_report_data = {
        "summary": {
            "agent": agent_report,
            "benchmarks": benchmark_reports
        },
        "telemetry": {
            "timestamps": agent_telemetry["timestamps"],
            "prices": agent_telemetry["prices"],
            "agent_equity": agent_telemetry["equity_curve"],
            "agent_drawdown": agent_telemetry["drawdowns"],
            "agent_actions": agent_telemetry["actions"],
            "agent_positions": agent_telemetry["positions"],
            "agent_entropy": agent_telemetry["entropy"],
            "indicators": agent_telemetry["indicators"],
            "trade_log": agent_telemetry["trade_log"],
            "baselines": {
                name: eq for name, (eq, _) in baselines.items()
            }
        }
    }

    # Save to JSON
    os.makedirs(SAVE_DIR, exist_ok=True)
    report_json_path = os.path.join(SAVE_DIR, "backtest_report.json")
    with open(report_json_path, "w") as f:
        json.dump(full_report_data, f, indent=2)
    print(f"Saved complete telemetry JSON report -> {report_json_path}")

    # Plot static charts as fallback/reference
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    # 1. Equity Curve comparison plot
    plt.figure(figsize=(12, 6))
    plt.plot(agent_telemetry["equity_curve"], label="RL Agent", linewidth=2)
    for name, (eq, _) in baselines.items():
        plt.plot(eq, label=name, alpha=0.7, linestyle="--")
    plt.title("Equity Curve Comparison")
    plt.legend()
    plt.savefig(os.path.join(DOCS_DIR, "equity_curve.png"), dpi=150)
    plt.close()

    # 2. Drawdown plot
    plt.figure(figsize=(12, 4))
    plt.fill_between(range(len(agent_telemetry["drawdowns"])), [-d * 100 for d in agent_telemetry["drawdowns"]], 0, color="red", alpha=0.4)
    plt.title("Agent Drawdown Curve (%)")
    plt.savefig(os.path.join(DOCS_DIR, "drawdown.png"), dpi=150)
    plt.close()

    print("============== BACKTEST SUCCESS ==============")
    print(f"Agent return      : {agent_report['cumulative_return']*100:+.2f}%")
    print(f"Agent Sharpe      : {agent_report['sharpe']:.2f}")
    print(f"Agent Max Drawdown: {agent_report['max_drawdown']*100:.2f}%")
    print(f"Buy & Hold return : {agent_report['buy_hold_return']*100:+.2f}%")
    print(f"Alpha             : {agent_report['alpha']*100:+.2f}pp")
    print("==============================================")

if __name__ == "__main__":
    main()
