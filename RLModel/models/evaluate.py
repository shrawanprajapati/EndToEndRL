"""
evaluate.py  |  M3 — RL Engineer
──────────────────────────────────────────────────────
PURPOSE : Load the trained PPO model + its VecNormalize stats and backtest
          it on held-out data (default: data/processed/test.csv — a window
          of price history the agent never saw during training/tuning).
READS   : models/saved/ppo_final.zip
          models/saved/vec_normalize.pkl
          data/processed/test.csv  (via env)
PRODUCES: models/saved/eval_results.json
RUNS    : python models/evaluate.py

WHY THIS MATTERS: the model was trained inside a VecNormalize wrapper, so it
only ever saw rescaled observations. We must load the EXACT same
normalisation stats here, set the wrapper to "not training" mode (so it
stops updating its running stats on test data, which would leak test-set
information back into the policy's input scaling), and disable reward
normalisation (we want real, human-readable portfolio returns here, not the
clipped/rescaled training reward).

ALPHA REPORTING: raw return alone is not a fair signal in trending markets.
A −9% agent return during a −12% bear market is actually good. We therefore
compute a buy-and-hold baseline for every episode window and report
alpha = agent_return − buy_hold_return. Positive alpha means the agent added
value over passive holding, regardless of whether the absolute return is
positive or negative.

The buy-and-hold price is read from `info["price_history"]` — a list of
closing prices for the episode window that TradingEnvironment must expose
(see NOTE below). If your env does not yet expose this key, add it to the
info dict returned by `step()` and `reset()`.
"""

import os
import sys
# This line forces Python to recognize the main RLModel folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import argparse
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.trading_env import TradingEnvironment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH      = os.path.join(ROOT_DIR, "models", "saved", "ppo_final.zip")
VECNORM_PATH    = os.path.join(ROOT_DIR, "models", "saved", "vec_normalize.pkl")
TEST_DATA_PATH  = os.path.join(ROOT_DIR, "data", "processed", "test.csv")
import argparse
RESULTS_PATH    = os.path.join(ROOT_DIR, "models", "saved", "eval_results.json")
N_EPISODES      = 20   # how many random 90-day windows to test on


def make_env():
    return TradingEnvironment(data_path=TEST_DATA_PATH)


def load_model_and_env():
    # DummyVecEnv (single process) is enough for evaluation — no need for SubprocVecEnv speed.
    env = DummyVecEnv([make_env])

    # Load the saved running mean/std and attach it to this fresh env.
    env = VecNormalize.load(VECNORM_PATH, env)
    env.training   = False  # freeze stats — do NOT let test data update them
    env.norm_reward = False  # we want real portfolio returns, not normalised reward

    model = RecurrentPPO.load(MODEL_PATH, env=env)
    return model, env


def run_episode(model, env):
    """Run one full episode deterministically.

    Returns
    -------
    values : list[float]
        Portfolio value at every step.
    prices : list[float]
        Underlying asset close price at every step, used to compute the
        buy-and-hold baseline.  Sourced from info["price"] (the asset close
        price at the current bar).  Falls back to None if the env does not
        expose this key, in which case alpha is skipped for the episode.
    """
def run_episode(model, env):
    obs = env.reset()
    done = [False]
    values = []
    prices = []
    
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    while not done[0]:
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=True
        )
        obs, _, done, info = env.step(action)
        episode_starts = done
        values.append(info[0]["portfolio_value"])
        if "price" in info[0]:
            prices.append(info[0]["price"])
    return values, prices or None


def max_drawdown(values):
    peak = values[0]
    worst = 0.0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        worst = max(worst, dd)
    return worst


def buy_hold_return(prices):
    """Return the simple return of holding the asset for the whole episode."""
    if prices is None or len(prices) < 2:
        return None
    return (prices[-1] - prices[0]) / prices[0]


def compute_sharpe(values, risk_free_rate=0.0):
    """Annualized Sharpe ratio from hourly portfolio values."""
    returns = np.diff(values) / np.array(values[:-1])
    if len(returns) < 2 or np.std(returns) < 1e-12:
        return 0.0
    excess = returns - risk_free_rate / (365 * 24)
    return float(np.mean(excess) / np.std(excess) * np.sqrt(365 * 24))


def compute_sortino(values, risk_free_rate=0.0):
    """Annualized Sortino ratio from hourly portfolio values."""
    returns = np.diff(values) / np.array(values[:-1])
    downside = returns[returns < 0]
    if len(downside) < 2 or np.std(downside) < 1e-12:
        return 0.0
    excess_mean = np.mean(returns) - risk_free_rate / (365 * 24)
    return float(excess_mean / np.std(downside) * np.sqrt(365 * 24))


def compute_calmar(values):
    """Annualized return divided by max drawdown."""
    returns = np.diff(values) / np.array(values[:-1])
    if len(returns) < 1:
        return 0.0
    ann_ret = float(np.mean(returns) * 365 * 24)
    mdd = max_drawdown(values)
    if mdd < 1e-12:
        return 0.0
    return float(ann_ret / mdd)


def compute_omega(values, threshold=0.0):
    """Sum of returns above threshold divided by absolute sum of returns below threshold."""
    returns = np.diff(values) / np.array(values[:-1])
    if len(returns) < 1:
        return 0.0
    excess = returns - threshold / (365 * 24)
    upside = excess[excess > 0]
    downside = excess[excess < 0]
    if len(downside) == 0 or np.sum(np.abs(downside)) < 1e-12:
        return 0.0
    return float(np.sum(upside) / np.sum(np.abs(downside)))


def main():
    global TEST_DATA_PATH, N_EPISODES
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/test.csv", help="Path to evaluation data CSV")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    args = parser.parse_known_args()[0]

    if not os.path.isabs(args.data):
        TEST_DATA_PATH = os.path.join(ROOT_DIR, args.data)
    else:
        TEST_DATA_PATH = args.data

    N_EPISODES = args.episodes
    print(f"[EVALUATION] Loading dataset: {TEST_DATA_PATH}")
    print(f"[EVALUATION] Running {N_EPISODES} episodes...")

    model, env = load_model_and_env()

    episode_returns    = []
    episode_maxdd      = []
    episode_bh_returns = []   # buy-and-hold return for each episode window
    episode_alphas     = []   # agent_return − buy_hold_return
    episode_sharpes    = []
    episode_sortinos   = []
    episode_calmars    = []
    episode_omegas     = []

    for ep in range(N_EPISODES):
        values, prices = run_episode(model, env)
        initial, final = values[0], values[-1]
        agent_ret = (final - initial) / initial
        episode_returns.append(agent_ret)
        episode_maxdd.append(max_drawdown(values))
        episode_sharpes.append(compute_sharpe(values))
        episode_sortinos.append(compute_sortino(values))
        episode_calmars.append(compute_calmar(values))
        episode_omegas.append(compute_omega(values))

        bh_ret = buy_hold_return(prices)
        episode_bh_returns.append(bh_ret)

        if bh_ret is not None:
            alpha = agent_ret - bh_ret
            episode_alphas.append(alpha)
            alpha_str = f"  buy&hold={bh_ret*100:6.2f}%  alpha={alpha*100:+6.2f}%"
        else:
            alpha_str = "  buy&hold=N/A  alpha=N/A"

        print(f"Episode {ep + 1:2d}/{N_EPISODES}  "
              f"return={agent_ret*100:6.2f}%"
              f"{alpha_str}  "
              f"max_drawdown={episode_maxdd[-1]*100:5.2f}%")

    returns_arr = np.array(episode_returns)
    dd_arr      = np.array(episode_maxdd)

    # Alpha stats (only computed when env exposes price info)
    have_alpha   = len(episode_alphas) == N_EPISODES
    alpha_arr    = np.array(episode_alphas) if have_alpha else None
    bh_arr       = np.array([r for r in episode_bh_returns if r is not None])

    print("\n-------- Backtest summary --------")
    print(f"Episodes evaluated        : {N_EPISODES}")
    print(f"Mean return               : {returns_arr.mean()*100:.2f}%")
    print(f"Std of returns            : {returns_arr.std()*100:.2f}%")
    print(f"Best episode              : {returns_arr.max()*100:.2f}%")
    print(f"Worst episode             : {returns_arr.min()*100:.2f}%")
    print(f"Win rate (agent > 0)      : {(returns_arr > 0).mean()*100:.1f}%")
    print(f"Mean max drawdown         : {dd_arr.mean()*100:.2f}%")
    print(f"Worst-case drawdown       : {dd_arr.max()*100:.2f}%")
    print(f"Mean Sharpe ratio         : {np.mean(episode_sharpes):.4f}")
    print(f"Mean Sortino ratio        : {np.mean(episode_sortinos):.4f}")
    print(f"Mean Calmar ratio         : {np.mean(episode_calmars):.4f}")
    print(f"Mean Omega ratio          : {np.mean(episode_omegas):.4f}")

    if have_alpha:
        beat_rate = (alpha_arr > 0).mean() * 100
        print(f"\n-------- Alpha vs buy & hold -----")
        print(f"Mean buy-and-hold return  : {bh_arr.mean()*100:.2f}%")
        print(f"Mean alpha                : {alpha_arr.mean()*100:+.2f}%")
        print(f"Std of alpha              : {alpha_arr.std()*100:.2f}%")
        print(f"Best alpha                : {alpha_arr.max()*100:+.2f}%")
        print(f"Worst alpha               : {alpha_arr.min()*100:+.2f}%")
        print(f"Beat rate (alpha > 0)     : {beat_rate:.1f}%")
        print("  -> Beat rate answers: 'did the agent outperform passive holding?'")
        print("     A positive mean alpha in a down market = the agent is working.")
    else:
        print("\n  [INFO] Alpha not computed — env does not expose info['price'].")
        print("     Add  info['price'] = float(close_price)  to TradingEnvironment.step()")
        print("     to unlock alpha reporting on every future run.")

    # ── Persist results ──────────────────────────────────────────────────────
    result = {
        "n_episodes":          N_EPISODES,
        "mean_return":         float(returns_arr.mean()),
        "std_return":          float(returns_arr.std()),
        "best_return":         float(returns_arr.max()),
        "worst_return":        float(returns_arr.min()),
        "win_rate":            float((returns_arr > 0).mean()),
        "mean_max_drawdown":   float(dd_arr.mean()),
        "worst_max_drawdown":  float(dd_arr.max()),
        "mean_sharpe":         float(np.mean(episode_sharpes)),
        "mean_sortino":        float(np.mean(episode_sortinos)),
        "mean_calmar":         float(np.mean(episode_calmars)),
        "mean_omega":          float(np.mean(episode_omegas)),
        "episode_sharpes":     episode_sharpes,
        "episode_sortinos":    episode_sortinos,
        "episode_calmars":     episode_calmars,
        "episode_omegas":      episode_omegas,
        "episode_returns":     episode_returns,
    }

    if have_alpha:
        result.update({
            "mean_buy_hold_return": float(bh_arr.mean()),
            "mean_alpha":           float(alpha_arr.mean()),
            "std_alpha":            float(alpha_arr.std()),
            "best_alpha":           float(alpha_arr.max()),
            "worst_alpha":          float(alpha_arr.min()),
            "beat_rate":            float((alpha_arr > 0).mean()),
            "episode_buy_hold_returns": episode_bh_returns,
            "episode_alphas":           episode_alphas,
        })

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[SUCCESS] Saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()