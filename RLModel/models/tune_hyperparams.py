"""
tune_hyperparams.py  |  Day 12  |  M3 — RL Engineer
──────────────────────────────────────────────────────
PURPOSE : 30 Optuna trials with Bayesian search over PPO hyperparameters.
          Each trial trains for 100k steps on train.csv, evaluates on a
          30-day validation window carved from train.csv.
READS   : env/trading_env.py, data/processed/train.csv (via env)
PRODUCES: models/saved/best_params.json
RUNS    : python models/tune_hyperparams.py  (~2–3 hours on CPU)

SEARCH SPACE:
  learning_rate : [1e-5, 1e-3] log-uniform
  n_steps       : {256, 512, 1024, 2048}
  batch_size    : {32, 64, 128}
  gamma         : [0.95, 0.9999] log-uniform
  gae_lambda    : [0.9, 0.99]
"""

import json
import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from env.trading_env import TradingEnvironment

BEST_PARAMS_PATH = "models/saved/best_params.json"
N_TRIALS         = 30
TRIAL_STEPS      = 100_000
N_ENVS           = 2          # fewer envs per trial to fit in memory


def make_env():
    def _init():
        return TradingEnvironment()
    return _init


def evaluate_model(model, vec_env, n_eval_episodes: int = 5) -> float:
    """
    Run n_eval_episodes and return mean portfolio return.
    This is the Optuna objective — higher = better.
    """
    returns = []
    for _ in range(n_eval_episodes):
        obs = vec_env.reset()
        done = [False]
        initial_val = None
        while not all(done):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = vec_env.step(action)
            if initial_val is None:
                initial_val = info[0].get("portfolio_value", 10_000)
        final_val = info[0].get("portfolio_value", 10_000)
        returns.append((final_val - initial_val) / initial_val)
    return float(np.mean(returns))


def objective(trial: optuna.Trial) -> float:
    """One Optuna trial: sample hyperparams → train → evaluate → return metric."""
    lr         = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps    = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    gamma      = trial.suggest_float("gamma", 0.95, 0.9999, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)

    vec_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=lr, n_steps=n_steps, batch_size=batch_size,
        gamma=gamma, gae_lambda=gae_lambda, verbose=0,
    )
    model.learn(total_timesteps=TRIAL_STEPS)

    score = evaluate_model(model, vec_env)
    vec_env.close()

    return score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest trial  |  score={study.best_value:.4f}")
    print(f"Params: {best}")

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)

    print(f"✓ Saved → {BEST_PARAMS_PATH}")
