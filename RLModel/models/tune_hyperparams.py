"""
tune_hyperparams.py  |  Day 12  |  M3 — RL Engineer
──────────────────────────────────────────────────────
PURPOSE : 30 Optuna trials with Bayesian search over PPO hyperparameters.
          Each trial trains for 100k steps on train.csv, evaluates on a
          30-day validation window carved from train.csv.
READS   : env/trading_env.py, data/processed/train.csv (via env)
PRODUCES: models/saved/best_params.json
RUNS    : PYTHONPATH=. python models/tune_hyperparams.py

SEARCH SPACE:
  learning_rate : [1e-5, 1e-3] log-uniform
  n_steps       : {256, 512, 1024, 2048}
  batch_size    : {32, 64, 128}
  gamma         : [0.95, 0.9999] log-uniform
  gae_lambda    : [0.9, 0.99]

macOS FIX: SubprocVecEnv deadlocks on macOS due to the 'spawn' multiprocessing
method. Replaced with DummyVecEnv which runs in the same process — no deadlocks.

FIX (this version), two bugs that were corrupting every trial's score:
  1. make_env() called TradingEnvironment() with no data_path, silently
     defaulting away from train.csv used everywhere else in the pipeline.
     Now explicit, matching train_ppo.py / train_final.py.
  2. evaluate_model() scored trials using the SAME VecNormalize wrapper that
     was still in training mode — so the running mean/std kept updating
     *during* "evaluation," leaking eval-window statistics into the obs
     scaling and inflating every trial's score. Now we freeze stats
     (training=False, norm_reward=False) before scoring, exactly like
     evaluate.py does at the end of the pipeline.
"""

import os
import sys
import json
import numpy as np

# This line forces Python to recognize the main RLModel folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env.trading_env import TradingEnvironment

SAVE_DIR          = "models/saved/"
BEST_PARAMS_PATH  = os.path.join(SAVE_DIR, "best_params.json")
N_TRIALS          = 4
TRIAL_STEPS       = 30_000
N_ENVS            = 1
TRAIN_DATA_PATH   = "data/processed/train.csv"


def make_env():
    def _init():
        return TradingEnvironment(data_path=TRAIN_DATA_PATH)
    return _init


def evaluate_model(model, vec_env, n_eval_episodes: int = 5) -> float:
    """
    Run n_eval_episodes and return mean portfolio return.
    """
    returns = []
    for _ in range(n_eval_episodes):
        obs = vec_env.reset()
        done = [False]
        initial_val = None
        lstm_states = None
        episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        
        while not all(done):
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True
            )
            obs, _, done, info = vec_env.step(action)
            episode_starts = done
            
            if initial_val is None:
                initial_val = info[0].get("portfolio_value", 10_000)
                
        final_val = info[0].get("portfolio_value", 10_000)
        returns.append((final_val - initial_val) / initial_val)
        
    return float(np.mean(returns))


def objective(trial: optuna.Trial) -> float:
    """One Optuna trial: sample hyperparams → train → evaluate → return metric."""
    lr         = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps    = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    gamma      = trial.suggest_float("gamma", 0.90, 0.9999, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    ent_coef   = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])

    vec_env = DummyVecEnv([make_env() for _ in range(N_ENVS)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        enable_critic_lstm=True,
        lstm_hidden_size=lstm_hidden_size
    )

    model = RecurrentPPO(
        "MultiInputLstmPolicy" if hasattr(vec_env.observation_space, "spaces") else "MlpLstmPolicy", vec_env,
        learning_rate=lr, n_steps=n_steps, batch_size=batch_size,
        gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef,
        clip_range=clip_range, max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs, verbose=0,
    )
    model.learn(total_timesteps=TRIAL_STEPS)

    # Freeze normalisation stats before scoring — do NOT let evaluation
    # itself keep updating the running mean/std (that's eval-set leakage).
    vec_env.training    = False
    vec_env.norm_reward = False

    score = evaluate_model(model, vec_env)
    vec_env.close()

    return score


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_params
    print(f"\nBest trial  |  score={study.best_value:.4f}")
    print(f"Params: {best}")

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)

    print(f"Saved -> {BEST_PARAMS_PATH}")
