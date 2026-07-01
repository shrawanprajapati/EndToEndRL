"""
train_final.py  |  Day 13  |  M3 — RL Engineer
────────────────────────────────────────────────
PURPOSE : Final 3M-step training run using best_params.json from Optuna.
          EvalCallback saves the best model automatically every 30k steps.
READS   : models/saved/best_params.json
          env/trading_env.py, data/processed/train.csv (via env)
PRODUCES: models/saved/ppo_final.zip          ← THE model used for backtesting
          models/saved/vec_normalize.pkl      ← REQUIRED to evaluate ppo_final correctly
RUNS    : PYTHONPATH=. python models/train_final.py  (~2–4 hours)

macOS FIX: SubprocVecEnv deadlocks on macOS due to the 'spawn' multiprocessing
method. Replaced with DummyVecEnv which runs in the same process — no deadlocks.

FIX (this version): eval_env had its own independently-initialized VecNormalize
that never received train_env's running mean/std. EvalCallback was therefore
picking the "best model" based on an eval env feeding the policy
differently-scaled observations than it was trained on — the checkpoint
selection itself was unreliable. Now we sync eval_env's stats from train_env
every time EvalCallback runs.
"""

import os
import sys
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common.callbacks import EvalCallback
from env.trading_env import TradingEnvironment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

SAVE_DIR          = os.path.join(ROOT_DIR, "models", "saved")
BEST_PARAMS_PATH  = os.path.join(SAVE_DIR, "best_params.json")
FINAL_MODEL_PATH  = os.path.join(SAVE_DIR, "ppo_final")
VECNORM_PATH      = os.path.join(SAVE_DIR, "vec_normalize.pkl")
TB_LOG_DIR        = os.path.join(ROOT_DIR, "tb_logs")
TOTAL_STEPS       = 20_00_000
N_ENVS            = 4


def make_env():
    def _init():
        return TradingEnvironment(data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "train.csv"))
    return _init


class SyncedEvalCallback(EvalCallback):
    """EvalCallback that copies train_env's VecNormalize stats into eval_env
    immediately before each evaluation, so the policy is evaluated on
    observations scaled the same way it was trained on."""

    def __init__(self, eval_env, train_env, **kwargs):
        super().__init__(eval_env, **kwargs)
        self._train_env = train_env

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sync_envs_normalization(self._train_env, self.eval_env)
        return super()._on_step()


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(BEST_PARAMS_PATH) as f:
        best = json.load(f)
    print(f"Loaded hyperparams: {best}")

    # Training environment
    train_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # Separate eval environment — stats kept in sync with train_env via callback below
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.training = False  # never let eval data update its own stats

    model = RecurrentPPO(
        "MultiInputLstmPolicy" if hasattr(train_env.observation_space, "spaces") else "MlpLstmPolicy", train_env,
        learning_rate   = best.get("learning_rate", 1.73e-05),
        n_steps         = best.get("n_steps",       2048),
        batch_size      = best.get("batch_size",    256),
        n_epochs        = 5,
        gamma           = best.get("gamma",          0.95),
        gae_lambda      = best.get("gae_lambda",     0.91),
        clip_range      = 0.15,
        ent_coef        = 0.02,  # Encourages exploration to discover profitable short selling
        vf_coef         = 0.7,
        max_grad_norm   = 0.5,
        device          = "cuda" if th.cuda.is_available() else "cpu",
        policy_kwargs   = dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            enable_critic_lstm=True,
            lstm_hidden_size=128
        ),
        verbose         = 1,
        tensorboard_log = TB_LOG_DIR,
    )

    eval_cb = SyncedEvalCallback(
        eval_env, train_env,
        best_model_save_path = SAVE_DIR,
        log_path             = SAVE_DIR,
        eval_freq            = 100_000 // N_ENVS,
        n_eval_episodes      = 5,  # Fewer episodes for faster validation checks
        deterministic        = True,
        verbose              = 1,
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_cb)
    model.save(FINAL_MODEL_PATH)

    # CRITICAL: save VecNormalize stats — evaluate.py needs these
    train_env.save(VECNORM_PATH)
    print(f"[SUCCESS] Saved final model: {FINAL_MODEL_PATH}.zip")
    print(f"[SUCCESS] Saved final normalizer: {VECNORM_PATH}")