"""
models/train_ppo.py — Model Training Script
Configured for 2,000,000 steps, wide [256, 256] network,
learning rate scheduling, SubprocVecEnv, and checkpoint metadata sidecars.
"""

import os
import sys
import math
import json
import datetime
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn
from env.trading_env import TradingEnvironment

SAVE_DIR     = "models/saved/"
SAVE_PATH    = os.path.join(SAVE_DIR, "ppo_final")
VECNORM_PATH = os.path.join(SAVE_DIR, "vec_normalize.pkl")
TB_LOG_DIR   = "./tb_logs/"
N_ENVS       = 4
TOTAL_STEPS  = 2_000_000

# Learning Rate schedule: starts at 1.53e-4 and decays to 3e-5
lr_schedule = get_linear_fn(1.53e-4, 3e-5, 1.0)

PPO_KWARGS = dict(
    learning_rate=lr_schedule,
    n_steps=2048,
    batch_size=256,
    n_epochs=5,
    gamma=0.951237,
    gae_lambda=0.90971,
    clip_range=0.15,
    ent_coef=0.02,
    vf_coef=0.7,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 128],
            vf=[256, 128]
        ),
        enable_critic_lstm=True,
        lstm_hidden_size=128
    ),
    verbose=1,
    tensorboard_log=TB_LOG_DIR
)

class CheckpointMetadataCallback(BaseCallback):
    """Custom callback to save checkpoint zips, normalizers, and metadata JSON files."""
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            checkpoint_name = f"ppo_ckpt_{self.num_timesteps}"
            model_path = os.path.join(self.save_path, checkpoint_name)
            
            # Save checkpoint PPO weights
            self.model.save(model_path)
            
            # Save normalizer statistics
            norm_path = os.path.join(self.save_path, f"vec_normalize_{self.num_timesteps}.pkl")
            if self.training_env is not None:
                self.training_env.save(norm_path)
            
            # Calculate rolling reward mean
            ep_rew_mean = -50.0
            if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
                ep_rew_mean = float(np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))

            # Dynamic Sharpe valuation based on step progress
            metadata = {
                "steps": self.num_timesteps,
                "ep_rew_mean": ep_rew_mean,
                "timestamp": str(datetime.date.today()),
                "sharpe_estimated": None,
                "sharpe_note": "Run evaluate.py for real Sharpe"
            }
            
            meta_path = model_path + "_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            print(f"[CHECKPOINT] Saved model, stats, and metadata for step {self.num_timesteps} to {meta_path}")
        return True

# Removed ResetNoiseCallback as we are using RecurrentPPO instead of NoisyLinear

class RolloutTelemetryCallback(BaseCallback):
    """Custom callback to log rollout statistics to TensorBoard."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_allocations = []
        self.step_drawdowns = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "drawdown" in info:
                self.step_drawdowns.append(float(info["drawdown"]))
            if "btc_allocation" in info:
                self.step_allocations.append(float(info["btc_allocation"]))
        return True

    def _on_rollout_end(self) -> None:
        if self.step_drawdowns:
            self.logger.record("rollout/mean_drawdown", float(np.mean(self.step_drawdowns)))
            self.step_drawdowns = []
        if self.step_allocations:
            self.logger.record("rollout/mean_allocation", float(np.mean(self.step_allocations)))
            self.step_allocations = []

def make_env(data_file):
    def _init():
        return TradingEnvironment(
            data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", data_file),
            domain_randomize=True
        )
    return _init

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Use SubprocVecEnv with 4 parallel environments loaded with train.csv
    vec_env = SubprocVecEnv([make_env("train.csv") for _ in range(N_ENVS)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = RecurrentPPO("MultiInputLstmPolicy" if hasattr(vec_env.observation_space, "spaces") else "MlpLstmPolicy", vec_env, device="cpu", **PPO_KWARGS)

    # Checkpoint every 100,000 steps per environment (400,000 total steps)
    checkpoint_cb = CheckpointMetadataCallback(
        save_freq=100_000 // N_ENVS,
        save_path=SAVE_DIR
    )
    
    telemetry_cb = RolloutTelemetryCallback()
    callbacks = CallbackList([checkpoint_cb, telemetry_cb])

    print("Starting PPO training (2,000,000 steps on CPU)...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks)
    
    model.save(SAVE_PATH)
    vec_env.save(VECNORM_PATH)
    
    # Save metadata for final model checkpoint
    final_metadata = {
        "steps": TOTAL_STEPS,
        "ep_rew_mean": float(np.mean([ep_info["r"] for ep_info in model.ep_info_buffer])) if len(model.ep_info_buffer) > 0 else -10.0,
        "timestamp": str(datetime.date.today()),
        "sharpe": None
    }
    with open(SAVE_PATH + "_metadata.json", "w") as f:
        json.dump(final_metadata, f, indent=2)

    print(f"[SUCCESS] Saved final model: {SAVE_PATH}.zip")
    print(f"[SUCCESS] Saved final normalizer: {VECNORM_PATH}")