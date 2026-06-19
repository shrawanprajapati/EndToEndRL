"""
models/train_ppo.py  |  Day 11  |  M3 — RL Engineer
First PPO training run — 200,000 steps to verify the agent is actually learning.

RUNS  : cd RLModel && python models/train_ppo.py
WATCH : tensorboard --logdir ./tb_logs   (in a separate terminal)

⚠  If reward curve is flat or going down after 200k steps →
   environment has a bug. Stop and debug before running tune_hyperparams.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from env.trading_env import TradingEnvironment

SAVE_PATH   = "models/saved/ppo_checkpoint_200k"
TB_LOG_DIR  = "./tb_logs/"
N_ENVS      = 4           # 4 parallel environments — speeds up data collection ~4x
TOTAL_STEPS = 200_000

# starting hyperparameters — Optuna will improve these on Day 12
PPO_KWARGS = dict(
    learning_rate   = 3e-4,   # how fast the network weights update
    n_steps         = 1024,   # steps collected per env before each PPO update
    batch_size      = 64,     # minibatch size during gradient update
    n_epochs        = 10,     # how many times we reuse each batch
    gamma           = 0.99,   # discount factor — how much future rewards matter
    gae_lambda      = 0.95,   # GAE smoothing — balances bias vs variance
    clip_range      = 0.2,    # PPO clipping — prevents too-large policy updates
    verbose         = 1,
    tensorboard_log = TB_LOG_DIR,
)


def make_env():
    # factory function required by SubprocVecEnv — each call creates one env
    def _init():
        return TradingEnvironment(data_path="data/processed/train.csv")
    return _init


if __name__ == "__main__":
    # SubprocVecEnv runs each env in its own process — true parallelism
    vec_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    # VecNormalize: tracks running mean/std of observations and rewards
    # this stabilises training significantly — keep norm_reward=True for PPO
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy", vec_env, **PPO_KWARGS)

    # save a checkpoint every 50k steps so we can inspect learning progress
    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000 // N_ENVS,  # divide by N_ENVS because SB3 counts per-env steps
        save_path   = "models/saved/",
        name_prefix = "ppo_ckpt",
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint_cb)
    model.save(SAVE_PATH)

    print(f"✓ Checkpoint saved → {SAVE_PATH}.zip")
    print("  Run: tensorboard --logdir ./tb_logs  to inspect the reward curve")