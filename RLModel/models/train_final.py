"""
train_final.py  |  Day 13  |  M3 — RL Engineer
────────────────────────────────────────────────
PURPOSE : Final 1M-step training run using best_params.json from Optuna.
          EvalCallback saves the best model automatically every 50k steps.
READS   : models/saved/best_params.json
          env/trading_env.py, data/processed/train.csv (via env)
PRODUCES: models/saved/ppo_final.zip  ← THE model used for backtesting
          docs/training_curves.png    ← export manually from TensorBoard
RUNS    : python models/train_final.py  (~2–4 hours)
"""

import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from env.trading_env import TradingEnvironment

BEST_PARAMS_PATH = "models/saved/best_params.json"
FINAL_MODEL_PATH = "models/saved/ppo_final"
TB_LOG_DIR       = "./tb_logs/"
TOTAL_STEPS      = 1_000_000
N_ENVS           = 4


def make_env():
    def _init():
        return TradingEnvironment()
    return _init


if __name__ == "__main__":
    with open(BEST_PARAMS_PATH) as f:
        best = json.load(f)
    print(f"Loaded hyperparams: {best}")

    # Training environments
    train_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # Separate eval environment (single env, deterministic)
    eval_env = VecNormalize(SubprocVecEnv([make_env()]), norm_obs=True, norm_reward=False)

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate = best.get("learning_rate", 3e-4),
        n_steps       = best.get("n_steps",       1024),
        batch_size    = best.get("batch_size",     64),
        gamma         = best.get("gamma",          0.99),
        gae_lambda    = best.get("gae_lambda",     0.95),
        verbose       = 1,
        tensorboard_log = TB_LOG_DIR,
    )

    # EvalCallback: evaluates every 50k steps, saves best model automatically
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/saved/",
        log_path="models/saved/",
        eval_freq=50_000 // N_ENVS,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_cb)
    model.save(FINAL_MODEL_PATH)

    print(f"✓ Final model saved → {FINAL_MODEL_PATH}.zip")
    print("  Export training curves: tensorboard --logdir ./tb_logs → save as docs/training_curves.png")
