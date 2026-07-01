import os
import sys
import argparse
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Add root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "RLModel"))

from env.trading_env import TradingEnvironment

def train(steps=100000):
    log_dir = os.path.join(ROOT_DIR, "RLModel", "tb_logs")
    save_dir = os.path.join(ROOT_DIR, "RLModel", "models", "saved")
    os.makedirs(save_dir, exist_ok=True)
    
    data_path = os.path.join(ROOT_DIR, "RLModel", "data", "processed", "train.csv")
    
    def make_env():
        return TradingEnvironment(data_path=data_path)
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=save_dir,
        name_prefix="ppo_v2"
    )
    
    print(f"Starting training for {steps} steps...")
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(os.path.join(save_dir, "ppo_v2_final"))
    env.save(os.path.join(save_dir, "vec_normalize_v2.pkl"))
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=100000)
    args = parser.parse_args()
    
    train(steps=args.total_steps)
