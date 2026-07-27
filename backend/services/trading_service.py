import os
import sys
import asyncio
from datetime import datetime
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent directory to path to import env
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, "RLModel"))

from env.trading_env import TradingEnvironment


def _unwrap_reset(reset_result):
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def _unwrap_step(step_result):
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = step_result
    return obs, reward, bool(done), info

class TradingService:
    def __init__(self):
        self.model_path = os.path.join(ROOT_DIR, "RLModel", "models", "saved", "ppo_final")
        self.venv_path = os.path.join(ROOT_DIR, "RLModel", "models", "saved", "vec_normalize.pkl")
        self.data_path = os.path.join(ROOT_DIR, "RLModel", "data", "processed", "featured_data.csv")
        
        self.env = None
        self.model = None
        self.is_ready = False

    def load(self):
        try:
            def make_env():
                return TradingEnvironment(data_path=self.data_path)
            
            base_env = DummyVecEnv([make_env])
            self.env = base_env

            # Import the custom policy so pickle can deserialize it from the zip
            from models.noisy_net import NoisyActorCriticPolicy
            self.model = PPO.load(
                self.model_path,
                env=self.env,
                custom_objects={"policy_class": NoisyActorCriticPolicy},
            )
            self.is_ready = True
            print("Trading Service Initialized: RL Model Loaded.")
            # Note: run_simulation_loop is started by the WebSocket endpoint, not here
        except Exception as e:
            print(f"Error initializing Trading Service: {e}")

    def predict(self, observation):
        """Run one step of inference with the PPO policy."""
        if not self.is_ready or self.model is None:
            return 0.0
        action, _ = self.model.predict(observation, deterministic=True)
        return float(np.asarray(action).reshape(-1)[0])

    async def run_simulation_loop(self, callback=None):
        """Runs a continuous simulation loop, calling callback(data) for each tick."""

        if not self.is_ready or self.env is None:
            return

        print("Starting live simulation stream...")
        
        obs = _unwrap_reset(self.env.reset())
        
        while True:
            action = self.predict(obs)
            obs, reward, done, info = _unwrap_step(self.env.step([action]))
            
            # Send tick to frontend clients
            data = {
                "type": "tick",
                "price": float(info[0]["price"]) if isinstance(info, list) else float(info["price"]),
                "action": float(action),
                "portfolio_value": float(info[0]["portfolio_value"]) if isinstance(info, list) else float(info["portfolio_value"]),
                "drawdown": float(info[0]["drawdown"]) if isinstance(info, list) else float(info["drawdown"]),
                "unrealized_pnl": float((info[0] if isinstance(info, list) else info).get("unrealized_pnl", 0.0)),
                "hmm_regime": (info[0] if isinstance(info, list) else info).get("hmm_regime", "Unknown"),
                "timestamp": datetime.now().isoformat()
            }
            
            if callback is not None:
                await callback(data)

            if done:
                obs = _unwrap_reset(self.env.reset())

            await asyncio.sleep(1)
