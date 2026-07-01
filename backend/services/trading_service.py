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
                
            from sb3_contrib import RecurrentPPO
            self.model = RecurrentPPO.load(self.model_path, env=self.env)
            self.is_ready = True
            print("Trading Service Initialized: RL Model Loaded.")
            
            asyncio.create_task(self.run_simulation_loop())
        except Exception as e:
            print(f"Error initializing Trading Service: {e}")

    def predict(self, observation, lstm_states=None, episode_starts=None):
        if not self.is_ready or self.model is None:
            return 0.0, lstm_states
        
        action, lstm_states = self.model.predict(
            observation, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=True
        )
        return float(np.asarray(action).reshape(-1)[0]), lstm_states

    async def run_simulation_loop(self):
        """Runs a continuous background simulation loop updating STATE."""
        import asyncio
        from backend.main import STATE, callback_tick

        if not self.is_ready or self.env is None:
            return

        print("Starting live simulation stream...")
        
        obs = _unwrap_reset(self.env.reset())
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        while True:
            action, lstm_states = self.predict(
                obs, 
                lstm_states=lstm_states, 
                episode_starts=episode_starts
            )
            obs, reward, done, info = _unwrap_step(self.env.step([action]))
            episode_starts = np.array([done])
            
            # Send tick to frontend clients
            data = {
                "type": "tick",
                "price": float(info["price"]),
                "action": float(action),
                "portfolio_value": float(info["portfolio_value"]),
                "drawdown": float(info["drawdown"]),
                "unrealized_pnl": float(info.get("unrealized_pnl", 0.0)),
                "hmm_regime": info.get("hmm_regime", "Unknown"),
                "timestamp": datetime.now().isoformat()
            }
            
            STATE["latest_tick"] = data
            await callback_tick(data)

            if done:
                obs = _unwrap_reset(self.env.reset())

            await asyncio.sleep(1)
