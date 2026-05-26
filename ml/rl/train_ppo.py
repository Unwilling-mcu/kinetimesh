"""
KinetiMesh — PPO Reinforcement Learning Power Dispatch Agent
Algorithm: Proximal Policy Optimization (Schulman et al., 2017)
Environment: KinetiMeshGridEnv-v1 (custom OpenAI Gymnasium env)
Author: Sanchayan | B.Tech IT
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import argparse, logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("kinetimesh.rl")


# ──────────────────────────────────────────
# 1. Custom Gymnasium Environment
# ──────────────────────────────────────────

class KinetiMeshGridEnv(gym.Env):
    """
    KinetiMesh Power Dispatch Environment.

    State s_t: [harvest_vec(18), demand_forecast(3), battery_soc(3),
                grid_price, time_of_day, day_of_week, train_schedule]
    Action a_t: Power allocation vector ∈ [0,1]^3 (softmax-normalized)
    Reward r_t: α·green_util − β·curtailment − γ·grid_import + δ·soc_health
    """

    metadata = {"render_modes": ["human"]}

    # Reward weights
    ALPHA = 1.0   # Green utilization bonus
    BETA  = 0.5   # Curtailment penalty
    GAMMA = 0.3   # Grid import penalty
    DELTA = 0.2   # Battery health bonus

    def __init__(self, n_zones: int = 3, n_nodes: int = 18):
        super().__init__()
        self.n_zones = n_zones
        self.n_nodes = n_nodes
        self.tick = 0
        self.train_active = False
        self.crowd_active = False

        # State space
        obs_dim = n_nodes + n_zones + n_zones + 3  # harvest + demand + soc + [price,tod,dow]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action space: continuous allocation ∈ [0,1]^n_zones (will be softmax-normalized)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_zones,), dtype=np.float32
        )

        # Node base powers
        self.node_bases = np.array([
            1.8,2.1,1.6,1.9,1.4,2.3,   # Rail PEH
            0.24,0.18,0.22,0.19,0.21,0.17,  # Floor PEH
            0.15,0.12,                    # Wind
            0.08,0.07,                    # Thermal
            0.09,0.08,                    # Street tiles
        ])[:n_nodes]

        self.battery_soc = np.array([0.6, 0.5, 0.4])  # Initial SoC per zone
        self.reset()

    def _get_harvest(self) -> np.ndarray:
        t = self.tick
        harvest = self.node_bases.copy()
        for i in range(self.n_nodes):
            harvest[i] *= (0.7 + 0.6*np.random.random())
            if i < 6:  # Rail nodes
                harvest[i] *= (0.8 + 0.4*abs(np.sin(t*0.08+self.node_bases[i])))
                if self.train_active: harvest[i] *= 4.5
            elif i < 14:  # Floor nodes
                harvest[i] *= (0.6 + 0.7*abs(np.sin(t*0.05+self.node_bases[i]*2)))
                if self.crowd_active: harvest[i] *= 3.2
        return harvest.astype(np.float32)

    def _get_demand(self) -> np.ndarray:
        h = (self.tick % (24*3600)) / 3600
        base_demand = 4.0 + 2.0*np.sin((h-12)*np.pi/12)
        demand = np.array([
            base_demand * 0.45,  # Zone A
            base_demand * 0.33,  # Zone B
            base_demand * 0.22,  # Zone C
        ])
        return demand.astype(np.float32)

    def _get_grid_price(self) -> float:
        h = (self.tick % (24*3600)) / 3600
        return float(0.8 + 0.4*abs(np.sin(h*np.pi/12)))  # ₹/kWh

    def _get_obs(self) -> np.ndarray:
        harvest = self._get_harvest()
        demand  = self._get_demand()
        price   = self._get_grid_price()
        tod     = (self.tick % 86400) / 86400
        dow     = (self.tick // 86400) % 7 / 7
        obs = np.concatenate([harvest, demand, self.battery_soc, [price, tod, dow]])
        return obs.astype(np.float32)

    def step(self, action: np.ndarray):
        self.tick += 60  # 60-second decision interval

        # Normalize action to valid allocation
        alloc = np.exp(action) / np.sum(np.exp(action))  # softmax

        # Compute harvest
        harvest = self._get_harvest()
        total_harvest = harvest.sum()
        demand  = self._get_demand()

        # Distribute harvested power to zones by allocation
        zone_harvest = alloc * total_harvest

        # Battery charging/discharging
        for i in range(self.n_zones):
            charge = zone_harvest[i] * 0.02
            discharge = demand[i] * 0.015
            self.battery_soc[i] = np.clip(self.battery_soc[i] + charge - discharge, 0.05, 0.98)

        # Reward computation
        total_demand = demand.sum()
        green_util   = min(1.0, total_harvest / max(total_demand, 0.01))
        curtailment  = max(0.0, total_harvest - total_demand)
        grid_import  = max(0.0, total_demand - total_harvest)
        grid_price   = self._get_grid_price()
        soc_health   = 1.0 - np.mean(np.abs(self.battery_soc - 0.6))

        reward = (
            self.ALPHA * green_util
            - self.BETA  * (curtailment / max(total_demand, 0.01))
            - self.GAMMA * (grid_import * grid_price / 10.0)
            + self.DELTA * soc_health
        )
        reward = float(reward * 100)  # Scale for PPO

        obs = self._get_obs()
        terminated = False
        truncated  = self.tick >= 86400 * 7  # 1 week episode
        info = {"green_util": green_util, "curtailment": curtailment,
                "grid_import": grid_import, "total_harvest": total_harvest}

        # Random events
        self.train_active = np.random.random() < 0.05
        self.crowd_active = np.random.random() < 0.08

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tick = 0
        self.train_active = False
        self.crowd_active = False
        self.battery_soc  = np.array([0.6, 0.5, 0.4])
        return self._get_obs(), {}

    def render(self):
        obs = self._get_obs()
        total_harvest = obs[:self.n_nodes].sum()
        print(f"Tick {self.tick:6d} | Harvest: {total_harvest:.3f} kW | "
              f"SoC: {self.battery_soc.round(2)}")


# ──────────────────────────────────────────
# 2. PPO Training
# ──────────────────────────────────────────

def train(args):
    log.info(f"KinetiMesh PPO Training | timesteps={args.timesteps:,}")

    env      = Monitor(KinetiMeshGridEnv())
    eval_env = Monitor(KinetiMeshGridEnv())

    model = PPO(
        "MlpPolicy", env,
        n_steps=2048, batch_size=256, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256, 256]},
        verbose=1,
    )

    callbacks = [
        EvalCallback(eval_env, eval_freq=10000, n_eval_episodes=5,
                     best_model_save_path="./models/", deterministic=True),
        CheckpointCallback(save_freq=50000, save_path="./checkpoints/",
                           name_prefix="kinetimesh_ppo"),
    ]

    log.info("Starting PPO training...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    model.save("kinetimesh_ppo_final")
    log.info("Training complete — model saved to kinetimesh_ppo_final.zip")

    # Quick evaluation
    obs, _ = eval_env.reset()
    total_reward = 0.0
    for _ in range(1440):  # 1 day
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = eval_env.step(action)
        total_reward += reward
        if term or trunc: break

    log.info(f"Evaluation — 1-day episode reward: {total_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()
    train(args)
