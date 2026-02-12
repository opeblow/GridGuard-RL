import logging
from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.grid import Grid


logger = logging.getLogger(__name__)


class GridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, episode_length: int = 200):
        super().__init__()

        self.grid = Grid()
        self.episode_length = episode_length
        self.step_count = 0

        self.max_delta = 100.0
        self.action_space = spaces.Box(
            low=np.array([-self.max_delta], dtype=np.float32),
            high=np.array([self.max_delta], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        obs_low = np.array([45.0, 0.0,0.0])
        obs_high = np.array([55.0, 3000.0, 3000.0])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.nominal_frequency = getattr(self.grid, "nominal_frequency", 50.0)
        self.base_load = 1000.0
        self.load_noise_std = 50.0
        self.surge_probability = 0.02
        self.surge_magnitude = 300.0
        self.cummulative_load_change = 0.0

    def reset(self, *, seed: Any = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and underlying grid.

        Returns (obs, info).
        """
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.grid.reset()
        initial_load_var = np.random.uniform(-100,100)
        self.grid.load = self.base_load + initial_load_var
        self.grid.generation = self.grid.load
        self.cummulative_load_change = 0.0

        obs = np.array([self.grid.frequency, float(self.grid.generation), float(self.grid.load)], dtype=np.float32)
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if isinstance(action, (list, tuple, np.ndarray)):
            delta = float(np.array(action).ravel()[0])
        else:
            delta = float(action)

        delta = float(np.clip(delta, -self.max_delta, self.max_delta))
        self.grid.change_generation(delta)
        noise= np.random.normal(0,self.load_noise_std)
        self.grid.change_load(noise)
        self.cummulative_load_change += abs(noise)

        if np.random.random() < self.surge_probability:
            surge = np.random.choice([-1,1]) * np.random.uniform(0.5,1.0) * self.surge_magnitude
            self.grid.change_load(surge)
            self.cummulative_load_change += abs(surge)

        freq,rocof = self.grid.step()
        self.step_count += 1
        freq_dev = freq - self.nominal_frequency
        max_freq_dev = 2.0
        max_rocof = 10.0
        max_action = float(self.max_delta)
        freq_penalty = -((freq_dev/max_freq_dev) ** 2) * 2.0
        rocof_penalty = -(abs(rocof)/ max_rocof) * 0.5
        action_penalty = -(abs(delta)/ max_action) * 0.05

        stability_bonus = 0.0
        if abs(freq_dev) < 0.2:
            stability_bonus = 0.3

        elif abs(freq_dev) < 0.5:
            stability_bonus = 0.1

        reward = freq_penalty + rocof_penalty + action_penalty + stability_bonus

        terminated = False
        truncated = False
        threshold_penalty = 0.0
        info:Dict[str,Any]= {
            "rocof":rocof,
            "cummulative_load_change":float(self.cummulative_load_change),
            "reward_components":{
                "freq_penalty":float(freq_penalty),
                "rocof_penalty":float(rocof_penalty),
                "action_penalty":float(action_penalty),
                "stability_bonus":float(stability_bonus),
                "threshold_penalty":0.0,

            },
        }
        if self.grid.check_threshold():
            terminated = True
            threshold_penalty = -10.0
            reward += threshold_penalty
            info["reward_components"]["threshold_penalty"] = float(threshold_penalty)

        if self.step_count >=self.episode_length:
            truncated = True

        reward = float(np.clip(reward,-15.0,5.0))
        obs = np.array([self.grid.frequency,float(self.grid.generation),float(self.grid.load)],dtype=np.float32)

        return obs , float(reward), bool(terminated) ,bool(truncated) , info
    

    def render(self,mode:str="human") -> None:
        logger.info(
            "Step %03d | Freq: %.3f Hz | Gen: %.1f |Load:%.1f",
            self.step_count,
            self.grid.frequency,
            self.grid.generation,
            self.grid.load,
        )

    
    def close(self) -> None:
        return None
    

    