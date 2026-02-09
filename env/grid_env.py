import logging
from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.grid import Grid


logger = logging.getLogger(__name__)


class GridEnv(gym.Env):
    """Gym environment wrapper for the `Grid` simulation.

    Key improvements:
    - explicit type hints on public methods
    - logging instead of printing
    - reward components normalized to prevent magnitude explosion
    """

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

        obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([100.0, 2000.0, 10000.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.nominal_frequency = getattr(self.grid, "nominal_frequency", 50.0)

    def reset(self, *, seed: Any = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and underlying grid.

        Returns (obs, info).
        """
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.grid.reset()

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

        freq, rocof = self.grid.step()

        self.step_count += 1

        # Compute normalized reward components
        freq_dev = freq - self.nominal_frequency

        # Normalization / scale constants (tunable)
        max_freq_dev = 5.0  # Hz (worst-case considered)
        max_rocof = 20.0  # Hz/s (worst-case considered)
        max_action = float(self.max_delta) if hasattr(self, "max_delta") else 1.0

        # Component-wise penalties (normalized to roughly [-1,0] when at scale)
        freq_penalty = -((freq_dev) ** 2) / (max_freq_dev ** 2)
        rocof_penalty = -(abs(rocof) / max_rocof)
        action_penalty = -((abs(delta) / max_action) * 0.1)
        survival_bonus = 0.5

        # Aggregate reward (components logged in info)
        reward = freq_penalty + rocof_penalty + action_penalty + survival_bonus

        terminated = False
        truncated = False
        info: Dict[str, Any] = {
            "rocof": rocof,
            "reward_components": {
                "freq_penalty": float(freq_penalty),
                "rocof_penalty": float(rocof_penalty),
                "action_penalty": float(action_penalty),
                "survival_bonus": float(survival_bonus),
                "threshold_penalty": 0.0,
            },
        }

        # Large penalty on threshold breach (episode termination)
        if self.grid.check_threshold():
            terminated = True
            threshold_penalty = -10.0
            reward += threshold_penalty
            info["reward_components"]["threshold_penalty"] = float(threshold_penalty)

        if self.step_count >= self.episode_length:
            truncated = True

        # Clip final reward to avoid magnitude explosion
        reward = float(np.clip(reward, -10.0, 10.0))

        obs = np.array([self.grid.frequency, float(self.grid.generation), float(self.grid.load)], dtype=np.float32)

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode: str = "human") -> None:
        logger.info(
            "Step %03d | Freq: %.3f Hz | Gen: %.1f | Load: %.1f",
            self.step_count,
            self.grid.frequency,
            self.grid.generation,
            self.grid.load,
        )

    def close(self) -> None:
        return None


