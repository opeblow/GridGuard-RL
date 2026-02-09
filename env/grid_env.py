import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.grid import Grid


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

        obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([100.0, 2000.0, 10000.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

       
        self.nominal_frequency = getattr(self.grid, 'nominal_frequency', 50.0)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.grid.reset()

        obs = np.array([self.grid.frequency, float(self.grid.generation), float(self.grid.load)], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
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
        max_freq_dev = 5.0   # Hz (worst-case considered)
        max_rocof = 20.0     # Hz/s (worst-case considered)
        max_action = float(self.max_delta) if hasattr(self, 'max_delta') else 1.0

        # Component-wise penalties (normalized to roughly [-1,0] when at scale)
        freq_penalty = - (freq_dev ** 2) / (max_freq_dev ** 2)
        rocof_penalty = - (abs(rocof) / max_rocof)
        action_penalty = - (abs(delta) / max_action) * 0.1
        survival_bonus = 0.5

        # Aggregate reward (components logged in info)
        reward = freq_penalty + rocof_penalty + action_penalty + survival_bonus

        terminated = False
        truncated = False
        info = {
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

    def render(self, mode="human"):
        print(f"Step {self.step_count:03d} | Freq: {self.grid.frequency:.3f} Hz | Gen: {self.grid.generation:.1f} | Load: {self.grid.load:.1f}")

    def close(self):
        return None


