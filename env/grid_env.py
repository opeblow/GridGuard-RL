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
        reward = - (freq_error ** 2) * 8.0
        reward -= 4.0 * abs(rocof)
        reward -= 0.001 * abs(delta)
        reward += 0.5  

        terminated = False
        truncated = False
        info = {"rocof": rocof}

        if self.grid.check_threshold():
            terminated = True
            reward -= 200.0 

        if self.step_count >= self.episode_length:
            truncated = True

        obs = np.array([self.grid.frequency, float(self.grid.generation), float(self.grid.load)], dtype=np.float32)

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode="human"):
        print(f"Step {self.step_count:03d} | Freq: {self.grid.frequency:.3f} Hz | Gen: {self.grid.generation:.1f} | Load: {self.grid.load:.1f}")

    def close(self):
        return None


