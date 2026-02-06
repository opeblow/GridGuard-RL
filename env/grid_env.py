import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.grid import Grid


class GridEnv(gym.Env):
    """Simple Gymnasium environment wrapper for the Grid simulation.

    Observation: np.array([frequency, generation, load], dtype=float32)
    Action: 1-d continuous array representing desired generation setpoint (MW)

    This implementation is intentionally lightweight to support fast iteration
    and to match the interfaces used by the training scripts.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, episode_length: int = 200):
        super().__init__()

        self.grid = Grid()
        self.episode_length = episode_length
        self.step_count = 0

        # Action: set generation setpoint in MW (single continuous value)
        self.min_generation = 0.0
        self.max_generation = 2000.0
        self.action_space = spaces.Box(
            low=np.array([self.min_generation], dtype=np.float32),
            high=np.array([self.max_generation], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # Observation: [frequency, generation, load]
        obs_low = np.array([0.0, self.min_generation, 0.0], dtype=np.float32)
        obs_high = np.array([100.0, self.max_generation, 100000.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Nominal frequency used for reward calculations
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
        # Accept scalars or arrays
        if isinstance(action, (list, tuple, np.ndarray)):
            setpoint = float(np.array(action).ravel()[0])
        else:
            setpoint = float(action)

        # Clip setpoint to allowed generation range
        setpoint = float(np.clip(setpoint, self.min_generation, self.max_generation))

        # Apply action by setting generation to the setpoint
        # Compute delta for small penalty on large changes
        delta = setpoint - float(self.grid.generation)
        self.grid.change_generation(delta)

        # Advance simulation
        freq, rocof = self.grid.step()

        self.step_count += 1

        # Reward: encourage frequency near nominal and small control actions
        freq_error = abs(freq - self.nominal_frequency)
        reward = -freq_error  # primary objective: minimize frequency deviation
        reward -= 0.0001 * abs(delta)  # small penalty for large control moves

        terminated = False
        truncated = False
        info = {"rocof": rocof}

        # Terminal condition: collapse/blackout threshold
        if self.grid.check_threshold():
            terminated = True
            reward -= 100.0

        # Truncate when episode length exceeded
        if self.step_count >= self.episode_length:
            truncated = True

        obs = np.array([self.grid.frequency, float(self.grid.generation), float(self.grid.load)], dtype=np.float32)

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode="human"):
        # Lightweight textual render for debugging
        print(f"Step {self.step_count:03d} | Freq: {self.grid.frequency:.3f} Hz | Gen: {self.grid.generation:.1f} | Load: {self.grid.load:.1f}")

    def close(self):
        return None


