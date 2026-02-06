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

        # Action: delta generation (MW change) applied each step
        # Using delta actions (rather than absolute setpoints) makes control
        # smoother and easier for the agent to learn.
        self.max_delta = 100.0
        self.action_space = spaces.Box(
            low=np.array([-self.max_delta], dtype=np.float32),
            high=np.array([self.max_delta], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # Observation: [frequency, generation, load]
        obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([100.0, 2000.0, 10000.0], dtype=np.float32)
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
        # Accept scalars or arrays; treat action as delta change in generation
        if isinstance(action, (list, tuple, np.ndarray)):
            delta = float(np.array(action).ravel()[0])
        else:
            delta = float(action)

        # Clip delta to allowed per-step change
        delta = float(np.clip(delta, -self.max_delta, self.max_delta))

        # Apply delta (Grid will further limit by its own ramp rate)
        self.grid.change_generation(delta)

        # Advance simulation
        freq, rocof = self.grid.step()

        self.step_count += 1

        # Reward shaping to reduce blackouts:
        # - Strong penalty for squared frequency deviation
        # - Penalty for large RoCoF (rapid changes)
        # - Small penalty for large control moves
        # - Small positive per-step reward to encourage uptime
        freq_error = freq - self.nominal_frequency
        reward = - (freq_error ** 2) * 8.0
        reward -= 4.0 * abs(rocof)
        reward -= 0.001 * abs(delta)
        reward += 0.5  # per-step survival bonus

        terminated = False
        truncated = False
        info = {"rocof": rocof}

        # Terminal condition: collapse/blackout threshold
        if self.grid.check_threshold():
            terminated = True
            reward -= 200.0  # stronger terminal penalty to discourage blackouts

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


