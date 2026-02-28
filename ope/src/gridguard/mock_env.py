import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces


@dataclass
class GridState:
    bus_voltages: np.ndarray
    phase_angles: np.ndarray
    line_loadings: np.ndarray
    generator_outputs: np.ndarray
    loads: np.ndarray
    frequency_deviation: float
    instability_score: float


class MockGrid:
    """Mock grid that generates realistic state data"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._initialize_state()
        
    def _initialize_state(self):
        self.base_voltages = 1.0 + self.rng.normal(0, 0.02, 14)
        self.base_angles = self.rng.normal(0, 0.05, 14)
        self.base_loadings = 0.4 + self.rng.uniform(0, 0.3, 20)
        self.base_gen_output = 50 + self.rng.uniform(0, 100, 5)
        self.base_load = 10 + self.rng.uniform(0, 30, 11)
        
        self.voltages = self.base_voltages.copy()
        self.angles = self.base_angles.copy()
        self.loadings = self.base_loadings.copy()
        
    def get_state(self) -> GridState:
        return GridState(
            bus_voltages=self.voltages,
            phase_angles=self.angles,
            line_loadings=self.loadings,
            generator_outputs=self.base_gen_output,
            loads=self.base_load,
            frequency_deviation=0.0,
            instability_score=0.0
        )
    
    def apply_disturbance(self, disturbance_type: str, magnitude: float = 1.0, difficulty: str = "medium"):
        base_effect = 1.0
        if difficulty == "hard":
            base_effect = 1.5
        elif difficulty == "easy":
            base_effect = 0.5
            
        if disturbance_type == "load_increase":
            self.voltages *= (1.0 - 0.02 * magnitude * base_effect)
            self.loadings *= (1.0 + 0.1 * magnitude * base_effect)
            
        elif disturbance_type == "load_spike":
            idx = self.rng.integers(0, len(self.base_load))
            self.base_load[idx] *= (1.5 * magnitude * base_effect)
            self.loadings += 0.2 * base_effect
            
        elif disturbance_type == "line_trip":
            idx = self.rng.integers(0, len(self.loadings))
            self.loadings[idx] *= (1.8 * base_effect)
            other_idx = (idx + 1) % len(self.loadings)
            self.loadings[other_idx] *= (1.3 * base_effect)
            
        elif disturbance_type == "multi_line_trip":
            for _ in range(3):
                idx = self.rng.integers(0, len(self.loadings))
                self.loadings[idx] *= (2.0 * base_effect)
                
        elif disturbance_type == "gen_outage":
            idx = self.rng.integers(0, len(self.base_gen_output))
            self.base_gen_output[idx] *= (0.3 * base_effect)
            self.voltages += self.rng.normal(-0.08 * base_effect, 0.02, 14)
            
        elif disturbance_type == "multi_gen_outage":
            for _ in range(2):
                idx = self.rng.integers(0, len(self.base_gen_output))
                self.base_gen_output[idx] *= 0.3
            self.voltages += self.rng.normal(-0.12 * base_effect, 0.03, 14)
            
        elif disturbance_type == "cascading_start":
            self.voltages *= (0.95 - 0.02 * base_effect)
            self.loadings *= (1.2 + 0.1 * base_effect)
            self.angles += self.rng.normal(0, 0.1 * base_effect, 14)
            
        elif disturbance_type == "voltage_collapse":
            idx = self.rng.integers(0, len(self.voltages))
            self.voltages[idx] *= (0.7 * base_effect)
            self.voltages = np.clip(self.voltages * 0.95, 0.7, 1.1)
            
        elif disturbance_type == "frequency_swing":
            self.voltages += self.rng.normal(-0.03 * base_effect, 0.01, 14)
            self.loadings += 0.15 * base_effect
            
        elif disturbance_type == "renewable_drop":
            gen_idx = self.rng.integers(2, 5)
            self.base_gen_output[gen_idx] *= (0.2 * base_effect)
            self.voltages += self.rng.normal(-0.04 * base_effect, 0.02, 14)
            
        elif disturbance_type == "weather_storm":
            self.voltages *= (0.92 - 0.02 * base_effect)
            self.loadings *= (1.15 + 0.1 * base_effect)
            for _ in range(2):
                idx = self.rng.integers(0, len(self.loadings))
                self.loadings[idx] *= 1.4
            
    def apply_action(self, action: np.ndarray) -> float:
        for i, adj in enumerate(action[:5]):
            if i < len(self.base_gen_output):
                self.base_gen_output[i] += adj * 10
                
        self.voltages += action[5] * 0.01
        self.loadings -= abs(action[6]) * 0.05
        
        self.voltages = np.clip(self.voltages, 0.8, 1.2)
        self.loadings = np.clip(self.loadings, 0.1, 1.5)
        
        return 0.0
        
    def reset(self):
        self._initialize_state()


class GridGuardEnv(gym.Env):
    """Gymnasium environment for GridGuard RL training"""
    
    def __init__(self, max_steps: int = 100, use_mock: bool = True):
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.use_mock = use_mock
        
        if use_mock:
            self.grid = MockGrid()
        else:
            from gridguard.environment import IEEE14BusGrid
            self.grid = IEEE14BusGrid()
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )
        
        # 14 voltages + 14 angles + 20 loadings + 5 gen + 1 freq + 1 instability = 55
        # But we'll use 54 to match original spec
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(54,), dtype=np.float32
        )
        
        self.cumulative_violations = 0
        self.rng = np.random.default_rng(42)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.grid.reset()
        self.current_step = 0
        self.cumulative_violations = 0
        
        if options and 'disturbance' in options:
            difficulty = options.get('difficulty', 'medium')
            self.grid.apply_disturbance(options['disturbance'], 
                                       options.get('magnitude', 1.2),
                                       difficulty)
        else:
            # Complex scenarios with varying difficulty
            easy_disturbs = ["load_increase", "line_trip", "gen_outage"]
            medium_disturbs = ["load_spike", "multi_line_trip", "cascading_start", 
                             "frequency_swing", "renewable_drop"]
            hard_disturbs = ["multi_gen_outage", "voltage_collapse", "weather_storm",
                           "cascading_start", "multi_line_trip"]
            
            roll = self.rng.random()
            if roll < 0.3:
                disturb = self.rng.choice(easy_disturbs)
                magnitude = self.rng.uniform(1.0, 1.5)
                difficulty = "easy"
            elif roll < 0.7:
                disturb = self.rng.choice(medium_disturbs)
                magnitude = self.rng.uniform(1.2, 1.8)
                difficulty = "medium"
            else:
                disturb = self.rng.choice(hard_disturbs)
                magnitude = self.rng.uniform(1.5, 2.2)
                difficulty = "hard"
            
            self.grid.apply_disturbance(disturb, magnitude, difficulty)
        
        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self) -> np.ndarray:
        state = self.grid.get_state()
        
        # 14 voltages + 14 angles + 20 line loadings + 5 gen outputs + 1 = 54
        obs = np.concatenate([
            state.bus_voltages,                      
            state.phase_angles,                      
            state.line_loadings[:20],                
            state.generator_outputs[:5] / 200.0,     
            [state.frequency_deviation]              
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        
        self.grid.apply_action(action)
        
        obs = self._get_obs()
        
        state = self.grid.get_state()
        
        voltage_dev = np.abs(state.bus_voltages - 1.0)
        voltage_violation = np.sum(voltage_dev > 0.1)
        voltage_severe = np.sum(voltage_dev > 0.15)
        
        loading_violation = np.sum(state.line_loadings > 0.9)
        loading_severe = np.sum(state.line_loadings > 1.0)
        
        thermal_violation = np.sum(state.line_loadings > 1.0)
        
        self.cumulative_violations += voltage_violation + loading_violation
        
        reward = 0.0
        
        voltage_stability = np.sum(voltage_dev < 0.05)
        reward += voltage_stability * 2.0
        
        loading_stability = np.sum(state.line_loadings < 0.7)
        reward += loading_stability * 1.5
        
        reward += 5.0
        
        reward -= (voltage_violation * 5 + loading_violation * 3 + thermal_violation * 15)
        
        if voltage_severe > 0 or loading_severe > 0:
            reward -= voltage_severe * 10
            reward -= loading_severe * 10
        
        terminated = False
        if thermal_violation > 0 or np.any(state.bus_voltages < 0.8):
            terminated = True
            reward -= 500
            
        truncated = self.current_step >= self.max_steps
        
        if not terminated and not truncated:
            reward += 2.0
            
        if truncated and voltage_violation == 0 and loading_violation == 0:
            reward += 50.0
            
        info = {
            'violations': self.cumulative_violations,
            'step': self.current_step,
            'voltage_min': float(np.min(state.bus_voltages)),
            'loading_max': float(np.max(state.line_loadings)),
            'reward_components': {
                'stability_bonus': voltage_stability * 2.0 + loading_stability * 1.5,
                'step_bonus': 5.0,
                'violation_penalty': -(voltage_violation * 5 + loading_violation * 3 + thermal_violation * 15)
            }
        }
        
        return obs, float(reward), terminated, truncated, info
    
    def render(self):
        state = self.grid.get_state()
        print(f"Step {self.current_step}")
        print(f"  Voltages: min={state.bus_voltages.min():.3f}, max={state.bus_voltages.max():.3f}")
        print(f"  Line loadings: max={state.line_loadings.max():.3f}")
        print(f"  Violations: {self.cumulative_violations}")


if __name__ == "__main__":
    env = GridGuardEnv(max_steps=10)
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}, term={term}, trunc={trunc}")
