import numpy as np
import pandas as pd
import pypsa
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


class IEEE14BusGrid:
    """IEEE 14-bus test system with PyPSA"""
    
    def __init__(self):
        self.network = pypsa.Network()
        self._build_grid()
        self._run_pf()
        
    def _build_grid(self):
        n = self.network
        
        for i in range(1, 15):
            n.add("Bus", f"Bus{i}", v_nom=15.0)
        
        lines = [
            ("Bus1", "Bus2", 0.05917, 0.22304),
            ("Bus1", "Bus5", 0.22304, 0.17388),
            ("Bus2", "Bus3", 0.04699, 0.19797),
            ("Bus2", "Bus4", 0.05811, 0.17632),
            ("Bus2", "Bus5", 0.05695, 0.17388),
            ("Bus3", "Bus4", 0.06701, 0.17103),
            ("Bus4", "Bus5", 0.01335, 0.04211),
            ("Bus5", "Bus6", 0.0, 0.25202),
            ("Bus4", "Bus7", 0.0, 0.20912),
            ("Bus4", "Bus8", 0.0, 0.55618),
            ("Bus7", "Bus8", 0.0, 0.2085),
            ("Bus8", "Bus9", 0.0, 0.11),
            ("Bus8", "Bus10", 0.0, 0.11),
            ("Bus9", "Bus10", 0.0, 0.11),
            ("Bus9", "Bus11", 0.0, 0.2085),
            ("Bus10", "Bus11", 0.0, 0.2085),
            ("Bus6", "Bus11", 0.0, 0.25202),
            ("Bus6", "Bus12", 0.0, 0.19207),
            ("Bus6", "Bus13", 0.0, 0.19207),
            ("Bus12", "Bus13", 0.0, 0.19207),
        ]
        
        for i, (b0, b1, r, x) in enumerate(lines, 1):
            n.add("Line", f"Line{i}", bus0=b0, bus1=b1, r=r, x=x, s_nom=1.0)
        
        for i, (bus, p_nom) in enumerate([("Bus1", 332.4), ("Bus2", 140.0), 
                                           ("Bus3", 100.0), ("Bus6", 100.0), ("Bus8", 100.0)], 1):
            n.add("Generator", f"Gen{i}", bus=bus, p_nom=p_nom, control="Slack" if i==1 else "PQ")
        
        for i, (bus, p, q) in enumerate([("Bus2", 21.7, 12.7), ("Bus3", 94.2, 19.0), ("Bus4", 47.8, -3.9),
            ("Bus5", 7.6, 1.6), ("Bus6", 11.2, 7.5), ("Bus9", 29.5, 16.6),
            ("Bus10", 9.0, 5.8), ("Bus11", 3.5, 1.8), ("Bus12", 6.1, 1.6),
            ("Bus13", 13.5, 5.8), ("Bus14", 14.9, 5.0)], 1):
            n.add("Load", f"Load{i}", bus=bus, p_set=p, q_set=q)
        
        n.set_snapshots(pd.date_range("2024-01-01", periods=1, freq="h"))
        
    def _run_pf(self):
        try:
            self.network.pf(snapshots=self.network.snapshots[:1], 
                          distributed_slack=True)
            self.pf_converged = True
        except:
            self.pf_converged = False
            
    def get_state(self) -> GridState:
        """Extracting current grid state"""
        if not self.pf_converged:
            return None
            
        voltages = self.network.buses_t.v_mag_pu.values[0]
        angles = self.network.buses_t.v_ang.values[0] if hasattr(self.network.buses_t, 'v_ang') else np.zeros(14)
        
        line_s = self.network.lines_t.p0.values[0] if hasattr(self.network.lines_t, 'p0') else np.zeros(20)
        line_ratings = self.network.lines.s_nom.values
        loadings = np.abs(line_s) / (line_ratings + 1e-6)
        
        gen_outputs = self.network.generators.p_set.values
        
        loads = self.network.loads.p_set.values
        
        return GridState(
            bus_voltages=voltages,
            phase_angles=angles,
            line_loadings=loadings,
            generator_outputs=gen_outputs,
            loads=loads,
            frequency_deviation=0.0,
            instability_score=0.0
        )
    
    def apply_disturbance(self, disturbance_type: str, magnitude: float = 1.0):
        """Applying a disturbance to the grid"""
        if disturbance_type == "load_increase":
            self.network.loads.p_set *= magnitude
        elif disturbance_type == "line_trip":
            lines = self.network.lines.index.tolist()
            if len(lines) > 0:
                self.network.lines.at[lines[0], 'operational_limit'] = 0.0
        elif disturbance_type == "gen_outage":
            gens = self.network.generators.index.tolist()
            if len(gens) > 1:
                self.network.generators.at[gens[1], 'p_min'] = 0.0
                self.network.generators.at[gens[1], 'p_max'] = 0.0
        
        self._run_pf()
        
    def apply_action(self, action: np.ndarray) -> float:
        """
        Apply soft control action
        action[0:5]: generator redispatch (p_nom adjustments)
        action[5:9]: transformer tap adjustments
        """
        reward = 0.0
        
        # Generator redispatch
        for i, gen in enumerate(self.network.generators.index[:5]):
            if i < min(5, len(action)):
                adjustment = action[i] * 10.0  # Scale factor
                self.network.generators.at[gen, 'p_nom'] += adjustment
                
        self._run_pf()
        
        
    
    def reset(self):
        """Reseting grid to base state"""
        self._build_grid()
        self._run_pf()
        
    def get_observation(self) -> np.ndarray:
        """Getting normalized observation vector"""
        state = self.get_state()
        if state is None:
            return np.zeros(54)
        
        obs = np.concatenate([
            state.bus_voltages,
            state.phase_angles,
            state.line_loadings,
            state.generator_outputs / 200.0,
            state.loads / 100.0,
            [state.frequency_deviation],
            [state.instability_score]
        ])
        return obs


class GridGuardEnv(gym.Env):
    """Gymnasium environment for GridGuard RL training"""
    
    def __init__(self, max_steps: int = 100):
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0
        
        self.grid = IEEE14BusGrid()
        
        # Action space: continuous for soft actions
        # [gen1, gen2, gen3, gen4, gen5, tap1, tap2, tap3, tap4]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )
        
        # Observation space: 54-dimensional
        # 14 voltages + 14 angles + 20 line loadings + 5 gen outputs + 11 loads = 54
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(54,), dtype=np.float32
        )
        
        self.episode_rewards = []
        self.cumulative_violations = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.grid.reset()
        self.current_step = 0
        self.cumulative_violations = 0
        
        # Apply random initial disturbance
        if options and 'disturbance' in options:
            self.grid.apply_disturbance(options['disturbance'], 
                                       options.get('magnitude', 1.2))
        
        obs = self.grid.get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        self.current_step +=1
        self.grid.apply_action(action)
        obs=self.grid.get_observation()
        state=self.grid.get_state()

        reward=0.0
        terminated=False
        truncated=False

        if state is not None:
            voltage_dev=np.abs(state.bus_voltages - 1.0)
            voltage_violation=np.sum(voltage_dev > 0.1)
            loading_violation=np.sum(state.line_loadings > 0.0)
            thermal_violation = np.sum(state.line_loadings > 1.0)

            self.cumulative_violations += voltage_violation + loading_violation

            reward -= (voltage_violation * 5 + loading_violation * 3 + thermal_violation * 10)
            voltage_quality = np.mean(np.maximum(0,1.0 - voltage_dev/0.1))
            reward += voltage_quality * 3.0
            if thermal_violation > 0 or np.any(state.bus_voltages < 0.8):
                terminated = True
                reward -= 100
        reward += 1.0
        reward=float(np.clip(reward,-50.0,10.0))
        if self.current_step >= self.max_steps:
            truncated = True

        info = {
            'violations':self.cumulative_violations,
            'step':self.current_step,
            'pf_converged':self.grid.pf_converged
        } 
        return obs,reward,terminated,truncated,info
       
    
    def render(self):
        state = self.grid.get_state()
        if state is not None:
            print(f"Step {self.current_step}")
            print(f"  Voltages: min={state.bus_voltages.min():.3f}, max={state.bus_voltages.max():.3f}")
            print(f"  Line loadings: max={state.line_loadings.max():.3f}")
            print(f"  Violations: {self.cumulative_violations}")
