import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random


@dataclass
class Scenario:
    name: str
    disturbance_type: str
    severity: float
    time_to_collapse: int
    recommended_action: str


class ScenarioGenerator:
    """Generating synthetic training scenarios for grid instability"""
    
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        
    def generate_batch(self, n_scenarios: int, difficulty: str = "medium") -> List[Dict]:
        """Generating a batch of training scenarios"""
        scenarios = []
        
        for _ in range(n_scenarios):
            scenario = self._generate_scenario(difficulty)
            scenarios.append(scenario)
            
        return scenarios
    
    def _generate_scenario(self, difficulty: str) -> Dict:
        """Generating a single scenario"""
        
        if difficulty == "easy":
            severity_range = (1.1, 1.3)
            time_range = (20, 40)
        elif difficulty == "medium":
            severity_range = (1.2, 1.5)
            time_range = (10, 30)
        else:
            severity_range = (1.4, 2.0)
            time_range = (5, 20)
            
        disturbance_types = [
            "load_increase",
            "line_trip",
            "gen_outage",
            "multi_component_failure",
            "cascading_initiation"
        ]
        
        disturbance = self.rng.choice(disturbance_types)
        severity = self.rng.uniform(*severity_range)
        time_to_collapse = int(self.rng.uniform(*time_range))
        
        return {
            "disturbance_type": disturbance,
            "severity": severity,
            "time_to_collapse": time_to_collapse,
            "recommended_action": self._get_recommendation(disturbance, severity),
            "initial_state": self._generate_initial_state(),
            "failure_markers": self._generate_failure_markers(time_to_collapse)
        }
    
    def _generate_initial_state(self) -> Dict:
        """Generating a realistic initial grid state"""
        return {
            "load_multiplier": self.rng.uniform(0.8, 1.2),
            "gen_available": self.rng.uniform(0.85, 1.0),
            "line_health": self.rng.uniform(0.9, 1.0),
            "weather_factor": self.rng.uniform(0.95, 1.05)
        }
    
    def _generate_failure_markers(self, time_to_collapse: int) -> List[Dict]:
        """Generating the progression of failure markers"""
        markers = []
        n_markers = min(5, time_to_collapse // 5)
        
        for i in range(n_markers):
            progress = (i + 1) / n_markers
            markers.append({
                "time_offset": int(time_to_collapse * progress),
                "voltage_dip": 1.0 - (progress * self.rng.uniform(0.05, 0.15)),
                "loading_increase": progress * self.rng.uniform(0.1, 0.3),
                "instability_score": progress * self.rng.uniform(0.3, 0.7)
            })
            
        return markers
    
    def _get_recommendation(self, disturbance: str, severity: float) -> str:
        """Getting recommended action for disturbance type"""
        
        recommendations = {
            "load_increase": "redispatch_generation" if severity < 1.4 else "shed_load_tier1",
            "line_trip": "redispatch_generation",
            "gen_outage": "activate_reserve_generation",
            "multi_component_failure": "island_critical_loads",
            "cascading_initiation": "emergency_load_shedding"
        }
        
        return recommendations.get(disturbance, "monitor")
    
    def generate_instability_sequence(self, base_state: Dict, n_steps: int = 50) -> np.ndarray:
        """Creating a sequence of states leading to instability"""
        sequence = []
        
        voltage = 1.0
        loading = 0.5
        instability = 0.0
        
        for step in range(n_steps):
            progress = step / n_steps
            
            noise_v = self.rng.normal(0, 0.01)
            noise_l = self.rng.normal(0, 0.02)
            
            voltage = max(0.7, voltage - progress * 0.02 + noise_v)
            loading = min(1.2, loading + progress * 0.03 + noise_l)
            instability = progress * self.rng.uniform(0.5, 0.9)
            
            state = np.array([
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                loading + self.rng.uniform(-0.05, 0.05),
                instability,
                progress
            ])
            
            sequence.append(state)
            
        return np.array(sequence)
    
    def generate_stable_sequence(self, base_state: Dict, n_steps: int = 50) -> np.ndarray:
        """Developing a sequence of stable operating states"""
        sequence = []
        
        for step in range(n_steps):
            voltage = 1.0 + self.rng.normal(0, 0.02)
            loading = 0.5 + self.rng.normal(0, 0.1)
            instability = self.rng.uniform(0, 0.1)
            
            state = np.array([
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                voltage + self.rng.uniform(-0.02, 0.02),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                loading + self.rng.uniform(-0.1, 0.1),
                instability,
                0.0
            ])
            
            sequence.append(state)
            
        return np.array(sequence)
    
    def create_training_dataset(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Creating a labeled training dataset"""
        X = []
        y = []
        
        for _ in range(n_samples // 2):
            base_state = self._generate_initial_state()
            
            unstable_seq = self.generate_instability_sequence(base_state)
            X.append(unstable_seq)
            y.append(np.ones(len(unstable_seq)))
            
            stable_seq = self.generate_stable_sequence(base_state)
            X.append(stable_seq)
            y.append(np.zeros(len(stable_seq)))
            
        return np.vstack(X), np.hstack(y)
