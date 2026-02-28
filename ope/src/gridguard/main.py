import numpy as np
import argparse
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridguard.mock_env import GridGuardEnv
from gridguard.scenario_generator import ScenarioGenerator
from gridguard.anomaly_detector import EarlyWarningSystem
from gridguard.agent import GridGuardAgent, RecommendationEngine


class GridGuardSystem:
    """Main GridGuard system integrating all components"""
    
    def __init__(self, model_path: str = None):
        self.env = GridGuardEnv(max_steps=100)
        self.scenario_gen = ScenarioGenerator(np.random.default_rng(42))
        self.warning_system = EarlyWarningSystem()
        self.recommendation_engine = RecommendationEngine()
        self.agent = None
        
        if model_path and os.path.exists(model_path):
            self.agent = GridGuardAgent(self.env)
            self.agent.load(model_path)
            
    def initialize(self):
        """Initialize and train components"""
        print("Generating training data...")
        
        # Generate data from environment with proper labels
        X_env = []
        y_env = []
        
        for _ in range(500):
            obs, _ = self.env.reset()
            
            # Check if this is a "challenging" state (random label based on obs)
            # In practice, we'd label based on actual grid state
            voltage_min = np.min(obs[:14])
            loading_max = np.max(obs[14:34])
            
            # Label as unstable if voltages are low or loadings are high
            label = 1 if (voltage_min < 0.95 or loading_max > 0.7) else 0
            
            X_env.append(obs)
            y_env.append(label)
        
        X_train = np.array(X_env)
        y_train = np.array(y_env)
        
        print(f"Training anomaly detector on {len(X_train)} samples...")
        print(f"  Unstable: {sum(y_train)}, Stable: {len(y_train) - sum(y_train)}")
        self.warning_system.train(X_train, y_train)
        
        print("Initializing RL agent...")
        self.agent = GridGuardAgent(self.env)
        
    def assess_grid_state(self, obs: np.ndarray) -> Dict:
        """Assess current grid state and provide recommendations"""
        
        warning = self.warning_system.assess(obs)
        
        if self.agent is not None:
            agent_rec = self.agent.get_recommendation(obs)
            warning['agent_recommendations'] = agent_rec
            
            tier_recs = self.recommendation_engine.get_tier_recommendation(
                warning['instability_score']
            )
            warning['tier_recommendations'] = tier_recs
            
        return warning
    
    def run_interactive(self):
        """Run interactive assessment mode"""
        print("\n" + "="*50)
        print("GridGuard Interactive Assessment Mode")
        print("="*50)
        
        obs, _ = self.env.reset()
        
        for step in range(50):
            print(f"\n--- Step {step} ---")
            
            assessment = self.assess_grid_state(obs)
            
            print(f"Instability Score: {assessment['instability_score']:.3f}")
            print(f"Alert Level: {assessment.get('level', 'N/A')}")
            
            if 'agent_recommendations' in assessment:
                print("\nRecommended Actions:")
                for rec in assessment['agent_recommendations']['recommendations']:
                    print(f"  - {rec}")
                    
            if 'tier_recommendations' in assessment:
                print("\nTier-based Recommendations:")
                for tier_rec in assessment['tier_recommendations']:
                    print(f"  - {tier_rec['action']} (impact: {tier_rec['impact']})")
            
            action = self.env.action_space.sample() if self.agent is None else \
                     self.agent.predict(obs, deterministic=True)[0]
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated:
                print("\n!!! GRID COLLAPSE !!!")
                break
                
            if truncated:
                print("\nScenario completed successfully")
                break
                
    def train(self, timesteps: int = 50000):
        """Train the RL agent"""
        
        if self.agent is None:
            self.agent = GridGuardAgent(self.env)
            
        print(f"Training agent for {timesteps} timesteps...")
        self.agent.train(total_timesteps=timesteps)
        print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="GridGuard - Grid Stabilization System")
    parser.add_argument('--mode', choices=['train', 'assess', 'demo'], default='demo',
                       help='Operating mode')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Training timesteps')
    
    args = parser.parse_args()
    
    system = GridGuardSystem(model_path=args.model)
    
    if args.mode == 'train':
        system.initialize()
        system.train(timesteps=args.timesteps)
    elif args.mode == 'assess':
        system.run_interactive()
    else:  # demo
        print("\n" + "="*50)
        print("GridGuard Demo Mode")
        print("="*50)
        
        system.initialize()
        
        print("\nGenerating scenario...")
        scenarios = system.scenario_gen.generate_batch(3, difficulty='medium')
        
        for i, scenario in enumerate(scenarios):
            print(f"\nScenario {i+1}: {scenario['disturbance_type']}")
            print(f"  Severity: {scenario['severity']:.2f}")
            print(f"  Recommended: {scenario['recommended_action']}")
            
        print("\nDemo complete. Use --mode assess for interactive mode.")


if __name__ == "__main__":
    main()
