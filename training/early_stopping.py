

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import numpy as np
from typing import Dict, List
import os


class PerformanceEarlyStoppingCallback(BaseCallback):
    
    
    def __init__(
        self,
        target_blackout_rate: float = 0.10,
        target_freq_deviation: float = 0.5,
        patience: int = 3,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        verbose: int = 1
    ):
        
        super().__init__()
        self.target_blackout_rate = target_blackout_rate
        self.target_freq_deviation = target_freq_deviation
        self.patience = patience
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose
        
       
        self.eval_count = 0
        self.improvement_count = 0
        self.best_blackout_rate = float('inf')
        self.best_freq_deviation = float('inf')
        self.history = []
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            self._evaluate()
        
        return True
    
    def _evaluate(self):
        self.eval_count += 1
        
        blackout_rate, freq_deviation = self._compute_metrics()
        
        if self.verbose >= 1:
            print(f"\n[Early Stopping Eval {self.eval_count}] "
                  f"Blackout: {blackout_rate*100:.1f}% (target: {self.target_blackout_rate*100:.1f}%) | "
                  f"Freq Dev: {freq_deviation:.4f}Hz (target: {self.target_freq_deviation:.4f}Hz)")
        
      
        target_met = (blackout_rate <= self.target_blackout_rate and 
                     freq_deviation <= self.target_freq_deviation)
     
        improved = False
        if blackout_rate < self.best_blackout_rate:
            self.best_blackout_rate = blackout_rate
            improved = True
        if freq_deviation < self.best_freq_deviation:
            self.best_freq_deviation = freq_deviation
            improved = True
        
        self.history.append({
            'eval': self.eval_count,
            'step': self.num_timesteps,
            'blackout_rate': blackout_rate,
            'freq_deviation': freq_deviation,
            'target_met': target_met
        })
        
        if target_met:
            self.improvement_count += 1
            if self.verbose >= 1:
                print(f"   TARGETS MET! ({self.improvement_count}/{self.patience})")
            
            if self.improvement_count >= self.patience:
                if self.verbose >= 1:
                    print(f"\n{'='*70}")
                    print(f"EARLY STOPPING: Agent reached performance targets!")
                    print(f"   Blackout rate: {blackout_rate*100:.1f}% (target: {self.target_blackout_rate*100:.1f}%)")
                    print(f"   Freq deviation: {freq_deviation:.4f}Hz (target: {self.target_freq_deviation:.4f}Hz)")
                    print(f"   Training steps: {self.num_timesteps:,}")
                    print(f"   Stopped after {self.eval_count} evaluations")
                    print(f"{'='*70}\n")
                return False  
        else:
            self.improvement_count = 0  
            if self.verbose >= 1 and not improved:
                print(f"   No improvement | Patience: {self.improvement_count}/{self.patience}")
        
        return True 
    
    def _compute_metrics(self) -> tuple:
        
        from env.grid_env import GridEnv
        
        blackout_counts = []
        freq_deviations = []
        
        for episode in range(self.n_eval_episodes):
            env = GridEnv()
            obs, _ = env.reset()
            done = False
            had_blackout = False
            freq_devs = []
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                freq_dev = abs(obs[0] - 50.0)
                freq_devs.append(freq_dev)
                
                if terminated:
                    had_blackout = True
            
            blackout_counts.append(1 if had_blackout else 0)
            freq_deviations.append(np.mean(freq_devs) if freq_devs else 0)
        
        blackout_rate = np.mean(blackout_counts)
        mean_freq_deviation = np.mean(freq_deviations)
        
        return blackout_rate, mean_freq_deviation
    
    def save_history(self, filepath: str):
        """Save evaluation history to file"""
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


class RewardEarlyStoppingCallback(BaseCallback):
    
    
    def __init__(
        self,
        target_reward: float = 50.0,
        patience: int = 3,
        eval_freq: int = 5000,
        verbose: int = 1
    ):
        super().__init__()
        self.target_reward = target_reward
        self.patience = patience
        self.eval_freq = eval_freq
        self.verbose = verbose
        
        self.eval_count = 0
        self.best_reward = float('-inf')
        self.patience_counter = 0
        self.reward_history = []
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            return self._check_reward()
        return True
    
    def _check_reward(self) -> bool:
        self.eval_count += 1
        
        if len(self.model.ep_info_buffer) > 0:
            recent_rewards = [
                ep_info['r'] for ep_info in list(self.model.ep_info_buffer)[-10:]
            ]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        else:
            avg_reward = 0
        
        self.reward_history.append(avg_reward)
        
        if self.verbose >= 1:
            print(f"[Reward Eval {self.eval_count}] Avg Reward: {avg_reward:.2f} "
                  f"(target: {self.target_reward:.2f})")
        
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.patience_counter = 0
            if self.verbose >= 1:
                print(f"   New best reward!")
        else:
            self.patience_counter += 1
            if self.verbose >= 1:
                print(f"   No improvement ({self.patience_counter}/{self.patience})")
        
        
        if avg_reward >= self.target_reward and self.patience_counter >= self.patience:
            if self.verbose >= 1:
                print(f"\nEARLY STOPPING: Target reward reached with plateau!")
                print(f"   Reward: {avg_reward:.2f}")
                print(f"   Steps: {self.num_timesteps:,}\n")
            return False
        
        return True
