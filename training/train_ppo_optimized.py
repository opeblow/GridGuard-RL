"""
Optimized PPO Training for GridGuard Power Grid RL Agent
Supports FAST_MODE (2-5 min) and FULL_MODE (30-60 min) configurations
with early stopping for intelligent training termination.

USAGE:
  python train_ppo.py --mode fast    # For rapid iteration
  python train_ppo.py --mode full    # For production quality
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from datetime import datetime
import json

from env.grid_env import GridEnv
from training.train_config import FastModeConfig, FullModeConfig, TrainingConfig
from training.early_stopping import PerformanceEarlyStoppingCallback


def _make_json_serializable(obj):
    """Recursively convert numpy types to native Python types so json.dump succeeds."""
    # numpy scalar
    if isinstance(obj, np.generic):
        return obj.item()
    # dict
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # other types (int/float/bool/None/str) - return as-is
    return obj


class TensorboardCallback:
    """Custom callback for logging additional metrics to Tensorboard"""
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
        return True


def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True):
    """
    Evaluate a trained policy
    
    Args:
        model: Trained model
        env: Environment to evaluate on
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Use deterministic actions
        
    Returns:
        dict: Evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    blackout_counts = []
    frequency_deviations = []
    
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        had_blackout = False
        freq_devs = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            freq_dev = abs(obs[0] - 50.0)
            freq_devs.append(freq_dev)
            
            if terminated:
                had_blackout = True
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        blackout_counts.append(1 if had_blackout else 0)
        frequency_deviations.append(np.mean(freq_devs) if freq_devs else 0)
    
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'blackout_rate': np.mean(blackout_counts) * 100,
        'mean_freq_deviation': np.mean(frequency_deviations),
        'uptime_percentage': (1 - np.mean(blackout_counts)) * 100
    }
    
    return metrics


def evaluate_random_policy(env, n_eval_episodes=10):
    """Evaluate a random policy as baseline"""
    episode_rewards = []
    episode_lengths = []
    blackout_counts = []
    frequency_deviations = []
    
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        had_blackout = False
        freq_devs = []
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            freq_dev = abs(obs[0] - 50.0)
            freq_devs.append(freq_dev)
            
            if terminated:
                had_blackout = True
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        blackout_counts.append(1 if had_blackout else 0)
        frequency_deviations.append(np.mean(freq_devs) if freq_devs else 0)
    
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'blackout_rate': np.mean(blackout_counts) * 100,
        'mean_freq_deviation': np.mean(frequency_deviations),
        'uptime_percentage': (1 - np.mean(blackout_counts)) * 100
    }
    
    return metrics


def evaluate_pid_policy(env, n_eval_episodes=10):
    """Evaluate a PID controller as baseline"""
    episode_rewards = []
    episode_lengths = []
    blackout_counts = []
    frequency_deviations = []
    
    Kp = 100.0
    Ki = 10.0
    Kd = 20.0
    
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        had_blackout = False
        freq_devs = []
        
        last_error = 0
        integral = 0
        
        while not done:
            target_freq = 50.0
            error = target_freq - obs[0]
            
            integral += error
            derivative = error - last_error
            
            adjustment = Kp * error + Ki * integral + Kd * derivative
            action = np.array([obs[1] + adjustment], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            freq_dev = abs(obs[0] - 50.0)
            freq_devs.append(freq_dev)
            
            last_error = error
            
            if terminated:
                had_blackout = True
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        blackout_counts.append(1 if had_blackout else 0)
        frequency_deviations.append(np.mean(freq_devs) if freq_devs else 0)
    
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'blackout_rate': np.mean(blackout_counts) * 100,
        'mean_freq_deviation': np.mean(frequency_deviations),
        'uptime_percentage': (1 - np.mean(blackout_counts)) * 100
    }
    
    return metrics


def main(mode: str = "fast"):
    """
    Main training function with configurable mode
    
    Args:
        mode: 'fast' for rapid iteration or 'full' for production quality
    """
    # Load configuration
    config = TrainingConfig.get_config(mode)
    
    print("=" * 80)
    print(f"  POWER GRID RL TRAINING - {config.MODE_NAME} MODE")
    print(f"  Expected Duration: {config.EXPECTED_DURATION}")
    print(f"  Use Case: {config.USE_CASE}")
    print("=" * 80)
    print()
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/ppo_{config.MODE_NAME.lower()}_{timestamp}"
    model_dir = f"models/ppo_{config.MODE_NAME.lower()}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f" Log directory: {log_dir}")
    print(f" Model directory: {model_dir}")
    print()
    
    # Create environments
    print(" Creating environment...")
    env = GridEnv()
    env = Monitor(env, log_dir)
    
    eval_env = GridEnv()
    eval_env = Monitor(eval_env, log_dir)
    
    print(" Environment created successfully")
    print()
    
    # Baseline evaluation (skip in fast mode)
    baselines = {}
    if not config.SKIP_BASELINE_EVAL:
        print("=" * 80)
        print(" BASELINE EVALUATION (Before RL Training)")
        print("=" * 80)
        print()
        
        print("Evaluating Random Policy...")
        random_metrics = evaluate_random_policy(eval_env, n_eval_episodes=config.N_EVAL_EPISODES)
        print(f"   Mean Reward: {random_metrics['mean_reward']:.2f} ± {random_metrics['std_reward']:.2f}")
        print(f"   Blackout Rate: {random_metrics['blackout_rate']:.1f}%")
        print(f"   Avg Freq Deviation: {random_metrics['mean_freq_deviation']:.4f} Hz")
        print(f"   Uptime: {random_metrics['uptime_percentage']:.1f}%")
        print()
        
        print("Evaluating PID Controller...")
        pid_metrics = evaluate_pid_policy(eval_env, n_eval_episodes=config.N_EVAL_EPISODES)
        print(f"   Mean Reward: {pid_metrics['mean_reward']:.2f} ± {pid_metrics['std_reward']:.2f}")
        print(f"   Blackout Rate: {pid_metrics['blackout_rate']:.1f}%")
        print(f"   Avg Freq Deviation: {pid_metrics['mean_freq_deviation']:.4f} Hz")
        print(f"   Uptime: {pid_metrics['uptime_percentage']:.1f}%")
        print()
        
        baselines = {'random': random_metrics, 'pid': pid_metrics}
        with open(f"{model_dir}/baseline_metrics.json", 'w') as f:
            json.dump(_make_json_serializable(baselines), f, indent=4)
    else:
        print(" [SKIPPED] Baseline evaluation (Fast mode)")
        print()
    
    # Training configuration
    print("=" * 80)
    print(" PPO HYPERPARAMETERS")
    print("=" * 80)
    print()
    
    hyperparams = config.PPO_PARAMS
    for key, value in hyperparams.items():
        if key != 'policy_kwargs':
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value['net_arch']}")
    
    print(f"   Training timesteps: {config.TOTAL_TIMESTEPS:,}")
    print(f"   Evaluation frequency: {config.EVAL_FREQ:,} steps")
    print(f"   Checkpoint frequency: {config.SAVE_FREQ:,} steps")
    print(f"   Eval episodes per cycle: {config.N_EVAL_EPISODES}")
    print()
    
    # Early stopping configuration
    print("=" * 80)
    print(" EARLY STOPPING CONFIGURATION")
    print("=" * 80)
    print()
    if config.EARLY_STOPPING_ENABLED:
        print(f"   Target Blackout Rate: < {config.TARGET_BLACKOUT_RATE*100:.1f}%")
        print(f"   Target Freq Deviation: < {config.TARGET_MEAN_FREQ_DEV:.4f} Hz")
        print(f"   Patience (evals): {config.EARLY_STOPPING_PATIENCE}")
        print(f"   Status: ENABLED ")
    else:
        print(f"   Status: DISABLED")
    print()
    
    # Create and configure PPO model
    print(" Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hyperparams['learning_rate'],
        n_steps=hyperparams['n_steps'],
        batch_size=hyperparams['batch_size'],
        n_epochs=hyperparams['n_epochs'],
        gamma=hyperparams['gamma'],
        gae_lambda=hyperparams['gae_lambda'],
        clip_range=hyperparams['clip_range'],
        ent_coef=hyperparams['ent_coef'],
        vf_coef=hyperparams['vf_coef'],
        max_grad_norm=hyperparams['max_grad_norm'],
        policy_kwargs=hyperparams['policy_kwargs'],
        verbose=0,
        tensorboard_log=log_dir if not config.SKIP_INTERMEDIATE_LOGGING else None
    )
    
    print(" Model initialized")
    print()
    
    # Setup callbacks
    print(" Setting up callbacks...")
    
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.SAVE_FREQ,
        save_path=model_dir,
        name_prefix='ppo_checkpoint',
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if config.EARLY_STOPPING_ENABLED:
        early_stopping = PerformanceEarlyStoppingCallback(
            target_blackout_rate=config.TARGET_BLACKOUT_RATE,
            target_freq_deviation=config.TARGET_MEAN_FREQ_DEV,
            patience=config.EARLY_STOPPING_PATIENCE,
            eval_freq=config.EVAL_FREQ,
            n_eval_episodes=config.N_EVAL_EPISODES,
            verbose=1
        )
        callbacks.append(early_stopping)
    
    callback_list = CallbackList(callbacks)
    
    print(f"   Callbacks configured: EvalCallback, CheckpointCallback", end="")
    if config.EARLY_STOPPING_ENABLED:
        print(", EarlyStoppingCallback")
    else:
        print()
    print()
    
    # Training
    print("=" * 80)
    print(" STARTING TRAINING")
    print("=" * 80)
    print()
    
    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=callback_list,
            progress_bar=False,
            tb_log_name="ppo_fast"
        )
        
        print()
        print("=" * 80)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        
    except KeyboardInterrupt:
        print()
        print(" Training interrupted by user")
        print(" Saving current model...")
        model.save(f"{model_dir}/interrupted_model")
        print(" Model saved")
        return
    
    # Save final model
    final_model_path = f"{model_dir}/final_model"
    model.save(final_model_path)
    print(f" Final model saved to: {final_model_path}")
    print()
    
    # Final evaluation
    print("=" * 80)
    print(" FINAL EVALUATION (Best Model)")
    print("=" * 80)
    print()
    
    best_model_path = f"{model_dir}/best_model"
    if os.path.exists(f"{best_model_path}.zip"):
        print(f" Loading best model from: {best_model_path}")
        best_model = PPO.load(best_model_path)
        
        print(f" Evaluating trained PPO agent ({config.FINAL_EVAL_EPISODES} episodes)...")
        ppo_metrics = evaluate_policy(best_model, eval_env, n_eval_episodes=config.FINAL_EVAL_EPISODES)
        print(f"   Mean Reward: {ppo_metrics['mean_reward']:.2f} ± {ppo_metrics['std_reward']:.2f}")
        print(f"   Blackout Rate: {ppo_metrics['blackout_rate']:.1f}%")
        print(f"   Avg Freq Deviation: {ppo_metrics['mean_freq_deviation']:.4f} Hz")
        print(f"   Uptime: {ppo_metrics['uptime_percentage']:.1f}%")
        print()
        
        # Save final metrics
        all_metrics = {
            'ppo': ppo_metrics,
            'baselines': baselines,
            'hyperparameters': hyperparams,
            'training_config': {
                'mode': config.MODE_NAME,
                'total_timesteps': config.TOTAL_TIMESTEPS,
                'eval_freq': config.EVAL_FREQ,
                'early_stopping_enabled': config.EARLY_STOPPING_ENABLED,
            }
        }
        
        with open(f"{model_dir}/final_metrics.json", 'w') as f:
            json.dump(_make_json_serializable(all_metrics), f, indent=4)
        
        # Performance comparison
        if baselines:
            print("=" * 80)
            print(" PERFORMANCE COMPARISON")
            print("=" * 80)
            print()
            print(f"{'Metric':<25} {'Random':<15} {'PID':<15} {'PPO (Ours)':<15}")
            print("-" * 80)
            
            random_metrics = baselines.get('random', {})
            pid_metrics = baselines.get('pid', {})
            
            if random_metrics:
                print(f"{'Mean Reward':<25} {random_metrics['mean_reward']:<15.2f} "
                      f"{pid_metrics['mean_reward']:<15.2f} {ppo_metrics['mean_reward']:<15.2f}")
                print(f"{'Blackout Rate (%)':<25} {random_metrics['blackout_rate']:<15.1f} "
                      f"{pid_metrics['blackout_rate']:<15.1f} {ppo_metrics['blackout_rate']:<15.1f}")
                print(f"{'Avg Freq Dev (Hz)':<25} {random_metrics['mean_freq_deviation']:<15.4f} "
                      f"{pid_metrics['mean_freq_deviation']:<15.4f} {ppo_metrics['mean_freq_deviation']:<15.4f}")
                print(f"{'Uptime (%)':<25} {random_metrics['uptime_percentage']:<15.1f} "
                      f"{pid_metrics['uptime_percentage']:<15.1f} {ppo_metrics['uptime_percentage']:<15.1f}")
                print()
    
    print("=" * 80)
    print(" OUTPUTS")
    print("=" * 80)
    print()
    print(f" Model Directory: {model_dir}/")
    print(f"   - best_model.zip")
    print(f"   - final_model.zip")
    print(f"   - final_metrics.json")
    print()
    print(" Next steps for portfolio demo:")
    print(f"   1. Load and test: python -c \"from stable_baselines3 import PPO; "
          f"model = PPO.load('{model_dir}/best_model'); print(model)\"")
    print("   2. View metrics: cat " + model_dir + "/final_metrics.json")
    print()
    print(" View training logs:")
    print(f"   tensorboard --logdir {log_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO agent for GridGuard power grid environment"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['fast', 'full'],
        default='fast',
        help='Training mode: fast (2-5 min) or full (30-60 min)'
    )
    
    args = parser.parse_args()
    
    try:
        import stable_baselines3
        print(" stable-baselines3 found ✓\n")
    except ImportError:
        print(" stable-baselines3 not found")
        print(" Install with: pip install stable-baselines3[extra] tensorboard")
        print()
        sys.exit(1)
    
    main(mode=args.mode)
