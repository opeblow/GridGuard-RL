"""
Grid Stability Test with Stress Test
Loads a trained PPO model and runs a 200-step simulation with a load surge at step 100.
Shows how the AI reacts to sudden grid disturbances.

Usage:
    python -m tests.grid_test
"""

import numpy as np
from stable_baselines3 import PPO
from env.grid_env import GridEnv


def run_grid_stability_test(model_path: str = "models/ppo_fast_20260206_125228/best_model", 
                            episode_length: int = 200,
                            stress_step: int = 100,
                            stress_load_increase: float = 500.0):
    """
    Run a grid stability test with a stress test.
    
    Args:
        model_path: Path to trained PPO model
        episode_length: Total simulation steps
        stress_step: Step at which to apply load surge
        stress_load_increase: Amount of load increase in MW
    """
    
    # Load trained model
    try:
        model = PPO.load(model_path)
        print(f"Loaded trained PPO model from: {model_path}\n")
    except FileNotFoundError:
        print(f" Model not found at: {model_path}")
        print("  Run 'python -m training.train_ppo_optimized --mode fast' first\n")
        return
    
    # Create environment
    env = GridEnv(episode_length=episode_length)
    obs, info = env.reset()
    
    print("=" * 90)
    print("GRID STABILITY TEST WITH STRESS TEST")
    print("=" * 90)
    print(f"{'Step':<8} {'Freq (Hz)':<14} {'Gen (MW)':<14} {'Load (MW)':<14} {'Action (Δ MW)':<14} {'Status':<12}")
    print("-" * 90)
    
    total_reward = 0
    episode_ended = False
    blackout_step = None
    
    for step in range(episode_length):
        # Stress test: increase load by 500 MW at specified step
        if step == stress_step:
            print(f"\n{'':8} {'':14} {'':14} {'':14} {'':14} *** STRESS TEST TRIGGERED ***")
            print(f"{'':8} {'':14} {'':14} {'':14} {'':14} Load surge: +{stress_load_increase} MW")
            print(f"{'':8} {'':14} {'':14} {'':14} {'':14} " + "-" * 25)
            env.grid.load += stress_load_increase
        
        # Get agent action
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Determine status for this step
        status = ""
        if step == stress_step:
            status = "SURGE"
        elif terminated:
            status = "BLACKOUT"
            blackout_step = step
        elif truncated:
            status = "END"
        else:
            status = "OK"
        
        # Print every 10 steps, plus steps around the stress test
        should_print = (
            step % 10 == 0 or 
            step == stress_step or 
            step == stress_step - 1 or 
            step == stress_step + 1 or
            (step > stress_step and step <= stress_step + 20 and step % 5 == 0)
        )
        
        if should_print:
            action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            freq_color = "" if 49.5 <= env.grid.frequency <= 50.5 else "" if 48.0 <= env.grid.frequency <= 52.0  else "critical"
            print(f"{step:<8} {env.grid.frequency:<14.4f} {env.grid.generation:<14.2f} {env.grid.load:<14.2f} {action_value:<14.4f} {status:<12} {freq_color}")
        
        if terminated or truncated:
            episode_ended = True
            break
    
    print("-" * 90)
    print()
    print("=" * 90)
    print("TEST RESULTS")
    print("=" * 90)
    print(f"Total Steps Run: {step + 1}/{episode_length}")
    print(f"Total Reward: {total_reward:.2f}")
    print()
    print("Final State:")
    print(f"  Frequency: {env.grid.frequency:.4f} Hz (Target: 50.0 Hz)")
    print(f"  Generation: {env.grid.generation:.2f} MW")
    print(f"  Load: {env.grid.load:.2f} MW")
    print(f"  Power Balance: {env.grid.generation - env.grid.load:.2f} MW")
    print()
    
    if episode_ended and blackout_step is not None:
        print(f" FAILURE: Blackout occurred at step {blackout_step}")
        print(f"   Frequency collapsed to {env.grid.frequency:.2f} Hz")
    elif episode_ended:
        print(f" Episode completed successfully (no blackout)")
    else:
        print(f" TEST PASSED: Grid remained stable throughout 200-step simulation")
        print(f"  AI successfully managed the {stress_load_increase} MW load surge at step {stress_step}")
    
    print()
    print("=" * 90)


if __name__ == "__main__":
    # Run the test
    run_grid_stability_test()
