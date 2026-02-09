
import argparse
import json
import numpy as np
import os

from env.grid_env import GridEnv


class StressEnv(GridEnv):
    """Wrapper that applies a scenario perturbation during an episode."""
    def __init__(self, scenario=None, **kwargs):
        super().__init__(**kwargs)
        self.scenario = scenario
        self._step = 0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._step = 0
        return obs, info

    def step(self, action):
        self._step += 1

        # Apply scenario perturbations BEFORE stepping the environment
        if self.scenario == 'load_surge' and self._step == 10:
            # sudden large load increase
            self.grid.change_load(500)
        elif self.scenario == 'noise':
            # small random noise each step
            noise = float(np.random.normal(0.0, 50.0))
            self.grid.change_load(noise)
        elif self.scenario == 'generator_outage' and self._step == 10:
            # large generator drop/outage
            self.grid.generation = max(0.0, self.grid.generation - 800.0)

        return super().step(action)


def run_episode(policy_fn, env, deterministic=True):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    blackout = False
    freq_devs = []
    freq_history = []
    # recovery/collapse heuristics
    recovery_tol = 0.1  # Hz tolerance for considering frequency recovered
    recovery_window = 3  # consecutive steps under tolerance
    collapse_threshold = 2.0  # Hz deviation considered a collapse
    perturb_step = 0
    if hasattr(env, 'scenario') and env.scenario in ('load_surge', 'generator_outage'):
        perturb_step = 10

    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        cur_dev = abs(obs[0] - env.nominal_frequency)
        freq_devs.append(cur_dev)
        freq_history.append(float(obs[0]))
        if terminated:
            blackout = True
        if terminated:
            blackout = True

    # compute collapse and recovery
    max_dev = float(np.max(freq_devs) if freq_devs else 0.0)
    collapsed = bool(blackout or (max_dev > collapse_threshold))

    recovery_time = None
    if not collapsed and perturb_step is not None and perturb_step < len(freq_history):
        # find first step after perturb when freq stays within tolerance for recovery_window
        for t in range(perturb_step, len(freq_history)):
            window = freq_history[t:t+recovery_window]
            if len(window) < recovery_window:
                break
            if all(abs(f - env.nominal_frequency) <= recovery_tol for f in window):
                recovery_time = int(t - perturb_step)
                break

    return {
        'reward': float(total_reward),
        'length': int(steps),
        'blackout': bool(blackout),
        'mean_freq_dev': float(np.mean(freq_devs) if freq_devs else 0.0),
        'max_freq_dev': max_dev,
        'collapsed': bool(collapsed),
        'recovery_time': (None if recovery_time is None else int(recovery_time)),
        'collapse_threshold': float(collapse_threshold),
    }


def random_policy_factory(env):
    def policy(_obs):
        return env.action_space.sample()
    return policy


def pid_policy_factory(env, Kp=50.0, Ki=5.0, Kd=10.0):
    integral = 0.0
    last_error = 0.0

    def policy(obs):
        nonlocal integral, last_error
        target = env.nominal_frequency
        error = target - float(obs[0])
        integral += error
        derivative = error - last_error
        last_error = error
        adjustment = Kp * error + Ki * integral + Kd * derivative
        # action is delta generation; keep within action_space
        action = np.clip(adjustment, env.action_space.low[0], env.action_space.high[0])
        return np.array([action], dtype=np.float32)

    return policy


def model_policy_factory(model):
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


def evaluate_policies_on_scenarios(policies, scenarios, n_episodes=10):
    results = {}
    for name, policy_fn in policies.items():
        results[name] = {}
        for scenario in scenarios:
            metrics = []
            env = StressEnv(scenario=scenario)
            for _ in range(n_episodes):
                # For each episode create a fresh env instance to avoid state leakage
                env_instance = StressEnv(scenario=scenario)
                m = run_episode(policy_fn, env_instance)
                metrics.append(m)

            # aggregate
            rewards = [m['reward'] for m in metrics]
            lengths = [m['length'] for m in metrics]
            blackouts = [m['blackout'] for m in metrics]
            freq_devs = [m['mean_freq_dev'] for m in metrics]
            max_devs = [m.get('max_freq_dev', 0.0) for m in metrics]
            collapsed = [m.get('collapsed', False) for m in metrics]
            recovery_times = [m.get('recovery_time') for m in metrics if m.get('recovery_time') is not None]

            results[name][scenario] = {
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'mean_length': float(np.mean(lengths)),
                'blackout_rate': float(np.mean(blackouts)) * 100.0,
                'collapse_rate': float(np.mean(collapsed)) * 100.0,
                'mean_freq_deviation': float(np.mean(freq_devs)),
                'mean_max_freq_deviation': float(np.mean(max_devs)),
                'mean_recovery_time': (float(np.mean(recovery_times)) if recovery_times else None),
                'uptime_percentage': float((1.0 - np.mean(blackouts)) * 100.0),
            }

    return results


def main(model_path=None, n_episodes=10, out_file='logs/stress_test_results.json'):
    policies = {}

    # Random baseline
    tmp_env = StressEnv()
    policies['random'] = random_policy_factory(tmp_env)

    # PID baseline
    policies['pid'] = pid_policy_factory(tmp_env)

    # PPO model (optional)
    if model_path is not None and os.path.exists(model_path):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            policies['ppo'] = model_policy_factory(model)
        except Exception as e:
            print(f"Failed to load PPO model: {e}")

    scenarios = ['load_surge', 'noise', 'generator_outage']

    print(f"Running stress tests for policies: {list(policies.keys())}")
    results = evaluate_policies_on_scenarios(policies, scenarios, n_episodes=n_episodes)

    # Save per-run results (backwards compatible)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Stress test results saved to: {out_file}")

    # Merge into logs/final_metrics.json under key 'stress_tests', preserving existing metrics
    final_metrics_path = os.path.join(os.path.dirname(out_file), 'final_metrics.json')
    final_data = {}
    if os.path.exists(final_metrics_path):
        try:
            with open(final_metrics_path, 'r') as f:
                final_data = json.load(f)
        except Exception:
            final_data = {}

    final_data.setdefault('stress_tests', {})
    # attach a timestamped entry for this run
    import datetime
    run_id = datetime.datetime.utcnow().isoformat() + 'Z'
    final_data['stress_tests'][run_id] = results

    with open(final_metrics_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    print(f"Merged stress test results into: {final_metrics_path}")
    for pname, pmetrics in results.items():
        print('=' * 60)
        print(f"Policy: {pname}")
        for scen, stats in pmetrics.items():
            print(f"  Scenario: {scen} | Mean Reward: {stats['mean_reward']:.2f} | Blackout Rate: {stats['blackout_rate']:.1f}% | Mean Freq Dev: {stats['mean_freq_deviation']:.4f} Hz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run stress-test evaluations and compare baselines')
    parser.add_argument('--model', type=str, default=None, help='Path to PPO model zip file (optional)')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per scenario')
    parser.add_argument('--out', type=str, default='logs/stress_test_results.json', help='Output JSON file')
    args = parser.parse_args()
    main(model_path=args.model, n_episodes=args.episodes, out_file=args.out)
