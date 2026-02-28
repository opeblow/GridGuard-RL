
import argparse
import json
import logging
import numpy as np
import os
import tempfile

from env.grid_env import GridEnv


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class StressEnv(GridEnv):
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

        if self.scenario == 'load_surge' and self._step == 10:
            
            self.grid.change_load(500)
        elif self.scenario == 'noise':
            noise = float(np.random.normal(0.0, 50.0))
            self.grid.change_load(noise)
        elif self.scenario == 'generator_outage' and self._step == 10:
           
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
    recovery_tol = 0.1  
    recovery_window = 3  
    collapse_threshold = 2.0  
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

    max_dev = float(np.max(freq_devs) if freq_devs else 0.0)
    collapsed = bool(blackout or (max_dev > collapse_threshold))

    recovery_time = None
    if not collapsed and perturb_step is not None and perturb_step < len(freq_history):
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
                env_instance = StressEnv(scenario=scenario)
                m = run_episode(policy_fn, env_instance)
                metrics.append(m)

         
            rewards = [m['reward'] for m in metrics]
            lengths = [m['length'] for m in metrics]
            blackouts = [m['blackout'] for m in metrics]
            freq_devs = [m['mean_freq_dev'] for m in metrics]
            max_devs = [m.get('max_freq_dev', 0.0) for m in metrics]
            collapsed = [m.get('collapsed', False) for m in metrics]
            recovery_times = [m.get('recovery_time') for m in metrics if m.get('recovery_time') is not None]
            collapse_thresholds = [m.get('collapse_threshold', 0.0) for m in metrics]

            results[name][scenario] = {
                'episodes': metrics, 
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'mean_length': float(np.mean(lengths)),
                'blackout_rate': float(np.mean(blackouts)) * 100.0,
                'collapse_rate': float(np.mean(collapsed)) * 100.0,
                'mean_freq_deviation': float(np.mean(freq_devs)),
                'mean_max_freq_deviation': float(np.mean(max_devs)),
                'mean_recovery_time': (float(np.mean(recovery_times)) if recovery_times else None),
                'uptime_percentage': float((1.0 - np.mean(blackouts)) * 100.0),
                'collapse_threshold': float(np.mean(collapse_thresholds) if collapse_thresholds else 0.0),
            }

    return results


def main(model_path=None, n_episodes=10, out_file='logs/stress_test_results.json'):
    policies = {}

    tmp_env = StressEnv()
    policies['random'] = random_policy_factory(tmp_env)

   
    policies['pid'] = pid_policy_factory(tmp_env)

    
    if model_path is not None and os.path.exists(model_path):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            policies['ppo'] = model_policy_factory(model)
        except Exception as e:
            print(f"Failed to load PPO model: {e}")

    scenarios = ['load_surge', 'noise', 'generator_outage']

    logger.info("Running stress tests for policies: %s", list(policies.keys()))
    results = evaluate_policies_on_scenarios(policies, scenarios, n_episodes=n_episodes)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Stress test results saved to: %s", out_file)

    if model_path is not None and os.path.exists(model_path):
        model_dir = os.path.dirname(model_path)
        final_metrics_path = os.path.join(model_dir, "final_metrics.json")
    else:
        final_metrics_path = os.path.join(os.path.dirname(out_file), "final_metrics.json")
    final_data = {}
    if os.path.exists(final_metrics_path):
        try:
            with open(final_metrics_path, "r") as f:
                final_data = json.load(f)
        except Exception:
            final_data = {}

    final_data.setdefault("stress_tests", {})
    import datetime

    run_id = datetime.datetime.utcnow().isoformat() + "Z"

    units_map = {
        'mean_freq_deviation': 'Hz',
        'mean_max_freq_deviation': 'Hz',
        'blackout_rate': '%',
        'collapse_rate': '%',
        'uptime_percentage': '%',
        'mean_reward': 'reward',
        'std_reward': 'reward',
        'mean_length': 'steps',
        'mean_recovery_time': 'steps',
    }

    final_data["stress_tests"][run_id] = {
        'results': results,
        'units': units_map,
        'scenarios': list(scenarios),
    }

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="final_metrics_", dir=os.path.dirname(final_metrics_path))
    try:
        with os.fdopen(tmp_fd, "w") as tf:
            json.dump(final_data, tf, indent=2)
        os.replace(tmp_path, final_metrics_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    logger.info("Merged stress test results into: %s", final_metrics_path)
    for pname, pmetrics in results.items():
        logger.info("%s", "=" * 60)
        logger.info("Policy: %s", pname)
        for scen, stats in pmetrics.items():
            logger.info(
                "  Scenario: %s | Mean Reward: %.2f | Blackout Rate: %.1f%% | Mean Freq Dev: %.4f Hz",
                scen,
                stats["mean_reward"],
                stats["blackout_rate"],
                stats["mean_freq_deviation"],
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run stress-test evaluations and compare baselines')
    parser.add_argument('--model', type=str, default=None, help='Path to PPO model zip file (optional)')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per scenario')
    parser.add_argument('--out', type=str, default='logs/stress_test_results.json', help='Output JSON file')
    args = parser.parse_args()
    main(model_path=args.model, n_episodes=args.episodes, out_file=args.out)
