import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env.grid_env import GridEnv


def run_audit(model_path: str, steps: int = 200, output_path: str = "grid_performance_audit.png"):
	
	if not os.path.exists(model_path) and os.path.exists(model_path + ".zip"):
		model_path = model_path + ".zip"

	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Model not found at: {model_path}")


	model = PPO.load(model_path)

	env = GridEnv(episode_length=steps)

	obs, _ = env.reset()

	freq_list = []
	load_list = []
	gen_list = []
	actions = []

	done = False
	terminated = False
	truncated = False

	for t in range(steps):
		action, _states = model.predict(obs, deterministic=True)
		actions.append(float(action[0]) if hasattr(action, '__iter__') else float(action))

		obs, reward, terminated, truncated, info = env.step(action)

		
		frequency = float(obs[0])
		generation = float(obs[1])
		load = float(obs[2])

		freq_list.append(frequency)
		load_list.append(load)
		gen_list.append(generation)

		if terminated or truncated:
			if terminated:
			
				remaining = steps - (t + 1)
				freq_list.extend([frequency] * remaining)
				load_list.extend([load] * remaining)
				gen_list.extend([generation] * remaining)
				actions.extend([0.0] * remaining)
			break

	fig, axes = plt.subplots(2, 2, figsize=(12, 8))

	steps_range = list(range(len(freq_list)))

	
	ax = axes[0, 0]
	ax.plot(steps_range, freq_list, label='Frequency (Hz)')
	ax.axhline(45.0, color='red', linestyle='--', label='45 Hz alert')
	ax.set_title('Frequency vs Steps')
	ax.set_xlabel('Step')
	ax.set_ylabel('Frequency (Hz)')
	ax.legend()


	ax = axes[0, 1]
	ax.plot(steps_range, load_list, label='Load')
	ax.plot(steps_range, gen_list, label='Generation')
	ax.set_title('Load vs Generation')
	ax.set_xlabel('Step')
	ax.set_ylabel('MW')
	ax.legend()

	ax = axes[1, 0]
	ax.step(list(range(len(actions))), actions, where='post')
	ax.set_title("Agent's Delta MW Actions")
	ax.set_xlabel('Step')
	ax.set_ylabel('Delta MW')

	ax = axes[1, 1]
	ax.axis('off')
	final_freq = freq_list[-1] if freq_list else None
	status = 'UNKNOWN'
	if final_freq is not None:
		if final_freq < 47.5:
			status = 'BLACKOUT'
		else:
			status = 'SUCCESS'

	status_text = f"Final Frequency: {final_freq:.3f} Hz\nStatus: {status}\nSteps simulated: {len(freq_list)}"
	ax.text(0.5, 0.5, status_text, ha='center', va='center', fontsize=14)

	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close(fig)

	print(f"Saved dashboard to: {output_path}")


if __name__ == '__main__':
	default_model_path = os.path.join('models', 'ppo_fast_20260209_174720', 'best_model')
	try:
		run_audit(default_model_path, steps=200, output_path='grid_performance_audit.png')
	except Exception as e:
		print(f"Error: {e}")

