from tensorboard.backend.event_processing import event_accumulator
import glob
import matplotlib.pyplot as plt
import numpy as np

path_dqn = glob.glob("runs/dqn_seed_*/events*")
path_ppo = glob.glob("runs/ppo_seed_*/events*")

all_rewards = []

for p in path_dqn:
    ea = event_accumulator.EventAccumulator(p)
    ea.Reload()

    rewards = [e.value for e in ea.Scalars("Reward")]
    all_rewards.append(rewards)

min_len = min(map(len, all_rewards))
all_rewards = np.array([r[:min_len] for r in all_rewards])

mean = all_rewards.mean(axis=0)
std = all_rewards.std(axis=0)

plt.plot(mean)
plt.fill_between(range(min_len), mean-std, mean+std, alpha=0.2)
plt.title("Multi-seed Learning Curve (DQN)")
plt.savefig("results/analysis/learning_curve_dqn.png", dpi=300)

all_rewards_ppo = []

for p in path_ppo:
    ea = event_accumulator.EventAccumulator(p)
    ea.Reload()

    rewards = [e.value for e in ea.Scalars("Reward")]
    all_rewards_ppo.append(rewards)

min_len_ppo = min(map(len, all_rewards_ppo))
all_rewards_ppo = np.array([r[:min_len_ppo] for r in all_rewards_ppo])

mean_ppo = all_rewards_ppo.mean(axis=0)
std_ppo = all_rewards_ppo.std(axis=0)

plt.plot(mean_ppo)
plt.fill_between(range(min_len_ppo), mean_ppo-std_ppo, mean_ppo+std_ppo, alpha=0.2)
plt.title("Multi-seed Learning Curve (PPO)")
plt.savefig("results/analysis/learning_curve_ppo.png", dpi=300)