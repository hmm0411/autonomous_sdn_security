from tensorboard.backend.event_processing import event_accumulator
import glob
import matplotlib.pyplot as plt
import numpy as np

paths = glob.glob("runs/dqn_seed_*/events*")

all_rewards = []

for p in paths:
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
plt.savefig("results/learning_curve_dqn.png", dpi=300)