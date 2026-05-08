import pandas as pd
import matplotlib.pyplot as plt
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

dqn = pd.read_csv("runs/models/dqn_metrics_by_seed.csv")["final_reward"]
ppo = pd.read_csv("runs/models/ppo_metrics_by_seed.csv")["final_reward"]

data = [dqn, ppo]

plt.figure(figsize=(8,6))
plt.boxplot(data, labels=["DQN", "PPO"], patch_artist=True)

plt.title("Final Reward per Seed Comparison")
plt.ylabel("Final Episode Reward")
plt.grid(True)

out_path = os.path.join(RESULT_DIR, "boxplot_final_reward.png")
plt.savefig(out_path, dpi=300)
print("Saved to:", out_path)

plt.show()