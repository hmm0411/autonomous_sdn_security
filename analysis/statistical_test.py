import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv("results/evaluation_summary.csv")

dqn = df[df.model == "DQN"]["mean_reward"]
ppo = df[df.model == "PPO"]["mean_reward"]

t, p = ttest_ind(dqn, ppo)

print("T-stat:", t)
print("P-value:", p)

if p < 0.05:
    print("Significant difference ✔")
else:
    print("No significant difference")