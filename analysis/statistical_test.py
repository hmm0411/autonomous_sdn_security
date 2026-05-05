import pandas as pd
from scipy.stats import ttest_ind

dqn = pd.read_csv("runs/models/dqn_metrics_by_seed.csv")["final_reward"]
ppo = pd.read_csv("runs/models/ppo_metrics_by_seed.csv")["final_reward"]

t, p = ttest_ind(dqn, ppo)

print("T-stat:", t)
print("P-value:", p)

if p < 0.05:
    print("Significant difference ✔")
else:
    print("No significant difference")