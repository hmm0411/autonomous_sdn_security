import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(ROOT_DIR, "results", "evaluation_summary.csv")
FIG_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
df = pd.read_csv(RESULTS_PATH)

# Pivot thành matrix
pivot = df.pivot(index="scenario", columns="model", values="mean_reward")

# Sort scenario theo thứ tự logic
scenario_order = ["normal", "ddos", "spoofing", "overflow", "packet_in", "port_scan"]
pivot = pivot.reindex(scenario_order)

plt.figure(figsize=(10,6))

sns.heatmap(
    pivot,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    linewidths=0.5,
    linecolor="black"
)

plt.title("Robustness Heatmap (Mean Reward per Scenario)")
plt.ylabel("Scenario")
plt.xlabel("Model")

out_path = os.path.join(FIG_DIR, "robustness_heatmap.png")
plt.savefig(out_path, dpi=300)
print("Saved to:", out_path)

plt.show()