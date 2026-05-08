import pandas as pd
import matplotlib.pyplot as plt
import os

# ===== FILE PATH =====
DQN_FILE = "results/dqn_online_test.csv"
PPO_FILE = "results/ppo_online_test.csv"

if not os.path.exists(DQN_FILE) or not os.path.exists(PPO_FILE):
    print("Timeline files not found.")
    exit()

df_dqn = pd.read_csv(DQN_FILE)
df_ppo = pd.read_csv(PPO_FILE)

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# =========================
# Flow Count
# =========================
axes[0].plot(df_dqn["step"], df_dqn["flow_count"], label="DQN", color="blue")
axes[0].plot(df_ppo["step"], df_ppo["flow_count"], label="PPO", color="green")
axes[0].set_ylabel("Flow Count")
axes[0].legend()

# =========================
# Entropy
# =========================
axes[1].plot(df_dqn["step"], df_dqn["entropy"], color="blue")
axes[1].plot(df_ppo["step"], df_ppo["entropy"], color="green")
axes[1].set_ylabel("Entropy")

# =========================
# Action Comparison
# =========================
axes[2].step(df_dqn["step"], df_dqn["action"], where="post", color="blue", label="DQN")
axes[2].step(df_ppo["step"], df_ppo["action"], where="post", color="green", label="PPO")
axes[2].set_ylabel("Action")
axes[2].set_yticks([0,1,2,3,4])
axes[2].legend()

# =========================
# Reward
# =========================
axes[3].plot(df_dqn["step"], df_dqn["reward"], color="blue")
axes[3].plot(df_ppo["step"], df_ppo["reward"], color="green")
axes[3].set_ylabel("Reward")
axes[3].set_xlabel("Time Step")

plt.tight_layout()
plt.savefig("results/online_timeline_comparison.png", dpi=300)
plt.show()