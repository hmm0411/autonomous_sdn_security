# compare_training.py
# Chạy từ thư mục gốc:
# python compare_training.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_learning_curves():

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    dqn_csv = os.path.join(RESULTS_DIR, "dqn_summary_results.csv")
    ppo_csv = os.path.join(RESULTS_DIR, "ppo_summary_results.csv")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df_dqn = None
    df_ppo = None

    # =============================
    # 1. Mean Reward Curve
    # =============================
    if os.path.exists(dqn_csv):
        df_dqn = pd.read_csv(dqn_csv)
        axes[0, 0].plot(df_dqn['episode'], df_dqn['mean_reward'],
                        label='DQN', color='red')
        axes[0, 0].fill_between(
            df_dqn['episode'],
            df_dqn['mean_reward'] - df_dqn['std_reward'],
            df_dqn['mean_reward'] + df_dqn['std_reward'],
            alpha=0.2, color='red'
        )

    if os.path.exists(ppo_csv):
        df_ppo = pd.read_csv(ppo_csv)
        axes[0, 0].plot(df_ppo['episode'], df_ppo['mean_reward'],
                        label='PPO', color='blue')
        axes[0, 0].fill_between(
            df_ppo['episode'],
            df_ppo['mean_reward'] - df_ppo['std_reward'],
            df_ppo['mean_reward'] + df_ppo['std_reward'],
            alpha=0.2, color='blue'
        )

    axes[0, 0].set_title("Mean Reward (Multi-seed)")
    axes[0, 0].legend()

    # =============================
    # 2.  Loss Curve
    # =============================
    if df_dqn is not None:
        axes[0, 1].plot(df_dqn['episode'], df_dqn['mean_loss'],
                        label='DQN Loss', color='red')

    if df_ppo is not None:
        axes[0, 1].plot(df_ppo['episode'], df_ppo['mean_loss'],
                        label='PPO Loss', color='blue')

    axes[0, 1].set_title("Training Loss")
    axes[0, 1].legend()

    # =============================
    # 3. Variance Comparison
    # =============================
    if df_dqn is not None:
        axes[1, 0].plot(df_dqn['episode'], df_dqn['std_reward'],
                        label='DQN Variance', color='red')

    if df_ppo is not None:
        axes[1, 0].plot(df_ppo['episode'], df_ppo['std_reward'],
                        label='PPO Variance', color='blue')

    axes[1, 0].set_title("Reward Std (Stability)")
    axes[1, 0].legend()

    # =============================
    # 4. Convergence Speed
    # =============================
    if df_dqn is not None and df_ppo is not None:

        dqn_conv = np.argmax(df_dqn['mean_reward'] > 0)
        ppo_conv = np.argmax(df_ppo['mean_reward'] > 0)

        axes[1, 1].bar(['DQN', 'PPO'],
                       [dqn_conv, ppo_conv],
                       color=['red', 'blue'])

        axes[1, 1].set_title("Episodes to Positive Reward")

    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "advanced_training_analysis.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    plot_learning_curves()