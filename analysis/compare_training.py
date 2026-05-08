# compare_training.py
# Chạy từ thư mục gốc:
# python3 analysis/compare_training.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def has_plot(ax):
    return len(ax.lines) > 0 or len(ax.patches) > 0


def first_positive_episode(df):
    positive_df = df[df["mean_reward"] > 0]

    if positive_df.empty:
        return None

    return int(positive_df["episode"].iloc[0])


def plot_learning_curves():

    SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    RESULTS_DIR = os.path.join(ROOT_DIR, "runs", "models")

    os.makedirs(os.path.join(RESULTS_DIR, "results"), exist_ok=True)

    dqn_csv = os.path.join(RESULTS_DIR, "dqn_summary_results.csv")
    ppo_csv = os.path.join(RESULTS_DIR, "ppo_summary_results.csv")

    print("DQN CSV:", dqn_csv, os.path.exists(dqn_csv))
    print("PPO CSV:", ppo_csv, os.path.exists(ppo_csv))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df_dqn = None
    df_ppo = None

    # =============================
    # 1. Mean Reward Curve
    # =============================
    if os.path.exists(dqn_csv):
        df_dqn = pd.read_csv(dqn_csv)

        axes[0, 0].plot(
            df_dqn["episode"],
            df_dqn["mean_reward"],
            label="DQN"
        )

        axes[0, 0].fill_between(
            df_dqn["episode"],
            df_dqn["mean_reward"] - df_dqn["std_reward"],
            df_dqn["mean_reward"] + df_dqn["std_reward"],
            alpha=0.2
        )

    if os.path.exists(ppo_csv):
        df_ppo = pd.read_csv(ppo_csv)

        axes[0, 0].plot(
            df_ppo["episode"],
            df_ppo["mean_reward"],
            label="PPO"
        )

        axes[0, 0].fill_between(
            df_ppo["episode"],
            df_ppo["mean_reward"] - df_ppo["std_reward"],
            df_ppo["mean_reward"] + df_ppo["std_reward"],
            alpha=0.2
        )

    axes[0, 0].set_title("Mean Reward (Multi-seed)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Mean Reward")

    if has_plot(axes[0, 0]):
        axes[0, 0].legend()

    # =============================
    # 2. Loss Curve
    # =============================
    if df_dqn is not None and "mean_loss" in df_dqn.columns:
        axes[0, 1].plot(
            df_dqn["episode"],
            df_dqn["mean_loss"],
            label="DQN Loss"
        )

    if df_ppo is not None and "mean_loss" in df_ppo.columns:
        axes[0, 1].plot(
            df_ppo["episode"],
            df_ppo["mean_loss"],
            label="PPO Loss"
        )

    axes[0, 1].set_title("Training Loss")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Mean Loss")

    if has_plot(axes[0, 1]):
        axes[0, 1].legend()
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No loss data",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes
        )

    # =============================
    # 3. Variance Comparison
    # =============================
    if df_dqn is not None and "std_reward" in df_dqn.columns:
        axes[1, 0].plot(
            df_dqn["episode"],
            df_dqn["std_reward"],
            label="DQN Reward Std"
        )

    if df_ppo is not None and "std_reward" in df_ppo.columns:
        axes[1, 0].plot(
            df_ppo["episode"],
            df_ppo["std_reward"],
            label="PPO Reward Std"
        )

    axes[1, 0].set_title("Reward Std (Stability)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Std Reward")

    if has_plot(axes[1, 0]):
        axes[1, 0].legend()

    # =============================
    # 4. Convergence Speed
    # =============================
    labels = []
    values = []

    if df_dqn is not None:
        dqn_conv = first_positive_episode(df_dqn)

        if dqn_conv is not None:
            labels.append("DQN")
            values.append(dqn_conv)

    if df_ppo is not None:
        ppo_conv = first_positive_episode(df_ppo)

        if ppo_conv is not None:
            labels.append("PPO")
            values.append(ppo_conv)

    if len(values) > 0:
        axes[1, 1].bar(labels, values)
        axes[1, 1].set_ylabel("Episode")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No model reached positive reward",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes
        )

    axes[1, 1].set_title("Episodes to Positive Reward")

    plt.tight_layout()

    out_path = os.path.join(
        RESULTS_DIR,
        "results",
        "advanced_training_analysis.png"
    )

    plt.savefig(out_path, dpi=300)
    print(f"Saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    plot_learning_curves()