# evaluation.py
# Chạy từ thư mục gốc:
# python evaluation.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")


def safe_read(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"⚠ Không tìm thấy file: {path}")
        return None


def plot_learning_curves():
    dqn = safe_read(os.path.join(RESULTS_DIR, "dqn_summary_results.csv"))
    ppo = safe_read(os.path.join(RESULTS_DIR, "ppo_summary_results.csv"))

    if dqn is None and ppo is None:
        return

    plt.figure(figsize=(10, 6))

    if dqn is not None:
        plt.plot(dqn["episode"], dqn["mean_reward"], label="DQN")
        plt.fill_between(
            dqn["episode"],
            dqn["mean_reward"] - dqn["std_reward"],
            dqn["mean_reward"] + dqn["std_reward"],
            alpha=0.2,
        )

    if ppo is not None:
        plt.plot(ppo["episode"], ppo["mean_reward"], label="PPO")
        plt.fill_between(
            ppo["episode"],
            ppo["mean_reward"] - ppo["std_reward"],
            ppo["mean_reward"] + ppo["std_reward"],
            alpha=0.2,
        )

    plt.title("Mean Reward ± Std")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "learning_curve.png"), dpi=300)
    plt.close()


def plot_loss_curve():
    dqn = safe_read(os.path.join(RESULTS_DIR, "dqn_summary_results.csv"))
    ppo = safe_read(os.path.join(RESULTS_DIR, "ppo_summary_results.csv"))

    if dqn is None and ppo is None:
        return

    plt.figure(figsize=(10, 6))

    if dqn is not None and "mean_loss" in dqn:
        plt.plot(dqn["episode"], dqn["mean_loss"], label="DQN")

    if ppo is not None and "mean_loss" in ppo:
        plt.plot(ppo["episode"], ppo["mean_loss"], label="PPO")

    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "loss_curve.png"), dpi=300)
    plt.close()


def plot_action_distribution():
    for model in ["dqn", "ppo"]:
        df = safe_read(os.path.join(RESULTS_DIR, f"{model}_seed_42_results.csv"))
        if df is None or "action" not in df:
            continue

        plt.figure(figsize=(6, 4))
        df["action"].value_counts().sort_index().plot(kind="bar")
        plt.title(f"Action Distribution ({model.upper()})")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{model}_action_distribution.png"), dpi=300)
        plt.close()


def plot_reward_distribution():
    df = safe_read(os.path.join(RESULTS_DIR, "dqn_seed_42_results.csv"))
    if df is None or "reward" not in df:
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="reward", kde=True)
    plt.title("Reward Distribution (DQN)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "reward_distribution.png"), dpi=300)
    plt.close()


def plot_switching_rate():
    dqn = safe_read(os.path.join(RESULTS_DIR, "dqn_summary_results.csv"))
    if dqn is None or "switching_rate" not in dqn:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(dqn["episode"], dqn["switching_rate"])
    plt.title("Switching Rate Over Training")
    plt.xlabel("Episode")
    plt.ylabel("Switching Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "switching_rate.png"), dpi=300)
    plt.close()


def plot_attack_reward_relation():
    df = safe_read(os.path.join(RESULTS_DIR, "dqn_seed_42_results.csv"))
    if df is None or "attack_indicator" not in df:
        return

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="attack_indicator", y="reward")
    plt.title("Reward vs Attack Indicator")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "reward_vs_attack.png"), dpi=300)
    plt.close()


def main():
    print("Generating publication figures...")
    plot_learning_curves()
    plot_loss_curve()
    plot_action_distribution()
    plot_reward_distribution()
    plot_switching_rate()
    plot_attack_reward_relation()
    print("All figures saved in 'figures/' folder.")


if __name__ == "__main__":
    main()