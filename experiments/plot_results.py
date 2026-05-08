import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

RESULT_PATH = os.path.join(ROOT_DIR, "results", "evaluation_summary.csv")
FIG_DIR = os.path.join(ROOT_DIR, "results", "figures")


def load_data():
    print("Looking for:", RESULT_PATH)

    if not os.path.exists(RESULT_PATH):
        raise FileNotFoundError(
            "Chưa có evaluation_summary.csv.\n"
            "Chạy: python3 -m experiments.evaluate trước."
        )

    return pd.read_csv(RESULT_PATH)


def ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_reward_comparison(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="scenario", y="mean_reward", hue="model")
    plt.title("Mean Reward by Scenario")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/reward_by_scenario.png", dpi=300)
    plt.close()


def plot_switching_comparison(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="scenario", y="mean_switching", hue="model")
    plt.title("Switching Rate by Scenario")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/switching_by_scenario.png", dpi=300)
    plt.close()


def plot_std_reward(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="scenario", y="std_reward", hue="model")
    plt.title("Reward Variability (Std Dev)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/reward_std.png", dpi=300)
    plt.close()


def plot_overall_model_ranking(df):
    plt.figure(figsize=(8, 6))
    summary = df.groupby("model")["mean_reward"].mean().reset_index()
    sns.barplot(data=summary, x="model", y="mean_reward")
    plt.title("Overall Average Reward Ranking")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/overall_ranking.png", dpi=300)
    plt.close()


def plot_heatmap(df):
    pivot = df.pivot(index="scenario", columns="model", values="mean_reward")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap="viridis")
    plt.title("Reward Heatmap")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/reward_heatmap.png", dpi=300)
    plt.close()


def plot_box_style_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="model", y="mean_reward")
    plt.title("Reward Distribution Across Scenarios")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/reward_distribution.png", dpi=300)
    plt.close()


def plot_robustness_score(df):
    robustness = df.groupby("model")["mean_reward"].std().reset_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(data=robustness, x="model", y="mean_reward")
    plt.title("Robustness (Lower Std = More Stable)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/robustness.png", dpi=300)
    plt.close()


def generate_all_plots():
    ensure_dir()
    df = load_data()

    plot_reward_comparison(df)
    plot_switching_comparison(df)
    plot_std_reward(df)
    plot_overall_model_ranking(df)
    plot_heatmap(df)
    plot_box_style_distribution(df)
    plot_robustness_score(df)

    print(f"\nAll figures saved in: {FIG_DIR}")


if __name__ == "__main__":
    generate_all_plots()