#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUNTIME_PATH = "logs/runtime_eval.csv"
OUT_DIR = "results/evaluation/policy_behavior"

ATTACK_ORDER = [
    "normal",
    "ddos_flood",
    "flow_overflow",
    "packet_in_flood",
    "ip_spoofing",
    "port_scanning",
]

CONFIG_ORDER = [
    "no_defense",
    "rule",
    "rl_dqn",
    "rl_ppo",
    "rl_guard_ppo",
    "rl_twin_ppo",
    "full_system_ppo",
]


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def clean_runtime(df):
    df = df.copy()

    for col in ["attack_type", "eval_config", "phase"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

    df = df[~df["eval_config"].isin(["collect", "collect_random"])]
    df = df[~df["phase"].isin(["idle", "unknown"])]
    df = df[df["attack_type"].isin(ATTACK_ORDER)]
    df = df[df["eval_config"].isin(CONFIG_ORDER)]

    return df


def plot_action_rate_heatmap(df, action_id):
    col = "action_final"

    tmp = df.copy()
    tmp[f"action_{action_id}_rate"] = (
        pd.to_numeric(tmp[col], errors="coerce") == action_id
    ).astype(float)

    pivot = tmp.pivot_table(
        index="attack_type",
        columns="eval_config",
        values=f"action_{action_id}_rate",
        aggfunc="mean",
    ).reindex(index=ATTACK_ORDER, columns=CONFIG_ORDER)

    plt.figure(figsize=(13, 6))
    plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label=f"Action {action_id} rate")
    plt.title(f"Action {action_id} rate by attack and config")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"heatmap_action_{action_id}_rate.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("[PLOT]", path)


def plot_dominant_action_heatmap(df):
    rows = []

    for (attack, config), g in df.groupby(["attack_type", "eval_config"]):
        actions = pd.to_numeric(g["action_final"], errors="coerce").dropna()
        if actions.empty:
            continue

        dominant = int(actions.value_counts().idxmax())

        rows.append({
            "attack_type": attack,
            "eval_config": config,
            "dominant_action": dominant,
        })

    table = pd.DataFrame(rows)

    pivot = table.pivot(
        index="attack_type",
        columns="eval_config",
        values="dominant_action",
    ).reindex(index=ATTACK_ORDER, columns=CONFIG_ORDER)

    plt.figure(figsize=(13, 6))
    plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=4)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Dominant final action")
    plt.title("Dominant final action by attack and config")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                plt.text(j, i, str(int(val)), ha="center", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "heatmap_dominant_final_action.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("[PLOT]", path)


def plot_phase_latency_profile(df):
    metric = "latency"

    pivot = df.pivot_table(
        index="eval_config",
        columns="phase",
        values=metric,
        aggfunc="mean",
    ).reindex(index=CONFIG_ORDER)

    phases = [p for p in ["warmup", "attack", "recovery"] if p in pivot.columns]
    pivot = pivot[phases]

    x = np.arange(len(pivot.index))
    width = 0.25

    plt.figure(figsize=(13, 6))

    for k, phase in enumerate(phases):
        plt.bar(x + k * width, pivot[phase].values, width=width, label=phase)

    plt.xticks(x + width, pivot.index, rotation=30, ha="right")
    plt.ylabel("Mean latency")
    plt.title("Latency profile by phase and config")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "bar_phase_latency_by_config.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("[PLOT]", path)


def plot_phase_reward_profile(df):
    metric = "reward"

    pivot = df.pivot_table(
        index="eval_config",
        columns="phase",
        values=metric,
        aggfunc="mean",
    ).reindex(index=CONFIG_ORDER)

    phases = [p for p in ["warmup", "attack", "recovery"] if p in pivot.columns]
    pivot = pivot[phases]

    x = np.arange(len(pivot.index))
    width = 0.25

    plt.figure(figsize=(13, 6))

    for k, phase in enumerate(phases):
        plt.bar(x + k * width, pivot[phase].values, width=width, label=phase)

    plt.xticks(x + width, pivot.index, rotation=30, ha="right")
    plt.ylabel("Mean reward")
    plt.title("Reward profile by phase and config")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "bar_phase_reward_by_config.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("[PLOT]", path)


def main():
    ensure_out()

    df = pd.read_csv(RUNTIME_PATH)
    df = clean_runtime(df)

    for action_id in range(5):
        plot_action_rate_heatmap(df, action_id)

    plot_dominant_action_heatmap(df)
    plot_phase_latency_profile(df)
    plot_phase_reward_profile(df)

    print("[OK] policy behavior plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()