#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SUMMARY_PATH = "results/evaluation/summary_full_benchmark.csv"
OUT_DIR = "results/evaluation/report_ready"

ATTACK_ORDER = [
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


def clean_summary(df):
    df = df.copy()

    for col in ["attack_type", "eval_config", "phase"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

    df = df[~df["eval_config"].isin(["collect", "collect_random"])]

    if "attack_type" in df.columns:
        df = df[df["attack_type"] != "unknown"]

    return df


def minmax_score(series, higher_is_better=True):
    s = pd.to_numeric(series, errors="coerce")
    mn = s.min()
    mx = s.max()

    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.ones(len(s)) * 0.5, index=s.index)

    norm = (s - mn) / (mx - mn)

    if higher_is_better:
        return norm

    return 1 - norm


def build_attack_only(df):
    attack_df = df[df["attack_type"].isin(ATTACK_ORDER)].copy()
    attack_df = attack_df[attack_df["eval_config"].isin(CONFIG_ORDER)].copy()
    return attack_df


def add_normalized_scores(df):
    df = df.copy()

    score_parts = []

    for attack, g in df.groupby("attack_type"):
        idx = g.index

        df.loc[idx, "score_reward"] = minmax_score(
            g["mean_reward"],
            higher_is_better=True,
        )

        df.loc[idx, "score_latency"] = minmax_score(
            g["mean_latency"],
            higher_is_better=False,
        )

        df.loc[idx, "score_loss"] = minmax_score(
            g["mean_packet_loss"],
            higher_is_better=False,
        )

        if "recovery_time_steps" in g.columns:
            df.loc[idx, "score_recovery"] = minmax_score(
                g["recovery_time_steps"],
                higher_is_better=False,
            )
        else:
            df.loc[idx, "score_recovery"] = 0.5

        if "switching_rate" in g.columns:
            df.loc[idx, "score_stability"] = minmax_score(
                g["switching_rate"],
                higher_is_better=False,
            )
        else:
            df.loc[idx, "score_stability"] = 0.5

    df["report_composite_score"] = (
        0.35 * df["score_reward"]
        + 0.20 * df["score_latency"]
        + 0.20 * df["score_loss"]
        + 0.15 * df["score_recovery"]
        + 0.10 * df["score_stability"]
    )

    return df


def plot_bar_metric_by_config(df, metric, title, ylabel, filename):
    data = (
        df.groupby("eval_config")[metric]
        .agg(["mean", "std"])
        .reindex(CONFIG_ORDER)
        .dropna(how="all")
    )

    x = np.arange(len(data))
    y = data["mean"].values
    err = data["std"].fillna(0).values

    plt.figure(figsize=(12, 5))
    plt.bar(x, y, yerr=err, capsize=4)
    plt.xticks(x, data.index, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print("[PLOT]", path)


def plot_heatmap(df, metric, title, filename, fmt=".2f"):
    pivot = df.pivot_table(
        index="attack_type",
        columns="eval_config",
        values=metric,
        aggfunc="mean",
    ).reindex(index=ATTACK_ORDER, columns=CONFIG_ORDER)

    plt.figure(figsize=(13, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label=metric)
    plt.title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                plt.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print("[PLOT]", path)


def add_vs_no_defense(df):
    df = df.copy()

    rows = []

    for attack, g in df.groupby("attack_type"):
        base = g[g["eval_config"] == "no_defense"]

        if base.empty:
            continue

        base = base.iloc[0]

        for _, row in g.iterrows():
            r = row.copy()

            base_latency = float(base.get("mean_latency", np.nan))
            base_loss = float(base.get("mean_packet_loss", np.nan))
            base_reward = float(base.get("mean_reward", np.nan))
            base_recovery = float(base.get("recovery_time_steps", np.nan))

            latency = float(row.get("mean_latency", np.nan))
            loss = float(row.get("mean_packet_loss", np.nan))
            reward = float(row.get("mean_reward", np.nan))
            recovery = float(row.get("recovery_time_steps", np.nan))

            r["latency_reduction_vs_no_defense_pct"] = (
                (base_latency - latency) / base_latency * 100
                if base_latency and not np.isnan(base_latency)
                else np.nan
            )

            r["packet_loss_reduction_vs_no_defense_pct"] = (
                (base_loss - loss) / base_loss * 100
                if base_loss and not np.isnan(base_loss)
                else np.nan
            )

            r["reward_gain_vs_no_defense"] = (
                reward - base_reward
                if not np.isnan(base_reward)
                else np.nan
            )

            r["recovery_reduction_vs_no_defense_pct"] = (
                (base_recovery - recovery) / base_recovery * 100
                if base_recovery and not np.isnan(base_recovery)
                else np.nan
            )

            rows.append(r)

    return pd.DataFrame(rows)


def plot_line_by_attack(df, metric, title, ylabel, filename):
    pivot = df.pivot_table(
        index="attack_type",
        columns="eval_config",
        values=metric,
        aggfunc="mean",
    ).reindex(index=ATTACK_ORDER, columns=CONFIG_ORDER)

    x = np.arange(len(pivot.index))

    plt.figure(figsize=(13, 6))

    for config in pivot.columns:
        y = pd.to_numeric(pivot[config], errors="coerce")
        if y.notna().sum() == 0:
            continue
        plt.plot(x, y.values, marker="o", label=config)

    plt.xticks(x, pivot.index, rotation=20, ha="right")
    plt.xlabel("Attack type")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print("[PLOT]", path)


def plot_report_dashboard(df):
    metrics = [
        ("mean_reward", "Mean reward"),
        ("defense_score", "Defense score"),
        ("mean_latency", "Mean latency"),
        ("mean_packet_loss", "Mean packet loss"),
        ("recovery_time_steps", "Recovery time steps"),
        ("report_composite_score", "Composite score"),
    ]

    available = [(m, t) for m, t in metrics if m in df.columns]

    rows = 2
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(18, 9))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, available):
        data = (
            df.groupby("eval_config")[metric]
            .mean()
            .reindex(CONFIG_ORDER)
        )

        x = np.arange(len(data))
        ax.bar(x, data.values)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(data.index, rotation=35, ha="right")

    for ax in axes[len(available):]:
        ax.axis("off")

    fig.suptitle("Report-ready multi-metric comparison by config", fontsize=16)
    fig.tight_layout()

    path = os.path.join(OUT_DIR, "report_dashboard_multi_metric_by_config.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("[PLOT]", path)


def main():
    ensure_out()

    df = pd.read_csv(SUMMARY_PATH)
    df = clean_summary(df)
    attack_df = build_attack_only(df)
    attack_df = add_normalized_scores(attack_df)
    attack_df = add_vs_no_defense(attack_df)

    attack_df.to_csv(
        os.path.join(OUT_DIR, "report_ready_attack_summary.csv"),
        index=False,
    )

    plot_report_dashboard(attack_df)

    plot_bar_metric_by_config(
        attack_df,
        "report_composite_score",
        "Composite defense score by config",
        "Composite score",
        "bar_report_composite_score_by_config.png",
    )

    plot_bar_metric_by_config(
        attack_df,
        "mean_reward",
        "Mean reward by config",
        "Mean reward",
        "bar_report_mean_reward_by_config.png",
    )

    plot_bar_metric_by_config(
        attack_df,
        "defense_score",
        "Defense score by config",
        "Defense score",
        "bar_report_defense_score_by_config.png",
    )

    plot_heatmap(
        attack_df,
        "report_composite_score",
        "Attack × config: composite score",
        "heatmap_report_composite_score.png",
    )

    plot_heatmap(
        attack_df,
        "reward_gain_vs_no_defense",
        "Attack × config: reward gain vs no_defense",
        "heatmap_reward_gain_vs_no_defense.png",
    )

    plot_heatmap(
        attack_df,
        "latency_reduction_vs_no_defense_pct",
        "Attack × config: latency reduction vs no_defense (%)",
        "heatmap_latency_reduction_vs_no_defense.png",
    )

    plot_heatmap(
        attack_df,
        "packet_loss_reduction_vs_no_defense_pct",
        "Attack × config: packet loss reduction vs no_defense (%)",
        "heatmap_loss_reduction_vs_no_defense.png",
    )

    plot_line_by_attack(
        attack_df,
        "report_composite_score",
        "Composite score across attack types",
        "Composite score",
        "line_report_composite_score_by_config.png",
    )

    plot_line_by_attack(
        attack_df,
        "reward_gain_vs_no_defense",
        "Reward gain vs no_defense across attack types",
        "Reward gain",
        "line_reward_gain_vs_no_defense.png",
    )

    print("[OK] report-ready plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()