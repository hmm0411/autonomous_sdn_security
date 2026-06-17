import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUNTIME_LOG = os.getenv("RUNTIME_LOG", "logs/runtime_eval.csv")
SUMMARY_PATH = os.getenv(
    "SUMMARY_PATH",
    "results/evaluation/summary_full_benchmark.csv",
)
OUT_DIR = os.getenv("EVAL_OUT_DIR", "results/evaluation")


ORDER = [
    "no_defense",
    "rule",
    "rl_dqn",
    "rl_ppo",
    "rl_guard_dqn",
    "rl_guard_ppo",
    "rl_twin_dqn",
    "rl_twin_ppo",
    "full_system_dqn",
    "full_system_ppo",
]


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def ordered_configs(df):
    configs = list(df["eval_config"].dropna().unique())
    ordered = [c for c in ORDER if c in configs]
    ordered += [c for c in configs if c not in ordered]
    return ordered


def plot_grouped_bar(summary, metric, title, ylabel, filename):
    configs = ordered_configs(summary)

    data = (
        summary.groupby("eval_config")[metric]
        .agg(["mean", "std"])
        .reindex(configs)
        .reset_index()
    )

    x = np.arange(len(data))
    y = data["mean"].fillna(0).values
    err = data["std"].fillna(0).values

    plt.figure(figsize=(12, 5))
    plt.bar(x, y, yerr=err, capsize=4)
    plt.xticks(x, data["eval_config"], rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print("[PLOT]", path)


def plot_heatmap(summary, metric, title, filename):
    pivot = summary.pivot_table(
        index="attack_type",
        columns="eval_config",
        values=metric,
        aggfunc="mean",
    )

    configs = [c for c in ORDER if c in pivot.columns]
    configs += [c for c in pivot.columns if c not in configs]
    pivot = pivot[configs]

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
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print("[PLOT]", path)


def plot_action_stacked(runtime):
    if "action_final" in runtime.columns:
        action_col = "action_final"
    elif "action" in runtime.columns:
        action_col = "action"
    else:
        return

    configs = ordered_configs(runtime)

    table = (
        runtime.groupby(["eval_config", action_col])
        .size()
        .unstack(fill_value=0)
        .reindex(configs)
        .fillna(0)
    )

    for a in range(5):
        if a not in table.columns:
            table[a] = 0

    table = table[[0, 1, 2, 3, 4]]
    table_pct = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    bottom = np.zeros(len(table_pct))
    x = np.arange(len(table_pct))

    plt.figure(figsize=(12, 5))

    for a in [0, 1, 2, 3, 4]:
        vals = table_pct[a].values
        plt.bar(x, vals, bottom=bottom, label=f"action {a}")
        bottom += vals

    plt.xticks(x, table_pct.index, rotation=30, ha="right")
    plt.ylabel("Action ratio")
    plt.title("Action distribution by evaluation config")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "action_distribution_stacked.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("[PLOT]", path)


def plot_twin_effect(summary):
    twin_rows = summary[
        summary["eval_config"].astype(str).str.contains("twin|full", regex=True)
    ].copy()

    if twin_rows.empty:
        return

    plot_grouped_bar(
        twin_rows,
        "twin_reject_rate",
        "Digital Twin rejected action rate",
        "Reject rate",
        "twin_reject_rate_by_config.png",
    )

    plot_grouped_bar(
        twin_rows,
        "mean_gap_latency",
        "Digital Twin latency sim-to-real gap",
        "Mean latency gap",
        "twin_gap_latency_by_config.png",
    )

    plot_grouped_bar(
        twin_rows,
        "mean_gap_loss",
        "Digital Twin packet loss sim-to-real gap",
        "Mean loss gap",
        "twin_gap_loss_by_config.png",
    )


def plot_attack_timeline(runtime, attack="ddos_flood", run_id=None):
    df = runtime[runtime["attack_type"] == attack].copy()
    if df.empty:
        return

    if run_id is not None and "run_id" in df.columns:
        df = df[df["run_id"].astype(str) == str(run_id)]

    if df.empty:
        return

    metrics = ["latency", "packet_loss", "controller_cpu", "reward"]

    for metric in metrics:
        if metric not in df.columns:
            continue

        plt.figure(figsize=(12, 5))

        for config, group in df.groupby("eval_config"):
            group = group.reset_index(drop=True)
            values = pd.to_numeric(group[metric], errors="coerce")
            plt.plot(range(len(values)), values, label=config)

        plt.title(f"{metric} timeline under {attack}")
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()

        path = os.path.join(OUT_DIR, f"timeline_{attack}_{metric}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print("[PLOT]", path)

    # Action dùng step plot
    action_col = "action_final" if "action_final" in df.columns else "action"
    if action_col in df.columns:
        plt.figure(figsize=(12, 5))

        for config, group in df.groupby("eval_config"):
            group = group.reset_index(drop=True)
            values = pd.to_numeric(group[action_col], errors="coerce")
            plt.step(range(len(values)), values, where="post", label=config)

        plt.title(f"Action timeline under {attack}")
        plt.xlabel("Step")
        plt.ylabel("Action")
        plt.yticks([0, 1, 2, 3, 4])
        plt.legend()
        plt.tight_layout()

        path = os.path.join(OUT_DIR, f"timeline_{attack}_action.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print("[PLOT]", path)


def main():
    ensure_out()

    summary = pd.read_csv(SUMMARY_PATH)
    runtime = pd.read_csv(RUNTIME_LOG)

    plot_grouped_bar(
        summary,
        "mean_latency",
        "Average latency by evaluation config",
        "Mean latency",
        "bar_mean_latency_by_config.png",
    )

    plot_grouped_bar(
        summary,
        "mean_packet_loss",
        "Average packet loss by evaluation config",
        "Mean packet loss",
        "bar_mean_packet_loss_by_config.png",
    )

    plot_grouped_bar(
        summary,
        "recovery_time_steps",
        "Recovery time by evaluation config",
        "Recovery time steps",
        "bar_recovery_time_by_config.png",
    )

    plot_grouped_bar(
        summary,
        "cumulative_reward",
        "Cumulative reward by evaluation config",
        "Cumulative reward",
        "bar_cumulative_reward_by_config.png",
    )

    plot_grouped_bar(
        summary,
        "false_positive_rate",
        "False positive rate by evaluation config",
        "False positive rate",
        "bar_false_positive_by_config.png",
    )

    plot_heatmap(
        summary,
        "mean_reward",
        "Attack × config heatmap: mean reward",
        "heatmap_attack_config_reward.png",
    )

    plot_heatmap(
        summary,
        "mean_latency",
        "Attack × config heatmap: mean latency",
        "heatmap_attack_config_latency.png",
    )

    plot_action_stacked(runtime)
    plot_twin_effect(summary)

    plot_attack_timeline(runtime, attack="ddos_flood")
    plot_attack_timeline(runtime, attack="flow_overflow")


if __name__ == "__main__":
    main()