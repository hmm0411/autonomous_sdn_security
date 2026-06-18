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


def clean_data(summary: pd.DataFrame, runtime: pd.DataFrame):
    if "eval_config" in summary.columns:
        summary = summary[
            ~summary["eval_config"].astype(str).isin(["collect", "collect_random"])
        ].copy()

    if "eval_config" in runtime.columns:
        runtime = runtime[
            ~runtime["eval_config"].astype(str).isin(["collect", "collect_random"])
        ].copy()

    if "phase" in runtime.columns:
        runtime = runtime[runtime["phase"].astype(str) != "idle"].copy()

    if "attack_type" in summary.columns:
        summary = summary[summary["attack_type"].astype(str) != "unknown"].copy()

    if "attack_type" in runtime.columns:
        runtime = runtime[runtime["attack_type"].astype(str) != "unknown"].copy()

    return summary, runtime


def ordered_configs(df: pd.DataFrame):
    if "eval_config" not in df.columns:
        return []

    configs = list(df["eval_config"].dropna().astype(str).unique())
    ordered = [c for c in CONFIG_ORDER if c in configs]
    ordered += [c for c in configs if c not in ordered]
    return ordered


def ordered_attacks(df: pd.DataFrame):
    if "attack_type" not in df.columns:
        return []

    attacks = list(df["attack_type"].dropna().astype(str).unique())
    ordered = [a for a in ATTACK_ORDER if a in attacks]
    ordered += [a for a in attacks if a not in ordered]
    return ordered


def plot_bar_by_config(summary, metric, title, ylabel, filename):
    if metric not in summary.columns:
        print(f"[SKIP] missing metric: {metric}")
        return

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
    if metric not in summary.columns:
        print(f"[SKIP] missing metric: {metric}")
        return

    pivot = summary.pivot_table(
        index="attack_type",
        columns="eval_config",
        values=metric,
        aggfunc="mean",
    )

    attacks = [a for a in ATTACK_ORDER if a in pivot.index]
    attacks += [a for a in pivot.index if a not in attacks]

    configs = [c for c in CONFIG_ORDER if c in pivot.columns]
    configs += [c for c in pivot.columns if c not in configs]

    pivot = pivot.reindex(index=attacks, columns=configs)

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


def plot_metric_by_attack_lines(summary, metric, title, ylabel, filename):
    """
    Biểu đồ đường gộp:
    - Trục X: attack_type
    - Mỗi line: eval_config
    - Trục Y: metric
    """
    if metric not in summary.columns:
        print(f"[SKIP] missing metric: {metric}")
        return

    attacks = ordered_attacks(summary)
    configs = ordered_configs(summary)

    data = (
        summary.groupby(["attack_type", "eval_config"], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    pivot = data.pivot(
        index="attack_type",
        columns="eval_config",
        values=metric,
    ).reindex(index=attacks, columns=configs)

    plt.figure(figsize=(13, 6))

    x = np.arange(len(pivot.index))

    for config in pivot.columns:
        y = pd.to_numeric(pivot[config], errors="coerce").values
        if np.all(pd.isna(y)):
            continue
        plt.plot(x, y, marker="o", label=config)

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


def plot_metric_by_attack_lines_attack_only(summary, metric, title, ylabel, filename):
    """
    Giống plot_metric_by_attack_lines nhưng bỏ normal.
    Dùng cho các chỉ số tấn công.
    """
    if "attack_type" not in summary.columns:
        return

    attack_summary = summary[summary["attack_type"].astype(str) != "normal"].copy()
    plot_metric_by_attack_lines(
        attack_summary,
        metric,
        title,
        ylabel,
        filename,
    )


def plot_action_distribution(runtime, action_col, filename, title):
    if action_col not in runtime.columns:
        print(f"[SKIP] missing action col: {action_col}")
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
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print("[PLOT]", path)


def plot_action_by_attack_lines(summary, action_rate_col, filename, title):
    if action_rate_col not in summary.columns:
        print(f"[SKIP] missing action rate col: {action_rate_col}")
        return

    plot_metric_by_attack_lines(
        summary,
        action_rate_col,
        title,
        "Action rate",
        filename,
    )


def plot_runtime_multi_attack_timeline(runtime, metric, filename, title):
    """
    Timeline gộp nhiều attack:
    - Chỉ dùng phase=attack
    - Mỗi attack lấy trung bình theo step_index
    - Mỗi line là attack type
    """
    if metric not in runtime.columns:
        print(f"[SKIP] missing runtime metric: {metric}")
        return

    df = runtime.copy()

    if "phase" in df.columns:
        df = df[df["phase"].astype(str) == "attack"].copy()

    if df.empty:
        print(f"[SKIP] no attack phase for {metric}")
        return

    if "attack_type" not in df.columns:
        return

    df = df[df["attack_type"].astype(str) != "normal"].copy()

    if df.empty:
        return

    plt.figure(figsize=(13, 6))

    for attack in ATTACK_ORDER:
        if attack == "normal":
            continue

        g = df[df["attack_type"].astype(str) == attack].copy()
        if g.empty:
            continue

        g = g.reset_index(drop=True)
        values = pd.to_numeric(g[metric], errors="coerce")
        plt.plot(range(len(values)), values, label=attack)

    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print("[PLOT]", path)


def main():
    ensure_out()

    summary = pd.read_csv(SUMMARY_PATH)
    runtime = pd.read_csv(RUNTIME_LOG)

    summary, runtime = clean_data(summary, runtime)

    print("[INFO] attacks in summary:", sorted(summary["attack_type"].unique()) if "attack_type" in summary.columns else "N/A")
    print("[INFO] configs in summary:", sorted(summary["eval_config"].unique()) if "eval_config" in summary.columns else "N/A")

    # =====================================================
    # Bar charts by config
    # =====================================================
    plot_bar_by_config(
        summary,
        "mean_latency",
        "Average latency by evaluation config",
        "Mean latency",
        "bar_mean_latency_by_config.png",
    )

    plot_bar_by_config(
        summary,
        "mean_packet_loss",
        "Average packet loss by evaluation config",
        "Mean packet loss",
        "bar_mean_packet_loss_by_config.png",
    )

    plot_bar_by_config(
        summary,
        "recovery_time_steps",
        "Recovery time by evaluation config",
        "Recovery time steps",
        "bar_recovery_time_by_config.png",
    )

    plot_bar_by_config(
        summary,
        "cumulative_reward",
        "Cumulative reward by evaluation config",
        "Cumulative reward",
        "bar_cumulative_reward_by_config.png",
    )

    plot_bar_by_config(
        summary,
        "false_positive_rate",
        "False positive rate by evaluation config",
        "False positive rate",
        "bar_false_positive_by_config.png",
    )

    if "defense_score" in summary.columns:
        plot_bar_by_config(
            summary,
            "defense_score",
            "Defense score by evaluation config",
            "Defense score",
            "bar_defense_score_by_config.png",
        )

    # =====================================================
    # Heatmaps
    # =====================================================
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

    if "defense_score" in summary.columns:
        plot_heatmap(
            summary,
            "defense_score",
            "Attack × config heatmap: defense score",
            "heatmap_attack_config_defense_score.png",
        )

    # =====================================================
    # New combined line charts: attack type on X axis
    # =====================================================
    plot_metric_by_attack_lines(
        summary,
        "mean_reward",
        "Mean reward across attack types",
        "Mean reward",
        "line_attack_mean_reward_by_config.png",
    )

    plot_metric_by_attack_lines(
        summary,
        "mean_latency",
        "Mean latency across attack types",
        "Mean latency",
        "line_attack_mean_latency_by_config.png",
    )

    plot_metric_by_attack_lines(
        summary,
        "mean_packet_loss",
        "Mean packet loss across attack types",
        "Mean packet loss",
        "line_attack_mean_packet_loss_by_config.png",
    )

    plot_metric_by_attack_lines(
        summary,
        "mean_controller_cpu",
        "Controller CPU across attack types",
        "Mean controller CPU",
        "line_attack_controller_cpu_by_config.png",
    )

    plot_metric_by_attack_lines(
        summary,
        "action_nonzero_rate",
        "Non-zero action rate across attack types",
        "Non-zero action rate",
        "line_attack_action_nonzero_rate_by_config.png",
    )

    plot_metric_by_attack_lines(
        summary,
        "switching_rate",
        "Action switching rate across attack types",
        "Switching rate",
        "line_attack_switching_rate_by_config.png",
    )

    plot_metric_by_attack_lines(
        summary,
        "recovery_time_steps",
        "Recovery time across attack types",
        "Recovery time steps",
        "line_attack_recovery_time_by_config.png",
    )

    if "defense_score" in summary.columns:
        plot_metric_by_attack_lines(
            summary,
            "defense_score",
            "Defense score across attack types",
            "Defense score",
            "line_attack_defense_score_by_config.png",
        )

    # Action-specific lines
    for a in range(5):
        col = f"action_{a}_rate"
        plot_action_by_attack_lines(
            summary,
            col,
            f"line_attack_action_{a}_rate_by_config.png",
            f"Action {a} rate across attack types",
        )

    # =====================================================
    # Action distributions
    # =====================================================
    plot_action_distribution(
        runtime,
        "action_requested",
        "action_requested_distribution_stacked.png",
        "Requested action distribution by config",
    )

    plot_action_distribution(
        runtime,
        "action_final",
        "action_final_distribution_stacked.png",
        "Final action distribution after Guard/Twin by config",
    )

    # Backward-compatible name
    plot_action_distribution(
        runtime,
        "action_final",
        "action_distribution_stacked.png",
        "Final action distribution by config",
    )

    # =====================================================
    # Digital Twin
    # =====================================================
    if "twin_reject_rate" in summary.columns:
        twin_rows = summary[
            summary["eval_config"].astype(str).str.contains("twin|full", regex=True)
        ].copy()

        if not twin_rows.empty:
            plot_bar_by_config(
                twin_rows,
                "twin_reject_rate",
                "Digital Twin rejected action rate",
                "Reject rate",
                "twin_reject_rate_by_config.png",
            )

            plot_bar_by_config(
                twin_rows,
                "mean_gap_latency",
                "Digital Twin latency sim-to-real gap",
                "Mean latency gap",
                "twin_gap_latency_by_config.png",
            )

            plot_bar_by_config(
                twin_rows,
                "mean_gap_loss",
                "Digital Twin packet loss sim-to-real gap",
                "Mean loss gap",
                "twin_gap_loss_by_config.png",
            )

    # =====================================================
    # Runtime timelines combined by attack
    # Không tách từng attack nữa
    # =====================================================
    plot_runtime_multi_attack_timeline(
        runtime,
        "latency",
        "timeline_all_attacks_latency.png",
        "Latency timeline across all attack types",
    )

    plot_runtime_multi_attack_timeline(
        runtime,
        "packet_loss",
        "timeline_all_attacks_packet_loss.png",
        "Packet loss timeline across all attack types",
    )

    plot_runtime_multi_attack_timeline(
        runtime,
        "controller_cpu",
        "timeline_all_attacks_controller_cpu.png",
        "Controller CPU timeline across all attack types",
    )

    plot_runtime_multi_attack_timeline(
        runtime,
        "reward",
        "timeline_all_attacks_reward.png",
        "Reward timeline across all attack types",
    )


if __name__ == "__main__":
    main()