import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = os.getenv("RUNTIME_LOG", "logs/runtime_eval.csv")
OUT_DIR = os.getenv("EVAL_OUT_DIR", "results/evaluation")


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="float64")

    return pd.to_numeric(df[col], errors="coerce").dropna()


def switching_rate(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(int).tolist()

    if len(values) < 2:
        return 0.0

    switch_count = sum(
        current != nxt
        for current, nxt in zip(values, values[1:])
    )

    return float(switch_count / (len(values) - 1))


def safe_mean(group: pd.DataFrame, col: str, default: float = 0.0) -> float:
    values = numeric_series(group, col)

    if values.empty:
        return default

    return float(values.mean())


def safe_std(group: pd.DataFrame, col: str, default: float = 0.0) -> float:
    values = numeric_series(group, col)

    if values.empty:
        return default

    return float(values.std())


def safe_var(group: pd.DataFrame, col: str, default: float = 0.0) -> float:
    values = numeric_series(group, col)

    if values.empty:
        return default

    return float(values.var())


def plot_timeline(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return

    plt.figure(figsize=(10, 5))

    if "mode" in df.columns:
        group_key = "mode"
    else:
        group_key = None

    if group_key:
        for mode, group in df.groupby(group_key):
            values = numeric_series(group, col)

            if values.empty:
                continue

            plt.plot(
                list(range(len(values))),
                values.to_numpy(dtype=float),
                label=str(mode),
            )
    else:
        values = numeric_series(df, col)
        if values.empty:
            plt.close()
            return

        plt.plot(
            list(range(len(values))),
            values.to_numpy(dtype=float),
            label=col,
        )

    plt.title(col)
    plt.xlabel("Step")
    plt.ylabel(col)

    if plt.gca().has_data():
        plt.legend()

    plt.tight_layout()

    fig_path = os.path.join(OUT_DIR, f"{col}_timeline_by_mode.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"[SUMMARY] saved figure: {fig_path}")


def build_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for keys, group in df.groupby(group_cols, dropna=False):
        group_df = pd.DataFrame(group)

        if not isinstance(keys, tuple):
            keys = (keys,)

        row: dict[str, Any] = {
            col: value
            for col, value in zip(group_cols, keys)
        }

        twin_safe_mean = safe_mean(group_df, "twin_safe", default=1.0)

        row.update(
            {
                "mean_reward": safe_mean(group_df, "reward"),
                "std_reward": safe_std(group_df, "reward"),
                "mean_latency": safe_mean(group_df, "latency"),
                "latency_var": safe_var(group_df, "latency"),
                "mean_packet_loss": safe_mean(group_df, "packet_loss"),
                "mean_packet_rate": safe_mean(group_df, "packet_rate"),
                "mean_byte_rate": safe_mean(group_df, "byte_rate"),
                "mean_flow_count": safe_mean(group_df, "flow_count"),
                "mean_flow_growth_rate": safe_mean(group_df, "flow_growth_rate"),
                "mean_src_ip_entropy": safe_mean(group_df, "src_ip_entropy"),
                "mean_controller_cpu": safe_mean(group_df, "controller_cpu"),
                "switching_rate": switching_rate(group_df["action"])
                if "action" in group_df.columns
                else 0.0,
                "mean_twin_gap_latency": safe_mean(group_df, "gap_latency"),
                "mean_twin_gap_loss": safe_mean(group_df, "gap_loss"),
                "twin_reject_rate": float(1.0 - twin_safe_mean),
                "samples": int(len(group_df)),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Runtime log not found: {INPUT_PATH}. "
            f"Run control_loop first."
        )

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        raise RuntimeError(f"Runtime log is empty: {INPUT_PATH}")

    if "mode" not in df.columns:
        raise ValueError("runtime_eval.csv must contain column: mode")

    preferred_group_cols = [
        "attack_type",
        "intensity",
        "run_id",
        "mode",
        "model",
    ]

    group_cols = [
        col for col in preferred_group_cols
        if col in df.columns
    ]

    if "mode" not in group_cols:
        group_cols.append("mode")

    # =========================
    # Full benchmark summary:
    # attack_type x intensity x run_id x mode x model
    # =========================
    summary_full = build_summary(df, group_cols)

    full_path = os.path.join(OUT_DIR, "summary_full_benchmark.csv")
    summary_full.to_csv(full_path, index=False)

    print("[SUMMARY] full benchmark:")
    print(summary_full.to_string(index=False))
    print(f"[SUMMARY] saved: {full_path}")

    # =========================
    # Compatibility file:
    # summary_4_modes.csv
    # =========================
    mode_group_cols = [
        col for col in ["mode", "model"]
        if col in df.columns
    ]

    if not mode_group_cols:
        mode_group_cols = ["mode"]

    summary_mode = build_summary(df, mode_group_cols)

    mode_path = os.path.join(OUT_DIR, "summary_4_modes.csv")
    summary_mode.to_csv(mode_path, index=False)

    by_mode_path = os.path.join(OUT_DIR, "summary_by_mode.csv")
    summary_mode.to_csv(by_mode_path, index=False)

    print("[SUMMARY] by mode:")
    print(summary_mode.to_string(index=False))
    print(f"[SUMMARY] saved: {mode_path}")
    print(f"[SUMMARY] saved: {by_mode_path}")

    for col in [
        "reward",
        "packet_rate",
        "byte_rate",
        "flow_count",
        "flow_growth_rate",
        "src_ip_entropy",
        "latency",
        "packet_loss",
        "controller_cpu",
        "action",
        "gap_latency",
        "gap_loss",
    ]:
        plot_timeline(df, col)


if __name__ == "__main__":
    main()