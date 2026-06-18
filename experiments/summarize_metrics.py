import os
from typing import Any

import numpy as np
import pandas as pd


INPUT_PATH = os.getenv("RUNTIME_LOG", "logs/runtime_eval.csv")
OUT_DIR = os.getenv("EVAL_OUT_DIR", "results/evaluation")


# =========================================================
# Basic helpers
# =========================================================
def to_num(df: pd.DataFrame, col: str, default: float | None = None) -> pd.Series:
    if col not in df.columns:
        if default is None:
            return pd.Series(dtype="float64")
        return pd.Series([default] * len(df), index=df.index, dtype="float64")

    return pd.to_numeric(df[col], errors="coerce")


def clean_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return to_num(df, col, default=default).fillna(default)


def safe_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    values = to_num(df, col)
    if values.empty:
        return default
    value = values.mean()
    return default if pd.isna(value) else float(value)


def safe_std(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    values = to_num(df, col)
    if values.empty:
        return default
    value = values.std()
    return default if pd.isna(value) else float(value)


def safe_var(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    values = to_num(df, col)
    if values.empty:
        return default
    value = values.var()
    return default if pd.isna(value) else float(value)


def safe_sum(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    values = to_num(df, col)
    if values.empty:
        return default
    value = values.sum()
    return default if pd.isna(value) else float(value)


def phase_filter(group: pd.DataFrame, phase_name: str) -> pd.DataFrame:
    if "phase" not in group.columns:
        return pd.DataFrame(columns=group.columns)
    return group[group["phase"].astype(str).str.lower() == phase_name].copy()


def metric_phase_group(group: pd.DataFrame) -> pd.DataFrame:
    """
    Main benchmark metrics are computed on attack phase only.
    If there is no phase column or no attack rows, fall back to full group.
    """
    attack = phase_filter(group, "attack")
    if attack.empty:
        return group
    return attack


def get_action_series(group: pd.DataFrame, preferred_col: str = "action_final") -> pd.Series:
    if preferred_col in group.columns:
        return clean_numeric(group, preferred_col, default=0.0).astype(int)

    if "action" in group.columns:
        return clean_numeric(group, "action", default=0.0).astype(int)

    return pd.Series([0] * len(group), index=group.index, dtype="int64")


def switching_rate(actions: pd.Series) -> float:
    values = pd.to_numeric(actions, errors="coerce").dropna().astype(int).tolist()
    if len(values) < 2:
        return 0.0
    switches = sum(a != b for a, b in zip(values, values[1:]))
    return float(switches / (len(values) - 1))


def action_entropy(actions: pd.Series) -> float:
    values = pd.to_numeric(actions, errors="coerce").dropna().astype(int)
    if values.empty:
        return 0.0
    probs = values.value_counts(normalize=True).values
    return float(-(probs * np.log2(probs + 1e-12)).sum())


# =========================================================
# Phase-based metrics
# =========================================================
def recovery_time_steps(group: pd.DataFrame) -> float:
    if "phase" not in group.columns or "sla_violation" not in group.columns:
        return 0.0

    rec = phase_filter(group, "recovery")
    if rec.empty:
        return 0.0

    return float(clean_numeric(rec, "sla_violation", default=0.0).sum())


def mitigation_success_rate(group: pd.DataFrame) -> float:
    if "phase" not in group.columns or "sla_violation" not in group.columns:
        return 0.0

    attack = phase_filter(group, "attack")
    if attack.empty:
        return 0.0

    violations = clean_numeric(attack, "sla_violation", default=0.0)
    return float(1.0 - violations.mean())


# =========================================================
# Group summarization
# =========================================================
def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    metric_group = metric_phase_group(group)

    actions_final = get_action_series(metric_group, preferred_col="action_final")

    if "action_requested" in metric_group.columns:
        actions_requested = clean_numeric(metric_group, "action_requested", default=0.0).astype(int)
    else:
        actions_requested = actions_final.copy()

    attack_type = (
        str(group["attack_type"].iloc[0]).lower()
        if "attack_type" in group.columns and not group.empty
        else "unknown"
    )

    action_changed_rate = (
        float((actions_requested != actions_final).mean()) if len(metric_group) else 0.0
    )
    action_nonzero_rate = float((actions_final != 0).mean()) if len(metric_group) else 0.0
    action_requested_nonzero_rate = (
        float((actions_requested != 0).mean()) if len(metric_group) else 0.0
    )

    out: dict[str, Any] = {
        "samples": int(len(group)),
        "attack_phase_samples": int(len(metric_group)),

        # Main metrics: attack phase only when available.
        "mean_reward": safe_mean(metric_group, "reward"),
        "std_reward": safe_std(metric_group, "reward"),
        "cumulative_reward": safe_sum(metric_group, "reward"),

        "mean_latency": safe_mean(metric_group, "latency"),
        "std_latency": safe_std(metric_group, "latency"),
        "latency_var": safe_var(metric_group, "latency"),
        "mean_packet_loss": safe_mean(metric_group, "packet_loss"),
        "std_packet_loss": safe_std(metric_group, "packet_loss"),
        "mean_packet_rate": safe_mean(metric_group, "packet_rate"),
        "mean_byte_rate": safe_mean(metric_group, "byte_rate"),
        "mean_flow_count": safe_mean(metric_group, "flow_count"),
        "mean_flow_growth_rate": safe_mean(metric_group, "flow_growth_rate"),
        "mean_src_ip_entropy": safe_mean(metric_group, "src_ip_entropy"),
        "mean_controller_cpu": safe_mean(metric_group, "controller_cpu"),

        # Overall metrics kept for diagnostics.
        "overall_mean_reward": safe_mean(group, "reward"),
        "overall_mean_latency": safe_mean(group, "latency"),
        "overall_mean_packet_loss": safe_mean(group, "packet_loss"),
        "overall_mean_packet_rate": safe_mean(group, "packet_rate"),
        "overall_mean_controller_cpu": safe_mean(group, "controller_cpu"),

        # Decision behavior: attack phase when available.
        "switching_rate": switching_rate(actions_final),
        "action_entropy": action_entropy(actions_final),
        "action_nonzero_rate": action_nonzero_rate,
        "action_requested_nonzero_rate": action_requested_nonzero_rate,
        "action_changed_rate": action_changed_rate,

        # SLA / mitigation.
        "sla_violation_rate": safe_mean(metric_group, "sla_violation"),
        "overall_sla_violation_rate": safe_mean(group, "sla_violation"),
        "recovery_time_steps": recovery_time_steps(group),
        "mitigation_success_rate": mitigation_success_rate(group),

        # Guard.
        "guard_enabled_rate": safe_mean(metric_group, "guard_enabled"),
        "guard_overrode_rate": safe_mean(metric_group, "guard_overrode"),

        # Digital Twin.
        "twin_enabled_rate": safe_mean(metric_group, "twin_enabled"),
        "twin_checked_rate": safe_mean(metric_group, "twin_checked"),
        "twin_safe_rate": safe_mean(metric_group, "twin_safe", default=1.0),
        "twin_reject_rate": safe_mean(metric_group, "twin_rejected"),
        "mean_gap_latency": safe_mean(metric_group, "gap_latency"),
        "mean_gap_loss": safe_mean(metric_group, "gap_loss"),
        "mean_pred_latency": safe_mean(metric_group, "pred_latency"),
        "mean_pred_loss": safe_mean(metric_group, "pred_loss"),

        # LLM runtime overhead.
        "llm_enabled_rate": safe_mean(metric_group, "llm_enabled"),
        "mean_llm_latency": safe_mean(metric_group, "llm_latency"),
    }

    for action in range(5):
        out[f"action_{action}_count"] = int((actions_final == action).sum())
        out[f"action_{action}_rate"] = (
            float((actions_final == action).mean()) if len(metric_group) else 0.0
        )
        out[f"requested_action_{action}_count"] = int((actions_requested == action).sum())
        out[f"requested_action_{action}_rate"] = (
            float((actions_requested == action).mean()) if len(metric_group) else 0.0
        )

    # False positive: any non-zero final action during normal traffic is unnecessary mitigation.
    out["false_positive_rate"] = action_nonzero_rate if attack_type == "normal" else 0.0

    return out


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "attack_type",
        "intensity",
        "run_id",
        "eval_config",
        "mode",
        "model",
    ]
    group_cols = [col for col in group_cols if col in df.columns]

    if not group_cols:
        raise ValueError("No grouping columns found. Expected at least eval_config/mode/model.")

    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(summarize_group(group))
        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# Derived comparison metrics
# =========================================================
def minmax_good(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    min_v = values.min()
    max_v = values.max()

    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series([0.5] * len(values), index=values.index, dtype="float64")

    norm = (values - min_v) / (max_v - min_v)
    return norm.fillna(0.0) if higher_is_better else (1.0 - norm).fillna(0.0)


def add_defense_score(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()

    summary["reward_score"] = minmax_good(summary["mean_reward"], higher_is_better=True)
    summary["latency_score"] = minmax_good(summary["mean_latency"], higher_is_better=False)
    summary["loss_score"] = minmax_good(summary["mean_packet_loss"], higher_is_better=False)
    summary["recovery_score"] = minmax_good(summary["recovery_time_steps"], higher_is_better=False)
    summary["stability_score"] = minmax_good(summary["switching_rate"], higher_is_better=False)
    summary["mitigation_score"] = clean_numeric(summary, "mitigation_success_rate", default=0.0)
    summary["normal_safety_score"] = 1.0 - clean_numeric(summary, "false_positive_rate", default=0.0)

    summary["defense_score"] = (
        0.25 * summary["reward_score"]
        + 0.20 * summary["latency_score"]
        + 0.20 * summary["loss_score"]
        + 0.15 * summary["recovery_score"]
        + 0.10 * summary["mitigation_score"]
        + 0.05 * summary["stability_score"]
        + 0.05 * summary["normal_safety_score"]
    )

    return summary


def add_improvement_vs_no_defense(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()

    compare_cols = [
        "latency_reduction_vs_no_defense_pct",
        "packet_loss_reduction_vs_no_defense_pct",
        "recovery_reduction_vs_no_defense_pct",
        "sla_reduction_vs_no_defense_pct",
        "reward_gain_vs_no_defense",
        "cumulative_reward_gain_vs_no_defense",
        "defense_score_gain_vs_no_defense",
    ]
    for col in compare_cols:
        summary[col] = np.nan

    base_group_cols = [
        col for col in ["attack_type", "intensity", "run_id"] if col in summary.columns
    ]

    if not base_group_cols or "eval_config" not in summary.columns:
        return summary

    for _, group_idx in summary.groupby(base_group_cols, dropna=False).groups.items():
        group = summary.loc[group_idx]
        baseline = group[group["eval_config"].astype(str) == "no_defense"]
        if baseline.empty:
            continue

        base = baseline.iloc[0]

        def reduction_pct(metric: str) -> pd.Series:
            base_value = float(base.get(metric, np.nan))
            if pd.isna(base_value) or base_value == 0:
                return pd.Series([np.nan] * len(group), index=group.index)
            return (base_value - pd.to_numeric(group[metric], errors="coerce")) / base_value * 100.0

        summary.loc[group.index, "latency_reduction_vs_no_defense_pct"] = reduction_pct("mean_latency")
        summary.loc[group.index, "packet_loss_reduction_vs_no_defense_pct"] = reduction_pct("mean_packet_loss")
        summary.loc[group.index, "recovery_reduction_vs_no_defense_pct"] = reduction_pct("recovery_time_steps")
        summary.loc[group.index, "sla_reduction_vs_no_defense_pct"] = reduction_pct("sla_violation_rate")

        summary.loc[group.index, "reward_gain_vs_no_defense"] = (
            pd.to_numeric(group["mean_reward"], errors="coerce") - float(base.get("mean_reward", 0.0))
        )
        summary.loc[group.index, "cumulative_reward_gain_vs_no_defense"] = (
            pd.to_numeric(group["cumulative_reward"], errors="coerce")
            - float(base.get("cumulative_reward", 0.0))
        )
        summary.loc[group.index, "defense_score_gain_vs_no_defense"] = (
            pd.to_numeric(group["defense_score"], errors="coerce")
            - float(base.get("defense_score", 0.0))
        )

    return summary


# =========================================================
# Output tables
# =========================================================
def save_outputs(summary: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    full_path = os.path.join(OUT_DIR, "summary_full_benchmark.csv")
    summary.to_csv(full_path, index=False)

    by_config = (
        summary.groupby(["eval_config"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
        if "eval_config" in summary.columns
        else pd.DataFrame()
    )
    by_config_path = os.path.join(OUT_DIR, "summary_by_config.csv")
    by_config.to_csv(by_config_path, index=False)

    summary_4_modes_path = os.path.join(OUT_DIR, "summary_4_modes.csv")
    by_config.to_csv(summary_4_modes_path, index=False)

    by_attack_config_cols = [
        col for col in ["attack_type", "intensity", "eval_config"] if col in summary.columns
    ]
    if by_attack_config_cols:
        by_attack_config = (
            summary.groupby(by_attack_config_cols, dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )
    else:
        by_attack_config = summary.copy()

    by_attack_config_path = os.path.join(OUT_DIR, "summary_by_attack_config.csv")
    by_attack_config.to_csv(by_attack_config_path, index=False)

    action_cols = [
        col
        for col in summary.columns
        if col.startswith("action_")
        or col.startswith("requested_action_")
        or col in [
            "attack_type",
            "intensity",
            "run_id",
            "eval_config",
            "mode",
            "model",
            "samples",
            "attack_phase_samples",
            "switching_rate",
            "action_entropy",
            "action_nonzero_rate",
            "action_requested_nonzero_rate",
            "action_changed_rate",
            "false_positive_rate",
        ]
    ]
    actions_path = os.path.join(OUT_DIR, "summary_actions.csv")
    summary[action_cols].to_csv(actions_path, index=False)

    twin_cols = [
        col
        for col in [
            "attack_type",
            "intensity",
            "run_id",
            "eval_config",
            "mode",
            "model",
            "twin_enabled_rate",
            "twin_checked_rate",
            "twin_safe_rate",
            "twin_reject_rate",
            "mean_gap_latency",
            "mean_gap_loss",
            "mean_pred_latency",
            "mean_pred_loss",
            "latency_var",
            "mean_reward",
            "defense_score",
        ]
        if col in summary.columns
    ]
    twin_path = os.path.join(OUT_DIR, "summary_twin.csv")
    summary[twin_cols].to_csv(twin_path, index=False)

    llm_cols = [
        col
        for col in [
            "attack_type",
            "intensity",
            "run_id",
            "eval_config",
            "mode",
            "model",
            "llm_enabled_rate",
            "mean_llm_latency",
            "mean_reward",
            "defense_score",
        ]
        if col in summary.columns
    ]
    llm_runtime_path = os.path.join(OUT_DIR, "summary_llm_runtime.csv")
    llm_path = os.path.join(OUT_DIR, "summary_llm.csv")
    summary[llm_cols].to_csv(llm_runtime_path, index=False)
    summary[llm_cols].to_csv(llm_path, index=False)

    print("[OK] saved:", full_path)
    print("[OK] saved:", by_config_path)
    print("[OK] saved:", summary_4_modes_path)
    print("[OK] saved:", by_attack_config_path)
    print("[OK] saved:", actions_path)
    print("[OK] saved:", twin_path)
    print("[OK] saved:", llm_runtime_path)
    print("[OK] saved:", llm_path)


# =========================================================
# Main
# =========================================================
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Runtime log not found: {INPUT_PATH}. Run control_loop benchmark first."
        )

    df = pd.read_csv(INPUT_PATH)
    if df.empty:
        raise RuntimeError(f"Runtime log is empty: {INPUT_PATH}")

    if "eval_config" not in df.columns:
        if "mode" in df.columns and "model" in df.columns:
            df["eval_config"] = df["mode"].astype(str) + "_" + df["model"].astype(str)
            df.loc[df["mode"].astype(str) == "no_defense", "eval_config"] = "no_defense"
            df.loc[df["mode"].astype(str) == "rule", "eval_config"] = "rule"
        else:
            raise ValueError("Missing eval_config column and cannot infer from mode/model.")

    for col in ["attack_type", "intensity", "phase", "mode", "model", "eval_config"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

    summary = build_group_summary(df)
    summary = add_defense_score(summary)
    summary = add_improvement_vs_no_defense(summary)

    save_outputs(summary)


if __name__ == "__main__":
    main()
