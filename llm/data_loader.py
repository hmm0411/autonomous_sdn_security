import glob
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except Exception:
    EventAccumulator = None  # type: ignore
    TENSORBOARD_AVAILABLE = False


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(ROOT, "runs")
MODELS_DIR = os.path.join(ROOT, "models")
DATA_PATH = os.path.join(ROOT, "data", "processed", "test_data.csv")

ACTION_MAP = {
    0: "no_action",
    1: "block_suspicious_flow",
    2: "limit_bandwidth",
    3: "redirect_traffic",
    4: "isolate_device",
}

STATE_KEYS = [
    "packet_rate",
    "byte_rate",
    "flow_count",
    "flow_growth_rate",
    "src_ip_entropy",
    "latency",
    "packet_loss",
    "controller_cpu",
]


def load_tb_scalars(run_dir: str) -> pd.DataFrame:
    if not TENSORBOARD_AVAILABLE or EventAccumulator is None:
        return pd.DataFrame()

    event_acc = EventAccumulator(run_dir)
    event_acc.Reload()

    tags = event_acc.Tags().get("scalars", [])
    series: Dict[str, Dict[int, float]] = {}

    for tag in tags:
        scalars = event_acc.Scalars(tag)
        short_name = tag.split("/")[-1].lower()
        series[short_name] = {scalar.step: float(scalar.value) for scalar in scalars}

    if not series:
        return pd.DataFrame()

    first_series = next(iter(series.values()))
    all_steps = sorted(first_series.keys())

    rows: List[Dict[str, Any]] = []

    for step in all_steps:
        row: Dict[str, Any] = {"episode": step}
        for col, data in series.items():
            row[col] = data.get(step, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def collect_all_runs(model_prefix: str = "") -> Dict[str, pd.DataFrame]:
    pattern = os.path.join(RUNS_DIR, f"{model_prefix}*")
    run_dirs = [
        directory
        for directory in glob.glob(pattern)
        if os.path.isdir(directory)
    ]

    result: Dict[str, pd.DataFrame] = {}

    for run_dir in sorted(run_dirs):
        run_name = os.path.basename(run_dir)
        try:
            df = load_tb_scalars(run_dir)
            if not df.empty:
                result[run_name] = df
        except Exception as e:
            print(f"[WARN] Cannot read {run_name}: {e}")

    return result


def get_training_summary() -> pd.DataFrame:
    all_runs = collect_all_runs()
    rows: List[Dict[str, Any]] = []

    for run_name, df in all_runs.items():
        if "reward" not in df.columns and "total_reward" not in df.columns:
            continue

        reward_col = "total_reward" if "total_reward" in df.columns else "reward"
        rewards = pd.to_numeric(df[reward_col], errors="coerce").dropna()

        if rewards.empty:
            continue

        model = (
            "dqn"
            if "dqn" in run_name.lower()
            else "ppo"
            if "ppo" in run_name.lower()
            else "unknown"
        )

        seed = run_name.replace("dqn_seed_", "").replace("ppo_seed_", "")

        row: Dict[str, Any] = {
            "run": run_name,
            "model": model.upper(),
            "seed": seed,
            "total_episodes": int(len(rewards)),
            "best_reward": round(float(rewards.max()), 4),
            "final_reward": round(float(rewards.iloc[-1]), 4),
            "mean_reward_last100": round(float(rewards.tail(100).mean()), 4),
            "mean_reward_all": round(float(rewards.mean()), 4),
        }

        loss_col = next((col for col in df.columns if "loss" in col.lower()), None)

        if loss_col:
            loss_values = pd.to_numeric(df[loss_col], errors="coerce").dropna()
            if not loss_values.empty:
                row["final_loss"] = round(float(loss_values.iloc[-1]), 6)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("model")


def _ensure_test_data() -> None:
    if os.path.exists(DATA_PATH):
        return

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    mock_data_path = os.path.join(ROOT, "rl_engine", "mock_data.py")

    if os.path.exists(mock_data_path):
        subprocess.run(
            [sys.executable, mock_data_path],
            cwd=ROOT,
            check=False,
        )


def _load_dqn():
    import torch
    from rl_engine.agent.dqn_agent import DQNAgent
    from rl_engine.config import ACTION_DIM, STATE_DIM

    agent = DQNAgent(STATE_DIM, ACTION_DIM)
    path = os.path.join(MODELS_DIR, "dqn_model.pth")

    checkpoint = torch.load(path, map_location=agent.device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        agent.q_net.load_state_dict(checkpoint["model_state_dict"])
    else:
        agent.q_net.load_state_dict(checkpoint)

    agent.q_net.eval()

    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0

    return agent, "DQN"


def _load_ppo():
    from rl_engine.agent.ppo_agent import PPOAgent
    from rl_engine.config import ACTION_DIM, STATE_DIM

    agent = PPOAgent(STATE_DIM, ACTION_DIM)
    agent.load(os.path.join(MODELS_DIR, "ppo_model.pth"))

    return agent, "PPO"


def _state_to_dict(state) -> Dict[str, float]:
    state_arr = np.asarray(state, dtype=float).reshape(-1)

    if len(state_arr) < 8:
        raise ValueError(f"Expected state len >= 8, got {len(state_arr)}")

    return {
        "packet_rate (pkt/s)": round(float(state_arr[0]), 4),
        "byte_rate (Bps)": round(float(state_arr[1]), 4),
        "flow_count": round(float(state_arr[2]), 4),
        "flow_growth_rate": round(float(state_arr[3]), 4),
        "src_ip_entropy (bits)": round(float(state_arr[4]), 4),
        "latency (ms)": round(float(state_arr[5]), 4),
        "packet_loss (%)": round(float(state_arr[6]), 4),
        "controller_cpu (0-100)": round(float(state_arr[7]), 4),
    }


def _state_to_qos(state) -> Dict[str, Any]:
    state_arr = np.asarray(state, dtype=float).reshape(-1)

    return {
        "latency": round(float(state_arr[5]), 4),
        "packet_loss": round(float(state_arr[6]), 4),
        "throughput": None,
    }


def select_agent_action(agent, state) -> int:
    predict_fn = getattr(agent, "predict", None)

    if callable(predict_fn):
        action = predict_fn(state)
    else:
        select_action_fn = getattr(agent, "select_action", None)

        if not callable(select_action_fn):
            raise AttributeError(
                f"Agent {type(agent).__name__} has neither predict() nor select_action()."
            )

        action = select_action_fn(state)

    if isinstance(action, tuple):
        action = action[0]

    if isinstance(action, np.ndarray):
        if action.size == 0:
            raise ValueError("Received empty ndarray action")
        action = action.reshape(-1)[0]

    return int(action)


def replay_model(
    model_type: str = "dqn",
    n_steps: int = 50,
    attack_filter: str = "all",
) -> List[Dict[str, Any]]:
    _ensure_test_data()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing test data: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    from rl_engine.offline_env import OfflineSDNEnv

    env = OfflineSDNEnv(dataframe=df)

    if model_type == "ppo":
        agent, model_name = _load_ppo()

    elif model_type == "best":
        summary = get_training_summary()

        if summary.empty:
            agent, model_name = _load_dqn()
        else:
            best_row = summary.loc[summary["best_reward"].idxmax()]
            selected = str(best_row["model"]).lower()
            agent, model_name = _load_dqn() if selected == "dqn" else _load_ppo()

    else:
        agent, model_name = _load_dqn()

    state, _ = env.reset()

    records: List[Dict[str, Any]] = []
    step = 0

    while len(records) < n_steps and step < len(df):
        action = select_agent_action(agent, state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        row = df.iloc[step]
        attack_indicator = float(row.get("attack_indicator", 0.0))

        if attack_filter == "attack" and attack_indicator <= 0.4:
            state = next_state
            step += 1
            continue

        if attack_filter == "normal" and attack_indicator > 0.2:
            state = next_state
            step += 1
            continue

        if attack_indicator == 0:
            attack_label = "no_attack"
        elif attack_indicator <= 0.4:
            attack_label = "low_threat"
        elif attack_indicator <= 0.6:
            attack_label = "medium_threat"
        else:
            attack_label = "high_threat"

        records.append(
            {
                "step": int(step),
                "model": model_name,
                "action_id": int(action),
                "action_name": ACTION_MAP.get(int(action), "unknown"),
                "reward": round(float(reward), 4),
                "state_dict": _state_to_dict(state),
                "qos": _state_to_qos(state),
                "attack_indicator": round(float(attack_indicator), 3),
                "attack_label": attack_label,
            }
        )

        state = next_state
        step += 1

        if terminated or truncated:
            state, _ = env.reset()

    return records


def sample_decisions(
    model_type: str = "dqn",
    n_normal: int = 5,
    n_attack: int = 10,
    n_noisy: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    normal_records = replay_model(
        model_type,
        n_steps=n_normal * 3,
        attack_filter="normal",
    )[:n_normal]

    attack_records = replay_model(
        model_type,
        n_steps=n_attack * 3,
        attack_filter="attack",
    )[:n_attack]

    noisy_records: List[Dict[str, Any]] = []

    for record in replay_model(
        model_type,
        n_steps=n_noisy * 4,
        attack_filter="all",
    )[:n_noisy]:
        noisy_record = dict(record)
        noisy_state = dict(record["state_dict"])

        for key in [
            "packet_rate (pkt/s)",
            "src_ip_entropy (bits)",
            "controller_cpu (0-100)",
        ]:
            noisy_state[key] = "N/A"

        noisy_record["state_dict"] = noisy_state
        noisy_record["attack_label"] = "data_missing"

        noisy_records.append(noisy_record)

    return {
        "normal": normal_records,
        "attack": attack_records,
        "noisy": noisy_records,
    }


def export_replay_to_csv(records: List[Dict[str, Any]], path: str) -> None:
    rows: List[Dict[str, Any]] = []

    for record in records:
        row: Dict[str, Any] = {
            "step": record["step"],
            "model": record["model"],
            "action_id": record["action_id"],
            "action_name": record["action_name"],
            "reward": record["reward"],
            "attack_indicator": record["attack_indicator"],
            "attack_label": record["attack_label"],
        }

        row.update(record["state_dict"])
        row.update({f"qos_{key}": value for key, value in record["qos"].items()})

        rows.append(row)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Exported {len(rows)} records -> {path}")


if __name__ == "__main__":
    print("=== Training Summary ===")
    summary_df = get_training_summary()

    if summary_df.empty:
        print("No TensorBoard summary found or tensorboard package missing.")
    else:
        print(summary_df.to_string(index=False))

    print("\n=== Sample Decisions ===")
    decisions = sample_decisions("dqn", n_normal=3, n_attack=5, n_noisy=2)

    for scenario, records in decisions.items():
        print(f"\n[{scenario.upper()}] {len(records)} records")

        for record in records:
            print(
                f"step={record['step']:>3} "
                f"action={record['action_name']:<25} "
                f"reward={record['reward']:>7.3f} "
                f"attack={record['attack_indicator']:.2f} "
                f"({record['attack_label']})"
            )

    all_records = replay_model("dqn", n_steps=100)
    export_replay_to_csv(all_records, "logs/dqn_replay.csv")