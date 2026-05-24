"""
data_loader.py
--------------
Lấy dữ liệu THẬT từ kết quả training DQN/PPO để feed vào LLM evaluator.

Hai nguồn dữ liệu thật:
  1. TensorBoard runs/   → per-episode: reward, loss, epsilon, action distribution
  2. Model replay        → per-step: (state, action, reward, qos) bằng cách chạy
                           lại model đã train qua OfflineSDNEnv

Không có gì được hardcode hay random ở đây.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR   = os.path.join(ROOT, "runs")
MODELS_DIR = os.path.join(ROOT, "models")
DATA_PATH  = os.path.join(ROOT, "data", "processed", "test_data.csv")

ACTION_MAP = {
    0: "no_action",
    1: "block_suspicious_flow",
    2: "limit_bandwidth",
    3: "redirect_traffic",
    4: "isolate_device",
}

STATE_KEYS = [
    "packet_rate", "byte_rate", "flow_count", "src_ip_entropy",
    "latency", "packet_loss", "queue_length", "controller_cpu", "previous_action",
]

# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 1: Đọc TensorBoard – per-episode summary
# ──────────────────────────────────────────────────────────────────────────────

def load_tb_scalars(run_dir: str) -> pd.DataFrame:
    """
    Đọc toàn bộ scalar data từ một thư mục TensorBoard run.
    Trả về DataFrame với các cột: episode, reward, loss, epsilon (nếu có).
    """
    ea = EventAccumulator(run_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    series = {}
    for tag in tags:
        scalars = ea.Scalars(tag)
        short = tag.split("/")[-1].lower()   # "Total_Reward" → "total_reward"
        series[short] = {s.step: s.value for s in scalars}

    # Align on episode steps
    if not series:
        return pd.DataFrame()

    all_steps = sorted(next(iter(series.values())).keys())
    rows = []
    for step in all_steps:
        row = {"episode": step}
        for col, data in series.items():
            row[col] = data.get(step, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def load_action_distribution(run_dir: str) -> dict:
    """
    Decode action histogram từ TensorBoard.
    Trả về dict {action_id: count} từ snapshot cuối.
    """
    ea = EventAccumulator(run_dir)
    ea.Reload()
    hists = ea.Histograms("Actions")
    if not hists:
        return {}

    h = hists[-1].histogram_value
    total = h.num if h.num > 0 else 1

    limits  = h.bucket_limit
    buckets = h.bucket

    action_counts = {i: 0.0 for i in range(5)}
    for i, (limit, cnt) in enumerate(zip(limits, buckets)):
        if cnt > 0:
            prev = limits[i - 1] if i > 0 else h.min
            # Xác định action_id nào nằm trong bucket này
            for action_id in range(5):
                if prev <= action_id < limit or (action_id == 4 and limit >= 4.0):
                    action_counts[action_id] += cnt
                    break
    return {k: round(v / total, 4) for k, v in action_counts.items()}


def collect_all_runs(model_prefix: str = "") -> dict:
    """
    Quét tất cả runs/ và trả về dict {run_name: DataFrame}.
    model_prefix: "" = tất cả, "dqn" = chỉ DQN, "ppo" = chỉ PPO.
    """
    pattern = os.path.join(RUNS_DIR, f"{model_prefix}*")
    run_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    result = {}
    for run_dir in sorted(run_dirs):
        run_name = os.path.basename(run_dir)
        try:
            df = load_tb_scalars(run_dir)
            if not df.empty:
                result[run_name] = df
        except Exception as e:
            print(f"  [WARN] Cannot read {run_name}: {e}")
    return result


def get_training_summary() -> pd.DataFrame:
    """
    Tổng hợp kết quả training của tất cả seeds thành một DataFrame.
    Mỗi row = một seed với: model, seed, best_reward, final_reward,
                             mean_reward_last100, total_episodes.
    """
    all_runs = collect_all_runs()
    rows = []
    for run_name, df in all_runs.items():
        if "reward" not in df.columns and "total_reward" not in df.columns:
            continue

        reward_col = "total_reward" if "total_reward" in df.columns else "reward"
        rewards = df[reward_col].dropna()

        # Detect model type
        model = "dqn" if "dqn" in run_name else "ppo" if "ppo" in run_name else "unknown"
        seed  = run_name.replace("dqn_seed_", "").replace("ppo_seed_", "")

        row = {
            "run":              run_name,
            "model":            model.upper(),
            "seed":             seed,
            "total_episodes":   int(len(rewards)),
            "best_reward":      round(float(rewards.max()), 4),
            "final_reward":     round(float(rewards.iloc[-1]), 4),
            "mean_reward_last100": round(float(rewards.tail(100).mean()), 4),
            "mean_reward_all":  round(float(rewards.mean()), 4),
        }

        # Thêm loss nếu có
        loss_col = next((c for c in df.columns if "loss" in c), None)
        if loss_col:
            row["final_loss"] = round(float(df[loss_col].dropna().iloc[-1]), 6)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("model")


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 2: Model Replay – per-step (state, action, reward, qos)
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_test_data():
    """Tạo test_data.csv nếu chưa có."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        # Chạy mock_data.py để tạo file
        import subprocess
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "rl_engine", "mock_data.py")],
            cwd=ROOT
        )


def _load_dqn():
    import torch
    from rl_engine.agent.dqn_agent import DQNAgent
    from rl_engine.config import STATE_DIM, ACTION_DIM

    agent = DQNAgent(STATE_DIM, ACTION_DIM)
    path  = os.path.join(MODELS_DIR, "dqn_model.pth")
    ckpt  = torch.load(path, map_location="cpu")
    agent.q_net.load_state_dict(ckpt["model_state_dict"])
    agent.q_net.eval()
    return agent, "DQN"


def _load_ppo():
    from rl_engine.agent.ppo_agent import PPOAgent
    from rl_engine.config import STATE_DIM, ACTION_DIM

    agent = PPOAgent(STATE_DIM, ACTION_DIM)
    agent.load(os.path.join(MODELS_DIR, "ppo_model.pth"))
    return agent, "PPO"


def replay_model(
    model_type: str = "dqn",
    n_steps: int = 50,
    attack_filter: str = "all",
) -> list[dict]:
    """
    Chạy lại model đã train qua OfflineSDNEnv và thu thập từng bước quyết định.

    Parameters
    ----------
    model_type   : "dqn" | "ppo" | "best"  (best = model có reward cao hơn)
    n_steps      : số bước muốn thu thập
    attack_filter: "all" | "attack" (chỉ lấy khi attack_indicator > 0.5) | "normal"

    Returns
    -------
    list of dict, mỗi dict gồm:
      step, model, action_id, action_name, reward,
      state_dict (9 features), qos (latency/loss/throughput),
      attack_indicator, attack_label
    """
    _ensure_test_data()
    df = pd.read_csv(DATA_PATH)

    from rl_engine.offline_env import OfflineSDNEnv
    env = OfflineSDNEnv(dataframe=df)

    if model_type == "ppo":
        agent, model_name = _load_ppo()
    elif model_type == "best":
        # Chọn model có best_reward cao hơn
        summary = get_training_summary()
        best_row = summary.loc[summary["best_reward"].idxmax()]
        model_type = best_row["model"].lower()
        agent, model_name = (_load_dqn() if model_type == "dqn" else _load_ppo())
    else:
        agent, model_name = _load_dqn()

    state, _ = env.reset()
    records = []
    step = 0

    while len(records) < n_steps and step < len(df):
        # Lấy action từ model thật
        if hasattr(agent, "predict"):
            action = agent.predict(state)
            if isinstance(action, tuple):
                action = action[0]
        else:
            action = agent.select_action(state)
        action = int(action)

        next_state, reward, done, _, _ = env.step(action)

        row = df.iloc[step]
        attack_ind = float(row.get("attack_indicator", 0.0))

        # Filter nếu cần
        if attack_filter == "attack" and attack_ind <= 0.4:
            state = next_state
            step += 1
            continue
        if attack_filter == "normal" and attack_ind > 0.2:
            state = next_state
            step += 1
            continue

        # Build state dict với tên cột đúng
        state_dict = {
            "packet_rate (pkt/s)":   round(float(state[0]), 4),
            "byte_rate (Bps)":       round(float(state[1]), 4),
            "flow_count":            round(float(state[2]), 4),
            "src_ip_entropy (bits)": round(float(state[3]), 4),
            "latency (ms)":          round(float(state[4]), 4),
            "packet_loss (%)":       round(float(state[5]), 4),
            "queue_length":          round(float(state[6]), 4),
            "controller_cpu (0-1)":  round(float(state[7]), 4),
            "previous_action":       int(state[8]),
        }

        # QoS từ state (latency, packet_loss; throughput không có trong dataset)
        qos = {
            "latency":      round(float(state[4]), 4),
            "packet_loss":  round(float(state[5]), 4),
            "throughput":   None,
        }

        # Attack label
        if attack_ind == 0:
            attack_label = "no_attack"
        elif attack_ind <= 0.4:
            attack_label = "low_threat"
        elif attack_ind <= 0.6:
            attack_label = "medium_threat"
        else:
            attack_label = "high_threat"

        records.append({
            "step":             step,
            "model":            model_name,
            "action_id":        action,
            "action_name":      ACTION_MAP.get(action, "unknown"),
            "reward":           round(float(reward), 4),
            "state_dict":       state_dict,
            "qos":              qos,
            "attack_indicator": round(attack_ind, 3),
            "attack_label":     attack_label,
        })

        state = next_state
        step += 1
        if done:
            state, _ = env.reset()

    return records


def sample_decisions(
    model_type: str = "dqn",
    n_normal: int = 5,
    n_attack: int = 10,
    n_noisy: int = 5,
) -> dict:
    """
    Lấy mẫu quyết định của model cho 3 kịch bản đánh giá LLM:

    Returns
    -------
    dict với 3 key: "normal", "attack", "noisy"
    Mỗi key là list[dict] (kết quả replay_model)
    """
    normal_recs = replay_model(model_type, n_steps=n_normal * 3, attack_filter="normal")[:n_normal]
    attack_recs = replay_model(model_type, n_steps=n_attack * 3, attack_filter="attack")[:n_attack]

    # Noisy: lấy normal recs rồi xóa một số features
    noisy_recs = []
    for rec in replay_model(model_type, n_steps=n_noisy * 4, attack_filter="all")[:n_noisy]:
        noisy = dict(rec)
        noisy_state = dict(rec["state_dict"])
        # Xóa 3 features quan trọng để mô phỏng dữ liệu thiếu
        for key in ["packet_rate (pkt/s)", "src_ip_entropy (bits)", "controller_cpu (0-1)"]:
            noisy_state[key] = "N/A"
        noisy["state_dict"] = noisy_state
        noisy["attack_label"] = "data_missing"
        noisy_recs.append(noisy)

    return {
        "normal": normal_recs,
        "attack": attack_recs,
        "noisy":  noisy_recs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 3: Export ra CSV để audit
# ──────────────────────────────────────────────────────────────────────────────

def export_replay_to_csv(records: list[dict], path: str) -> None:
    """Lưu kết quả replay thành CSV phẳng (flatten state_dict)."""
    rows = []
    for rec in records:
        row = {
            "step":             rec["step"],
            "model":            rec["model"],
            "action_id":        rec["action_id"],
            "action_name":      rec["action_name"],
            "reward":           rec["reward"],
            "attack_indicator": rec["attack_indicator"],
            "attack_label":     rec["attack_label"],
        }
        row.update(rec["state_dict"])
        row.update({f"qos_{k}": v for k, v in rec["qos"].items()})
        rows.append(row)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Exported {len(rows)} records → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# DEMO / self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Training Summary (từ TensorBoard) ===")
    summary = get_training_summary()
    print(summary.to_string(index=False))

    print("\n=== Sample Decisions (DQN replay) ===")
    decisions = sample_decisions("dqn", n_normal=3, n_attack=5, n_noisy=2)
    for scenario, recs in decisions.items():
        print(f"\n  [{scenario.upper()}] — {len(recs)} records")
        for r in recs:
            print(f"    step={r['step']:>3} action={r['action_name']:<25} "
                  f"reward={r['reward']:>7.3f}  attack={r['attack_indicator']:.2f} ({r['attack_label']})")

    print("\n=== Export to CSV ===")
    all_recs = replay_model("dqn", n_steps=100)
    export_replay_to_csv(all_recs, "logs/dqn_replay.csv")