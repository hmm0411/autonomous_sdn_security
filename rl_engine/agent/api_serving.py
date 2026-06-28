import argparse
import os
import time

from flask import Flask, request, jsonify
import joblib
import mlflow
import mlflow.pytorch
import numpy as np
from prometheus_client import start_http_server, Counter, Histogram
import torch


app = Flask(__name__)

model_type = os.getenv("MODEL_TYPE", "dqn").strip().lower()
model_prod = None
model_staging = None
scaler = None


def load_scaler():
    from sklearn.utils.validation import check_is_fitted

    local_paths = [
        "/app/models/scaler.pkl",
        "/app/models/scaler_dqn.pkl" if model_type == "dqn" else "/app/models/scaler_ppo.pkl",
        "models/scaler.pkl",
        "models/scaler_dqn.pkl" if model_type == "dqn" else "models/scaler_ppo.pkl",
    ]

    for path in local_paths:
        if os.path.exists(path):
            try:
                s = joblib.load(path)
                check_is_fitted(s)
                print(f"[+] Load scaler local: {path}", flush=True)
                return s
            except Exception as e:
                print(f"[-] Cannot load scaler {path}: {e}", flush=True)

    print("[!] Không tìm thấy scaler local. Dùng identity scaler.", flush=True)
    return None


def choose_action(model, state_tensor):
    out = model(state_tensor)

    if isinstance(out, tuple):
        logits_or_policy = out[0]
    else:
        logits_or_policy = out

    return int(torch.argmax(logits_or_policy, dim=-1).item())


# ==========================================
# PROMETHEUS SERVING METRICS
# ==========================================
INFERENCE_REQUESTS = Counter(
    "serving_requests_total",
    "Tổng số lượng request gọi dự đoán",
    ["model"],
)

INFERENCE_LATENCY = Histogram(
    "serving_latency_seconds",
    "Độ trễ sinh ra quyết định của model",
    ["model"],
)

ACTION_CHOSEN = Counter(
    "serving_actions_total",
    "Tần suất chọn hành động của model",
    ["model", "stage", "action"],
)


def _registered_model_name() -> str:
    if model_type == "ppo":
        return os.getenv("PPO_REGISTERED_MODEL_NAME", "SDN_PPO_Model")

    return os.getenv("DQN_REGISTERED_MODEL_NAME", "SDN_DQN_Model")


def _local_model_paths() -> list[str]:
    if model_type == "ppo":
        return [
            "/app/models/ppo_model.pth",
            "models/ppo_model.pth",
            "/app/models/ppo.pth",
            "models/ppo.pth",
        ]

    return [
        "/app/models/dqn_model.pth",
        "models/dqn_model.pth",
        "/app/models/dqn.pth",
        "models/dqn.pth",
    ]


def _load_local_model(path: str):
    from rl_engine.config import STATE_DIM, ACTION_DIM

    if model_type == "dqn":
        from rl_engine.agent.dqn_agent import DQNAgent

        agent = DQNAgent(STATE_DIM, ACTION_DIM)
        agent.load(path)
        agent.q_net.eval()
        return agent.q_net

    if model_type == "ppo":
        from rl_engine.agent.ppo_agent import PPOAgent

        agent = PPOAgent(STATE_DIM, ACTION_DIM)
        agent.load(path)
        agent.model.eval()
        return agent.model

    raise ValueError(f"Unsupported MODEL_TYPE={model_type}")


def _load_production_model(registered_model_name: str):
    model_uri = f"models:/{registered_model_name}/Production"
    m = mlflow.pytorch.load_model(model_uri)
    m.eval()
    print(f"[+] Loaded {model_type.upper()} Production from MLflow: {model_uri}", flush=True)
    return m


def _load_staging_model(registered_model_name: str):
    model_uri = f"models:/{registered_model_name}/Staging"
    m = mlflow.pytorch.load_model(model_uri)
    m.eval()
    print(f"[+] Loaded {model_type.upper()} Staging from MLflow: {model_uri}", flush=True)
    return m


def load_models():
    global model_prod
    global model_staging
    global scaler

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[MLFLOW] tracking_uri={tracking_uri}", flush=True)

    registered_model_name = _registered_model_name()
    scaler = load_scaler()

    allow_untrained_fallback = os.getenv(
        "ALLOW_UNTRAINED_FALLBACK", "false"
    ).lower() == "true"

    model_prod = None
    model_staging = None

    # =========================
    # 1. Load Production model
    # =========================
    try:
        model_prod = _load_production_model(registered_model_name)

    except Exception as e:
        print(f"[MODEL_LOAD_ERROR] Production MLflow load failed: {e}", flush=True)

        loaded_local = False
        for local_path in _local_model_paths():
            if not os.path.exists(local_path):
                continue

            try:
                print(f"[+] Trying local model: {local_path}", flush=True)
                model_prod = _load_local_model(local_path)
                loaded_local = True
                print(f"[+] Loaded local {model_type.upper()} model: {local_path}", flush=True)
                break
            except Exception as local_error:
                print(
                    f"[-] Cannot load local model {local_path}: {local_error}",
                    flush=True,
                )

        if not loaded_local:
            if allow_untrained_fallback:
                print(
                    "[!] Using untrained Linear fallback. "
                    "Only use this for smoke test, not benchmark.",
                    flush=True,
                )
                model_prod = torch.nn.Linear(8, 5)
                model_prod.eval()
            else:
                raise RuntimeError(
                    "No valid trained Production model found. "
                    "Fix MLflow/local model path or set "
                    "ALLOW_UNTRAINED_FALLBACK=true only for smoke test."
                )

    # =========================
    # 2. Load Staging model
    # =========================
    try:
        model_staging = _load_staging_model(registered_model_name)

    except Exception as e:
        print(f"[-] No Staging model or load failed: {e}", flush=True)
        model_staging = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok" if model_prod is not None else "degraded",
            "model_type": model_type,
            "has_production_model": model_prod is not None,
            "has_staging_model": model_staging is not None,
            "has_scaler": scaler is not None,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    INFERENCE_REQUESTS.labels(model=model_type).inc()

    data = request.get_json(silent=True) or {}
    state_raw = data.get("state")

    if state_raw is None:
        return jsonify({"error": "Missing field: state"}), 400

    if not isinstance(state_raw, (list, tuple)):
        return jsonify({"error": "state must be a list of 8 numeric values"}), 400

    if len(state_raw) != 8:
        return jsonify({"error": "Invalid state size. Expected 8."}), 400

    try:
        state_np = np.array(state_raw, dtype=np.float32)

        if scaler is not None:
            state_scaled = scaler.transform([state_np])[0]
        else:
            state_scaled = state_np

        state_tensor = torch.FloatTensor(state_scaled).unsqueeze(0)

    except Exception as e:
        return jsonify({"error": f"Invalid state values: {e}"}), 400

    if model_prod is None:
        return jsonify({"error": "Production model not loaded"}), 503

    action_prod = 0
    action_staging = 0

    try:
        with torch.no_grad():
            action_prod = choose_action(model_prod, state_tensor)

            ACTION_CHOSEN.labels(
                model=model_type,
                stage="production",
                action=str(action_prod),
            ).inc()

            action_staging = action_prod

            if model_staging is not None:
                action_staging = choose_action(model_staging, state_tensor)

                ACTION_CHOSEN.labels(
                    model=model_type,
                    stage="staging",
                    action=str(action_staging),
                ).inc()

    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    latency = time.time() - start_time
    INFERENCE_LATENCY.labels(model=model_type).observe(latency)

    return jsonify(
        {
            "action": int(action_prod),
            "action_staging": int(action_staging),
            "model": model_type,
            "latency_seconds": float(latency),
        }
    )


@app.route("/reload", methods=["POST"])
def reload_model_api():
    try:
        print("[*] Nhận lệnh Reload Model từ MLOps Pipeline...", flush=True)
        load_models()
        return jsonify(
            {
                "status": "success",
                "message": "Đã cập nhật Model Production mới nhất!",
                "has_production_model": model_prod is not None,
                "has_staging_model": model_staging is not None,
            }
        ), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_models()

    metrics_port = 9002 if model_type == "dqn" else 9003
    start_http_server(metrics_port, addr="0.0.0.0")
    print(f"[+] Prometheus Serving Metrics started on port {metrics_port}", flush=True)
    app.run(host="0.0.0.0", port=args.port)
