import argparse
import time
from flask import Flask, request, jsonify
import mlflow.pytorch
import numpy as np
import os
import torch
import joblib
from prometheus_client import start_http_server, Counter, Histogram
from mlflow.artifacts import download_artifacts
from rl_engine.agent.dqn_agent import DQNAgent

app = Flask(__name__)

# Thay thế việc import từ control_loop bằng os.getenv
model_type = os.getenv("MODEL_TYPE", "dqn").strip().lower()
model_prod = None
model_staging = None
scaler = None

def load_scaler():
    local_paths = [
        "/app/models/scaler.pkl",
        "/app/models/scaler_dqn.pkl" if model_type == "dqn" else "/app/models/scaler_ppo.pkl",
        "models/scaler.pkl",
        "models/scaler_dqn.pkl" if model_type == "dqn" else "models/scaler_ppo.pkl",
    ]

    for path in local_paths:
        if os.path.exists(path):
            try:
                print(f"[+] Load scaler local: {path}")
                return joblib.load(path)
            except Exception as e:
                print(f"[-] Cannot load scaler {path}: {e}")

    print("[!] Không tìm thấy scaler local. Dùng identity scaler.")
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
    'serving_requests_total', 
    'Tổng số lượng request gọi dự đoán', 
    ['model']
)
INFERENCE_LATENCY = Histogram(
    'serving_latency_seconds', 
    'Độ trễ sinh ra quyết định của model', 
    ['model']
)
ACTION_CHOSEN = Counter(
    'serving_actions_total', 
    'Tần suất chọn hành động của model', 
    ['model', 'stage', 'action']
)

def load_models():
    global model_prod, model_staging, scaler

    registered_model_name = "SDN_DQN_Model" if model_type == "dqn" else "SDN_PPO_Model"
    model_uri = f"models:/{registered_model_name}/Production"

    scaler = load_scaler()

    try:
        if model_type == "dqn":
            agent = DQNAgent(
                state_dim=8,
                action_dim=5
            )

            agent.load("/app/models/dqn_model.pth")

            model_prod = agent.q_net
            model_prod.eval()

            print("[+] Loaded DQN from /app/models/dqn_model.pth")

        else:
            model_prod = mlflow.pytorch.load_model(model_uri)
            model_prod.eval()

    except Exception as e:
        print(f"[!] MODEL LOAD FAILED: {e}")
        model_prod = torch.nn.Linear(8, 5)
        model_prod.eval()
    try:
        model_staging = mlflow.pytorch.load_model(f"models:/{registered_model_name}/Staging")
        model_staging.eval()
        print(f"[+] Đã load {model_type.upper()} STAGING từ MLflow")
    except Exception as e:
        print(f"[-] Không có model Staging hoặc load lỗi: {e}")
        model_staging = None
    
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_type": model_type,
        "has_production_model": model_prod is not None,
        "has_staging_model": model_staging is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    INFERENCE_REQUESTS.labels(model=model_type).inc()
    
    data = request.get_json()
    state_raw = data.get('state')
    
    if not state_raw or len(state_raw) != 8:
        return jsonify({"error": f"Invalid state size. Expected 8."}), 400
    
    # Tiền xử lý
    if scaler:
        state_scaled = scaler.transform([state_raw])[0]
    else:
        state_scaled = np.array(state_raw, dtype=np.float32)
        
    state_tensor = torch.FloatTensor(state_scaled).unsqueeze(0)

    action_prod = 0
    action_staging = 0

    with torch.no_grad():
    # Dự đoán từ Production
        if model_prod:
            out_prod = model_prod(state_tensor)
            action_prod = int(out_prod[0].argmax().item() if isinstance(out_prod, tuple) else out_prod.argmax().item())
            ACTION_CHOSEN.labels(model=model_type, stage='production', action=str(action_prod)).inc()
        
        # Dự đoán từ Staging
        action_staging = action_prod 
        if model_staging:
            out_staging = model_staging(state_tensor)
            action_staging = int(out_staging[0].argmax().item() if isinstance(out_staging, tuple) else out_staging.argmax().item())
            ACTION_CHOSEN.labels(model=model_type, stage='staging', action=str(action_staging)).inc()   
            
    # Ghi nhận độ trễ (Latency)
    latency = time.time() - start_time
    INFERENCE_LATENCY.labels(model=model_type).observe(latency)
            
    return jsonify({
        "action": action_prod,          
        "action_staging": action_staging, 
        "model": model_type.upper()
    })

@app.route('/reload', methods=['POST'])
def reload_model_api():
    try:
        print("[*] Nhận lệnh Reload Model từ MLOps Pipeline...")
        load_models() # Hàm này bạn đã viết sẵn ở trên rồi
        return jsonify({"status": "success", "message": "Đã cập nhật Model Production mới nhất!"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    metrics_port = 9002 if model_type == "dqn" else 9003
    start_http_server(metrics_port, addr='0.0.0.0')
    print(f"[+] Prometheus Serving Metrics started on port {metrics_port}")
    
    app.run(host='0.0.0.0', port=args.port)
