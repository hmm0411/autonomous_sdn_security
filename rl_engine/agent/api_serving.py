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

from control_loop.serving import MODEL_TYPE

from control_loop.serving import MODEL_TYPE

app = Flask(__name__)

model_type = os.getenv("MODEL_TYPE", "dqn").strip().lower()
model_prod = None
model_staging = None
scaler = None

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
    
    # 1. Load Scaler
    try:
        scaler_uri = f"models:/{registered_model_name}/Production/preprocessor/scaler.pkl"
        scaler_path = download_artifacts(artifact_uri=scaler_uri)
        scaler = joblib.load(scaler_path)
        print("[+] Đã load Scaler thành công.")
    except Exception as e:
        print(f"[-] Không tìm thấy Scaler ({e}), sẽ dùng RAW Data.")
        scaler = None

    # 2. Load Model Production
    try:
        model_prod = mlflow.pytorch.load_model(f"models:/{registered_model_name}/Production")
        model_prod.eval()
        print(f"[+] Đã load {model_type.upper()} PRODUCTION")
    except Exception as e:
        print(f"[!] Lỗi khi load model Production: {e}")
        model_prod = None
        
    # 3. Load Model Staging
    try:
        model_staging = mlflow.pytorch.load_model(f"models:/{registered_model_name}/Staging")
        model_staging.eval()
        print(f"[+] Đã load {model_type.upper()} STAGING")
    except Exception:
        print("[-] Không có model Staging. Chạy chế độ Single Production.")
        model_staging = None

# Gọi hàm load ngay khi khởi động
load_models()
    
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model_type": MODEL_TYPE}

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