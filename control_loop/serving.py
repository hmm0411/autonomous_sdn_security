import argparse
from flask import Flask, request, jsonify
import mlflow
import mlflow.pytorch
import torch
import joblib
import os
import time
import numpy as np
from prometheus_client import start_http_server, Gauge

# ===== CONFIG =====
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
)

# Lấy loại model từ Docker environment (dqn hoặc ppo)
MODEL_TYPE = os.getenv("MODEL_TYPE", "dqn").lower()

model = None
scaler = None

app = Flask(__name__)

model_used_g = Gauge("model_used", "Model selected", ["model_type"])

# ===== LOAD MODEL TỰ ĐỘNG CHỌN DQN HAY PPO =====
def load_models():
    global model, scaler

    # Xác định tên model trên MLflow Registry
    registered_model_name = "SDN_DQN_Model" if MODEL_TYPE == "dqn" else "SDN_PPO_Model"

    for attempt in range(10):
        try:
            # 1. Thử load từ MLflow Production
            model_uri = f"models:/{registered_model_name}/Production"
            print(f"[*] Đang thử load {MODEL_TYPE.upper()} từ MLflow: {model_uri} (Lần {attempt + 1}/10)")
            
            model = mlflow.pytorch.load_model(model_uri)
            model.eval()

            # Load scaler
            scaler_path = "/app/models/scaler.pkl"
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Không tìm thấy scaler tại {scaler_path}")
            scaler = joblib.load(scaler_path)

            print(f"[+] Load model {model_uri} THÀNH CÔNG!")
            return

        except Exception as e:
            print(f"[-] Lỗi load từ MLflow: {e}")
            
            # 2. Cơ chế sinh tồn (Fallback): Load file local nếu MLflow tạch
            try:
                print(f"[*] Thử load file local /app/models/{MODEL_TYPE}_model.pth...")
                if MODEL_TYPE == "dqn":
                    from rl_engine.agent.dqn_agent import DQNAgent
                    local_agent = DQNAgent(8, 5)
                    local_agent.load("/app/models/dqn_model.pth")
                    model = local_agent.q_net
                else:
                    from rl_engine.agent.ppo_agent import PPOAgent
                    local_agent = PPOAgent(8, 5) 
                    local_agent.load("/app/models/ppo_model.pth")
                    model = local_agent.model
                
                model.eval()
                scaler = joblib.load("/app/models/scaler.pkl")
                print(f"[+] Load model LOCAL {MODEL_TYPE.upper()} thành công làm fallback!")
                return
            except Exception as local_e:
                print(f"[-] Load local cũng thất bại: {local_e}")
            
            time.sleep(5)

    raise Exception(f"Critical: Không thể load {MODEL_TYPE.upper()} model sau 10 lần thử.")

# ===== HEALTH =====
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model_type": MODEL_TYPE}

# ===== PREDICT =====
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "state" not in data:
            return jsonify({"error": "Thiếu trường 'state' trong JSON"}), 400
            
        state_raw = data["state"]

        if len(state_raw) != 8:
            return jsonify({"error": "invalid state size"}), 400

        # Transform và chuyển sang Tensor
        state_scaled = scaler.transform([state_raw])
        state_tensor = torch.FloatTensor(state_scaled)

        with torch.no_grad():
            output = model(state_tensor)
            
            # DQN output là Q-values -> argmax. PPO output có thể là action_probs -> tùy logic của bạn
            # Để an toàn cho cả 2, giả định output trả ra list logits hoặc Q-values
            if isinstance(output, tuple):
                action = output[0].argmax().item() # Nếu model PPO trả về tuple (action, log_prob, v.v..)
            else:
                action = output.argmax().item()

        model_used_g.labels(model_type=MODEL_TYPE).set(1)

        return jsonify({
            "action": int(action),
            "model": MODEL_TYPE
        })

    except Exception as e:
        print("Predict error:", e)
        return jsonify({"action": 0, "model": "fallback"})

# ===== RELOAD =====
@app.route("/reload", methods=["POST"])
def reload_model():
    try:
        load_models()
        return {"status": "reloaded", "model_type": MODEL_TYPE}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

# ===== RUN =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Nhận port từ docker-compose command
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Phân luồng Prometheus Port cho DQN (9002) và PPO (9003) khớp với prometheus.yml
    prom_port = 9002 if MODEL_TYPE == "dqn" else 9003
    start_http_server(prom_port)
    print(f"[+] Started Prometheus metrics on port {prom_port}")
    
    load_models()
    
    # Chạy Flask trên port lấy từ argparse
    app.run(host="0.0.0.0", port=args.port)