import time
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST
from rl_engine.config import STATE_DIM, ACTION_DIM
from rl_engine.agent.ppo_agent import PPOAgent

app = Flask(__name__)

# Khai báo Prometheus Metrics
INFERENCE_TIME = Gauge('rl_inference_time_ms', 'Inference latency in ms', ['agent'])
ACTION_COUNTER = Counter('rl_action_total', 'Total actions executed', ['agent', 'action_name'])
ACTION_DICT = {0: "Normal", 1: "Block", 2: "Rate Limit", 3: "Redirect"}

try:
    print("[*] Loading PPO Model...")
    agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    checkpoint = torch.load("models/ppo_model.pth", map_location=torch.device('cpu'))
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.model.eval()
    print("[+] PPO Model Loaded Successfully!")
except Exception as e:
    print(f"[-] Lỗi load model PPO: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_time = time.time()
        state_data = request.json["state"]
        state_tensor = torch.tensor(state_data).float().unsqueeze(0)
        
        with torch.no_grad():
            action_probs, _ = agent.model(state_tensor)
            action = int(action_probs.argmax().item())
            
        # Ghi nhận thời gian và hành động
        latency = (time.time() - start_time) * 1000
        INFERENCE_TIME.labels(agent='PPO').set(latency)
        ACTION_COUNTER.labels(agent='PPO', action_name=ACTION_DICT.get(action, "Unknown")).inc()
            
        return jsonify({"action": action})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001)