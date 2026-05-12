from flask import Flask, request, jsonify
import mlflow.pytorch
import torch
import json
from prometheus_client import start_http_server, Gauge

app = Flask(__name__)

CONFIG_PATH = "config/model_config.json"

# ===== METRICS =====
model_used_g = Gauge("model_used", "Model selected", ["model"])

# ===== LOAD MODEL =====
def load_models():
    global dqn_model, ppo_model

    print("Loading models from MLflow...")

    dqn_model = mlflow.pytorch.load_model("models:/SDN_DQN_Model/Production")
    ppo_model = mlflow.pytorch.load_model("models:/SDN_PPO_Model/Production")

    dqn_model.eval()
    ppo_model.eval()

load_models()

# ===== CONFIG =====
def get_active_model():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)["active_model"]
    except:
        return "AUTO"   # default auto mode

# ===== PREDICT =====
@app.route("/predict", methods=["POST"])
def predict():
    try:
        state = request.json["state"]
        state = torch.tensor(state).float().unsqueeze(0)

        mode = get_active_model()

        with torch.no_grad():

            # ===== AUTO MODE =====
            if mode == "AUTO":
                q_dqn = dqn_model(state)
                q_ppo = ppo_model(state)

                if q_dqn.max().item() > q_ppo.max().item():
                    action = q_dqn.argmax().item()
                    model_used = "DQN"
                else:
                    action = q_ppo.argmax().item()
                    model_used = "PPO"

            # ===== FORCE MODE =====
            elif mode == "DQN":
                action = dqn_model(state).argmax().item()
                model_used = "DQN"

            else:
                action = ppo_model(state).argmax().item()
                model_used = "PPO"

        model_used_g.labels(model=model_used).set(1)

        return jsonify({
            "action": int(action),
            "model": model_used
        })

    except Exception as e:
        print("Predict error:", e)

        # ===== FAILOVER =====
        try:
            action = ppo_model(state).argmax().item()
            return jsonify({"action": int(action), "model": "PPO"})
        except:
            return jsonify({"action": 0, "model": "fallback"})

# ===== RELOAD =====
@app.route("/reload", methods=["POST"])
def reload_model():
    load_models()
    return jsonify({"status": "reloaded"})

# ===== RUN =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)