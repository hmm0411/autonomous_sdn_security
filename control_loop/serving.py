from flask import Flask, request, jsonify
import mlflow.pytorch
import torch
import joblib
from prometheus_client import start_http_server, Gauge
from mlops.mlflow_manager import get_best_model

model = None
scaler = None

app = Flask(__name__)

model_used_g = Gauge("model_used", "Model selected", ["model"])

# ===== LOAD MODEL =====
def load_models():
    global model, scaler

    best_model, _ = get_best_model()
    model_uri = f"models:/{best_model.name}/{best_model.version}"

    model = mlflow.pytorch.load_model(model_uri)
    scaler = joblib.load("models/scaler.pkl")

    print(f"[+] Loaded model: {model_uri}")

# ===== PREDICT =====
@app.route("/predict", methods=["POST"])
def predict():
    try:
        state = request.json["state"]

        state = scaler.transform([state])
        state = torch.tensor(state).float()

        with torch.no_grad():
            action = model(state).argmax().item()

        model_used_g.labels(model="AUTO").set(1)

        return jsonify({
            "action": int(action),
            "model": "AUTO"
        })

    except Exception as e:
        print("Predict error:", e)
        return jsonify({"action": 0, "model": "fallback"})

# ===== RELOAD =====
@app.route("/reload", methods=["POST"])
def reload_model():
    load_models()
    return jsonify({"status": "reloaded"})

# ===== RUN =====
if __name__ == "__main__":
    start_http_server(9002)
    load_models()
    app.run(host="0.0.0.0", port=8000)