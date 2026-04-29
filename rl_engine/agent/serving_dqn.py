import mlflow.pytorch
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

import mlflow
mlflow.set_tracking_uri("http://mlflow:5000")

model = mlflow.pytorch.load_model("models:/SDN_DQN_Model/Production")
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    state = torch.tensor(request.json["state"]).float().unsqueeze(0)
    action = model(state).argmax().item()
    return jsonify({"action": int(action)})

app.run(host="0.0.0.0", port=9000)