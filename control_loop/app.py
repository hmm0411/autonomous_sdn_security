from flask import Flask, request, jsonify
import mlflow
import mlflow.pytorch
import torch

app = Flask(__name__)

mlflow.set_tracking_uri("http://mlflow:5000")

model = mlflow.pytorch.load_model("models:/SDN_DQN_Model/Production")
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        state = request.json["state"]
        state = torch.tensor(state).float().unsqueeze(0)

        with torch.no_grad():
            action = model(state).argmax().item()

        return jsonify({"action": int(action)})

    except Exception as e:
        print("Predict error:", e)
        return jsonify({"action": 0})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)