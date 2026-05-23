from flask import Flask, request, jsonify
import numpy as np
import argparse
import os
from rl_engine.agent.train_dqn import DQNAgent
from rl_engine.agent.train_ppo import PPOAgent

# Khởi tạo các agent
dqn_agent = DQNAgent(9, 5)  
ppo_agent = PPOAgent()

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

app = Flask(__name__)

# Logic chọn model dựa trên biến môi trường MODEL_TYPE
model_type = os.getenv("MODEL_TYPE", "dqn")

if model_type == "dqn":
    agent = DQNAgent(9, 5)
    # Tải model đã train từ MLflow/MinIO
elif model_type == "ppo":
    agent = PPOAgent()
    # Tải model PPO
    
@app.route('/predict', methods=['POST'])
def predict():
    # ... logic lấy model tương ứng ...
    return jsonify({"action": 1, "model": model_type})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    app.run(host='0.0.0.0', port=args.port)