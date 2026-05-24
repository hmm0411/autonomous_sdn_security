import argparse
from flask import Flask, request, jsonify
import numpy as np
import os

# KHÔNG IMPORT FILE TRAIN TRỰC TIẾP, HÃY IMPORT CLASS AGENT TỪ FILE ĐỊNH NGHĨA AGENT (dqn_agent.py)
from rl_engine.agent.dqn_agent import DQNAgent 
from rl_engine.agent.ppo_agent import PPOAgent

app = Flask(__name__)

model_type = os.getenv("MODEL_TYPE", "dqn").strip().lower()

if model_type == "dqn":
    agent = DQNAgent(state_dim=8, action_dim=5)
    model_path = "/app/models/dqn_model.pth"
    print(f"DEBUG: Đang load DQN tại: {model_path}")
    agent.load(model_path)
    
elif model_type == "ppo":
    # PHẢI LÀ PPOAgent
    agent = PPOAgent(state_dim=8, action_dim=5) 
    model_path = "/app/models/ppo_model.pth"
    print(f"DEBUG: Đang load PPO tại: {model_path}")
    agent.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    state = np.array(data['state'])
    
    if len(state) != 8:
        return jsonify({"error": f"Invalid state size. Expected 8, got {len(state)}"}), 400
    
    result = agent.select_action(state)
    
    # Nếu select_action trả về tuple (action, extra_info), hãy lấy [0]
    if isinstance(result, tuple):
        action = result[0]
    else:
        action = result
        
    return jsonify({"action": int(action), "model": model_type})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    app.run(host='0.0.0.0', port=args.port)