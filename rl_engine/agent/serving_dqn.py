import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from rl_engine.config import STATE_DIM, ACTION_DIM
from rl_engine.agent.dqn_agent import DQNAgent # Import cấu trúc Agent của bạn

app = Flask(__name__)

# Cách an toàn nhất cho Demo: Load trực tiếp file .pth từ ổ cứng
try:
    print("[*] Loading DQN Model...")
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    checkpoint = torch.load("models/dqn_model.pth", map_location=torch.device('cpu'))
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])
    agent.q_net.eval()
    print("[+] DQN Model Loaded Successfully!")
except Exception as e:
    print(f"[-] Lỗi load model DQN: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Lấy state từ request, thêm chiều batch (1, state_dim)
        state_data = request.json["state"]
        state_tensor = torch.tensor(state_data).float().unsqueeze(0)
        
        # Đưa vào mô hình dự đoán hành động
        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
            action = q_values.argmax().item()
            
        return jsonify({"action": int(action)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # DQN chạy đúng port 9000
    app.run(host="0.0.0.0", port=9000)