import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from rl_engine.config import STATE_DIM, ACTION_DIM
from rl_engine.agent.ppo_agent import PPOAgent # Import cấu trúc Agent của bạn

app = Flask(__name__)

# Cách an toàn nhất cho Demo: Load trực tiếp file .pth từ ổ cứng
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
        # Lấy state từ request, thêm chiều batch (1, state_dim)
        state_data = request.json["state"]
        state_tensor = torch.tensor(state_data).float().unsqueeze(0)
        
        # Đưa vào mô hình dự đoán hành động
        with torch.no_grad():
            # Tùy thuộc vào code PPO của bạn, thường nó trả về (action_probs, state_values)
            action_probs, _ = agent.model(state_tensor)
            action = action_probs.argmax().item()
            
        return jsonify({"action": int(action)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Đã sửa lại đúng port 9001 cho PPO
    app.run(host="0.0.0.0", port=9001)