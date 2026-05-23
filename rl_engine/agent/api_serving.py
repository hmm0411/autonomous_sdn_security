from flask import Flask, request, jsonify
import numpy as np
# Import agent của bạn ở đây
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() # Dùng get_json() an toàn hơn
    if not data or 'state' not in data:
        return jsonify({"error": "Invalid request"}), 400
        
    state = np.array(data['state'])
    # Gọi hàm dự đoán của agent ở đây
    # action = trained_agent.select_action(state) 
    return jsonify({"action": 1, "model": "dqn"})