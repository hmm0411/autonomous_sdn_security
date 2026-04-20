from flask import Flask, request
import requests
import os

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def handle_alert():
    data = request.json
    if data['status'] == 'firing':
        print("Phát hiện hiệu năng thấp! Đang kích hoạt huấn luyện lại...")
        # Gọi sang container rl-agent hoặc thực thi script train_ppo.py
        # os.system("python /app/rl_engine/agent/train_ppo.py") 
    return "Alert received", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)