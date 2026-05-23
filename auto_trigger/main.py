from flask import Flask, request, jsonify
import subprocess
import datetime

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    data = request.json or {}
    
    # Duyệt qua các cảnh báo nhận được từ Alertmanager
    if 'alerts' in data:
        for alert in data['alerts']:
            alert_name = alert['labels'].get('alertname')
            status = alert.get('status') # 'firing' (đang lỗi) hoặc 'resolved' (đã hết lỗi)

            if alert_name == "Low_Reward_Detected" and status == "firing":
                print(f"[{datetime.datetime.now()}] NHẬN BÁO ĐỘNG: Reward quá thấp! Đang kích hoạt Rollback...")
                
                # Gọi kịch bản Rollback thông qua bash
                try:
                    subprocess.run(
                        ["python3", "-c", "from mlops.mlflow_manager import rollback_model; rollback_model('SDN_DQN_Model')"], 
                        check=True
                    )
                    print("Đã Rollback thành công!")
                except Exception as e:
                    print(f"Lỗi khi Rollback: {e}")

    return jsonify({"status": "received"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)