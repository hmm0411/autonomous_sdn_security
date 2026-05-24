# auto-trigger/main.py (Trích đoạn xử lý Webhook từ Alertmanager)
from flask import Flask, request
import subprocess
import mlflow
from mlflow.tracking import MlflowClient

app = Flask(__name__)
client = MlflowClient()

@app.route('/alert-webhook', methods=['POST'])
def handle_alert():
    alert_data = request.get_json()
    status = alert_data.get('status') # 'firing' hoặc 'resolved'
    alert_name = alert_data.get('alerts')[0].get('labels').get('alertname')

    if status == 'firing' and alert_name == 'Low_Reward_Anomalies':
        print("[!] Phát hiện Reward hệ thống suy giảm nghiêm trọng. Kích hoạt quy trình Retrain tự động...")
        
        # Gọi Kubernetes Job để chạy script huấn luyện lại ngầm
        # Hoặc dùng lệnh chạy Docker nếu bạn chưa kịp chuyển sang K8s
        subprocess.run(["docker", "exec", "rl-agent-dqn", "python", "rl_engine/agent/train_dqn.py"])
        
        # Sau khi train xong, chạy quy trình Đánh giá (Evaluation) tự động
        evaluate_and_promote_model()
        
    return {"status": "processed"}, 200

def evaluate_and_promote_model():
    print("[*] Đang đánh giá hiệu năng mô hình mới huấn luyện...")
    # Lấy thông tin phiên bản Production hiện tại và phiên bản mới nhất trong Staging
    model_name = "SDN_DQN_Model"
    
    prod_version = client.get_latest_versions(model_name, stages=["Production"])[0]
    staging_version = client.get_latest_versions(model_name, stages=["Staging"])[0]
    
    # Đọc chỉ số metric từ MLflow của 2 bản để so sánh
    prod_reward = client.get_metric_history(prod_version.run_id, "final_mean_reward")[-1].value
    staging_reward = client.get_metric_history(staging_version.run_id, "final_mean_reward")[-1].value
    
    print(f"-> Reward của bản Production hiện tại: {prod_reward}")
    print(f"-> Reward của bản Staging mới huấn luyện: {staging_reward}")
    
    if staging_reward > prod_reward:
        print("[+] Mô hình mới tốt hơn! Tiến hành TỰ ĐỘNG PROMOTE lên Production.")
        # Lên đời model mới
        client.transition_model_version_stage(name=model_name, version=staging_version.version, stage="Production", archive_existing_versions=True)
        # Gọi lệnh reload API Serving để nhận bộ não mới ngay lập tức mà không downtime
        requests.post("http://rl-serving-dqn:8000/reload")
    else:
        print("[-] Mô hình mới không đạt kỳ vọng hoặc tệ hơn bản cũ. Giữ nguyên bản cũ (Tự động Rollback/Chặn đẩy dịch vụ).")
        # Hạ mác hoặc lưu kho lưu trữ bản lỗi này lại
        client.transition_model_version_stage(name=model_name, version=staging_version.version, stage="Archived")