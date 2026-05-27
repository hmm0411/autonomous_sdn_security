import os
import logging
import time
import threading
import subprocess
import requests
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest
from kubernetes import client, config
from mlflow.tracking import MlflowClient

# 1. Cấu hình Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 2. Khởi tạo K8s & MLflow Client
try:
    config.load_incluster_config()
    job_api = client.BatchV1Api()
    mlflow_client = MlflowClient()
    logger.info("[+] Đã load cấu hình Kubernetes thành công.")
except Exception as e:
    logger.error(f"[-] Lỗi load cấu hình K8s: {e}")

NAMESPACE = "sdn-security"

# Prometheus metrics
alerts_received = Counter('alerts_received_total', 'Total alerts received', ['severity'])
webhook_processing_time = Histogram('webhook_processing_seconds', 'Time to process webhook')

def create_training_job(model_name):
    """Tạo một Kubernetes Job để retrain model"""
    job_name = f"retrain-{model_name.lower()}-{int(time.time())}"
    
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name},
        "spec": {
            "template": {
                "spec": {
                    "restartPolicy": "OnFailure",
                    "containers": [{
                        "name": "trainer",
                        "image": "hmm0411/sdn-rl-agent:latest",
                        "command": ["python", "-m", f"rl_engine.agent.train_{model_name.lower()}"],
                        "env": [
                            {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow.sdn-security.svc.cluster.local:5000"},
                            {"name": "PYTHONPATH", "value": "/app"}
                        ],
                        "volumeMounts": [{"name": "app-vol", "mountPath": "/app"}]
                    }],
                    "volumes": [{"name": "app-vol", "hostPath": {"path": "/home/g23520964/autonomous_sdn_security"}}]
                }
            }
        }
    }
    try:
        job_api.create_namespaced_job(namespace=NAMESPACE, body=job_manifest)
        logger.info(f"[*] Đã tạo Job training thành công: {job_name}")
    except Exception as e:
        logger.error(f"[-] Lỗi tạo Job: {e}")

def run_promote_pipeline():
    """Logic thăng cấp model"""
    logger.info("[*] Kích hoạt Automated Promote Pipeline...")
    try:
        # Gọi script promote.py (đảm bảo bạn đã gộp logic hoặc trỏ đúng đường dẫn)
        subprocess.run(["python", "/app/mlops/promote.py"], check=True)
        
        # Reload API Serving
        requests.post("http://rl-serving-dqn:8000/reload", timeout=5)
        requests.post("http://rl-serving-ppo:8001/reload", timeout=5)
        logger.info("[+] Promote hoàn tất!")
    except Exception as e:
        logger.error(f"[-] Pipeline Promote thất bại: {e}")

@app.route('/alert-webhook', methods=['POST'])
@webhook_processing_time.time()
def handle_alert():
    try:
        data = request.json
        if not data or 'alerts' not in data:
            return jsonify({'status': 'ok'}), 200
            
        for alert in data['alerts']:
            alert_name = alert.get('labels', {}).get('alertname', '')
            status = alert.get('status')
            
            if status == "firing":
                if alert_name in ["Low_Reward_Detected", "PPO_Reward_Drop"]:
                    model = "DQN" if "DQN" in alert_name else "PPO"
                    logger.warning(f"Phát hiện {alert_name}. Trigger Auto-Retrain!")
                    threading.Thread(target=create_training_job, args=(model,)).start()
                
                elif alert_name == "Staging_Outperforms_Production":
                    logger.warning("Shadow Model thắng! Trigger Auto-Promote!")
                    threading.Thread(target=run_promote_pipeline).start()
                    
        return jsonify({'status': 'triggered'}), 202
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain; version=0.0.4"}

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port)