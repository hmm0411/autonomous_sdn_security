import os
import logging
import time
import threading
import subprocess
import requests
import mlflow
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest
from kubernetes import client, config
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://mlflow.sdn-security.svc.cluster.local:5000")
# 1. Cấu hình Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 2. Khởi tạo K8s & MLflow Client
def get_job_api():
    """
    Load lại Kubernetes in-cluster config mỗi lần cần gọi API.
    Tránh lỗi client/token cũ sau khi rollout pod hoặc đổi ServiceAccount.
    """
    config.load_incluster_config()
    return client.BatchV1Api()


try:
    job_api = get_job_api()
    mlflow_client = MlflowClient()
    logger.info("[+] Đã load cấu hình Kubernetes thành công.")
except Exception as e:
    job_api = None
    logger.exception(f"[-] Lỗi load cấu hình K8s: {e}")

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
        "metadata": {
            "name": job_name,
            "namespace": NAMESPACE,
            "labels": {
                "app": "retrain-job",
                "model": model_name.lower()
            }
        },
        "spec": {
            "ttlSecondsAfterFinished": 3600, # Tự động xóa Job sau 1 giờ
            "template": {
                "spec": {
                    "restartPolicy": "OnFailure",
                    "containers": [{
                        "name": "trainer",
                        "image": "hmm0411/sdn-rl-agent:latest",
                        "command": ["python", "-m", f"rl_engine.agent.train_{model_name.lower()}"],
                        "env": [
                            {"name": "PYTHONPATH", "value": "/app"},
                            {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow.sdn-security.svc.cluster.local:5000"},
                            {"name": "MLFLOW_S3_ENDPOINT_URL", "value": "http://s3:9005"},
                            {"name": "MLFLOW_S3_IGNORE_TLS", "value": "true"},
                            {"name": "AWS_DEFAULT_REGION", "value": "us-east-1"},
                            {
                                "name": "AWS_ACCESS_KEY_ID",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "minio-credentials",
                                        "key": "ACCESS_KEY"
                                    }
                                }
                            },
                            {
                                "name": "AWS_SECRET_ACCESS_KEY",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "minio-credentials",
                                        "key": "SECRET_KEY"
                                    }
                                }
                            }
                        ],
                        "volumeMounts": [{"name": "app-vol", "mountPath": "/app"}]
                    }],
                    "volumes": [{"name": "app-vol", "hostPath": {"path": "/home/g23520964/autonomous_sdn_security"}}]
                }
            }
        }
    }
    try:
        job_api = get_job_api()
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
        data = request.get_json(silent=True)

        if not data or "alerts" not in data:
            return jsonify({"status": "ok", "message": "no alerts"}), 200

        triggered_actions = []

        for alert in data["alerts"]:
            labels = alert.get("labels", {})
            alert_name = labels.get("alertname", "")
            severity = labels.get("severity", "unknown")
            status = alert.get("status", "unknown")

            alerts_received.labels(severity=severity).inc()

            logger.info(
                f"[Alertmanager] alert={alert_name}, severity={severity}, status={status}"
            )

            if status != "firing":
                continue

            # 1. Runtime security alerts: chỉ ghi nhận để demo, không retrain liên tục
            if alert_name in [
                "SDN_Critical_Attack_Detected",
                "SDN_Warning_Attack_Detected",
                "SDN_Block_Action_Triggered",
                "SDN_Isolate_Action_Triggered",
            ]:
                logger.warning(
                    f"Nhận cảnh báo runtime security: {alert_name}. "
                    "Dashboard/Alertmanager đã ghi nhận sự kiện."
                )
                triggered_actions.append(f"logged:{alert_name}")

            # 2. Model quality alerts: trigger retrain
            elif alert_name in ["Low_Reward_Detected", "DQN_Reward_Drop"]:
                logger.warning(f"Phát hiện {alert_name}. Trigger Auto-Retrain DQN!")
                job_name = create_training_job("DQN")

                if job_name:
                    triggered_actions.append(f"retrain:DQN:{job_name}")
                else:
                    triggered_actions.append("retrain:DQN:failed")

            elif alert_name == "PPO_Reward_Drop":
                logger.warning("Phát hiện PPO_Reward_Drop. Trigger Auto-Retrain PPO!")
                job_name = create_training_job("PPO")

                if job_name:
                    triggered_actions.append(f"retrain:PPO:{job_name}")
                else:
                    triggered_actions.append("retrain:PPO:failed")

            # 3. Promote model
            elif alert_name == "Staging_Outperforms_Production":
                logger.warning("Shadow Model thắng! Trigger Auto-Promote!")
                threading.Thread(
                    target=run_promote_pipeline,
                    daemon=True
                ).start()
                triggered_actions.append("promote")

            else:
                logger.info(f"Alert chưa có action mapping: {alert_name}")
                triggered_actions.append(f"ignored:{alert_name}")

        return jsonify({
            "status": "triggered",
            "actions": triggered_actions
        }), 202

    except Exception as e:
        logger.exception(f"Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain; version=0.0.4"}

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port)