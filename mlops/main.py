import os
import logging
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest
import subprocess
import threading
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Prometheus metrics
alerts_received = Counter('alerts_received_total', 'Total alerts received', ['severity'])
webhook_processing_time = Histogram('webhook_processing_seconds', 'Time to process webhook')

def run_retrain_pipeline():
    """Chạy ngầm pipeline để không block Webhook response"""
    logger.info("[*] Kích hoạt Automated Data Pipeline...")
    try:
        logger.info("-> Chạy Data Processor...")
        subprocess.run(["python", "/app/rl_engine/data_processor.py"], check=True)
        
        logger.info("-> Bắt đầu Retrain Model...")
        # Ở đây bạn có thể cấu hình gọi dqn hoặc ppo tùy logic
        subprocess.run(["python", "/app/rl_engine/agent/train_dqn.py"], check=True)
        
        logger.info("[+] Retrain hoàn tất. Model mới đã đẩy lên MLflow (Staging).")
    except Exception as e:
        logger.error(f"[-] Pipeline Retrain thất bại: {e}")

def run_promote_pipeline():
    """Chạy ngầm pipeline thăng cấp model"""
    logger.info("[*] Kích hoạt Automated Promote Pipeline...")
    try:
        # Bước 1: Gọi script thăng cấp trên MLflow
        result = subprocess.run(["python", "/app/experiments/evaluate.py"], capture_output=True, text=True)
        logger.info(f"-> Kết quả đánh giá: {result.stdout}")
        logger.info("-> Chuyển trạng thái model trên MLflow Registry...")
        subprocess.run(["python", "/app/mlops/promote.py"], check=True)
        
        # Bước 2: Bắn tín hiệu để API Serving reload lại model mới
        logger.info("-> Bắn tín hiệu Reload cho API Serving...")
        try:
            requests.post("http://rl-serving-dqn:8000/reload", timeout=5)
            requests.post("http://rl-serving-ppo:8001/reload", timeout=5)
        except requests.exceptions.RequestException as req_err:
            logger.warning(f"Lỗi gọi Reload API (Có thể container chưa bật): {req_err}")
            
        logger.info("[+] Promote hoàn tất. Model mới đã làm chủ mạng!")
    except Exception as e:
        logger.error(f"[-] Pipeline Promote thất bại: {e}")

@app.route('/webhook', methods=['POST'])
@webhook_processing_time.time()
def handle_alert():
    try:
        data = request.json
        if not data or 'alerts' not in data:
            return jsonify({'status': 'ok'}), 200
            
        for alert in data['alerts']:
            alert_name = alert.get('labels', {}).get('alertname', '')
            status = alert.get('status')
            
            # 1. TRIGGER AUTO-RETRAIN (Khi điểm thấp)
            if alert_name in ["Low_Reward_Detected", "PPO_Reward_Drop"] and status == "firing":
                logger.warning(f"Phát hiện {alert_name}. Trigger Auto-Retrain!")
                threading.Thread(target=run_retrain_pipeline).start()
                
            # 2. TRIGGER AUTO-PROMOTE (Khi Shadow Model làm tốt hơn)
            elif alert_name == "Staging_Outperforms_Production" and status == "firing":
                logger.warning("Shadow Model thắng Production! Trigger Auto-Promote!")
                threading.Thread(target=run_promote_pipeline).start()
                
        return jsonify({'status': 'triggered'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest()

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'MLOps Auto-Trigger',
        'endpoints': ['/webhook', '/health', '/metrics']
    }), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    host = os.getenv('HOST', '0.0.0.0')
    app.run(host=host, port=port)