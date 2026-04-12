import os
import subprocess
import sys
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def start_mlflow():
    """Khởi chạy MLflow Tracking Server"""
    logger.info("Đang khởi động MLflow Server trên port 5000...")
    # Lệnh này tương đương với command trong docker-compose
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "/mlruns",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    subprocess.run(cmd)

def start_auto_trigger():
    """Khởi chạy Webhook Server để nhận Alert từ Prometheus/Alertmanager"""
    logger.info("Đang khởi động Auto-retrain Trigger trên port 5001...")
    # Import ở đây để tránh lỗi nếu chỉ muốn chạy MLflow
    from auto_trigger import app 
    app.run(host='0.0.0.0', port=5001)

if __name__ == "__main__":
    # Lấy mode từ biến môi trường (set trong docker-compose)
    mode = os.getenv("SERVICE_MODE", "trigger").lower()

    if mode == "mlflow":
        start_mlflow()
    elif mode == "trigger":
        start_auto_trigger()
    else:
        logger.error(f"❌ Mode '{mode}' không hợp lệ. Vui lòng chọn 'mlflow' hoặc 'trigger'.")
        sys.exit(1)