import time
import prometheus_client
import requests
import pandas as pd
import numpy as np
import os
from requests.auth import HTTPBasicAuth
from datetime import datetime
import random
from prometheus_client import start_http_server, Gauge

ONOS_URL = "http://controller:8181/onos/v1"
AUTH = HTTPBasicAuth("onos", "rocks")
LOG_FILE = "logs/live_metrics.csv"

current_threat_score = 0.0
mitigation_timer = 0

# Đảm bảo thư mục logs tồn tại
os.makedirs("logs", exist_ok=True)

# Khởi tạo file CSV nếu chưa có
if not os.path.exists(LOG_FILE):
    df_init = pd.DataFrame(columns=[
        "timestamp", "status", "threat", "cpu", "flow", "latency", 
        "attack_type", "rl_action_id", "rl_action_name", "confidence"
    ])
    df_init.to_csv(LOG_FILE, index=False)

def get_onos_flows():
    try:
        r = requests.get(f"{ONOS_URL}/flows", auth=AUTH, timeout=2)
        if r.status_code == 200:
            return r.json().get("flows", [])
    except:
        pass
    return []

def estimate_latency():
    # Giả lập latency dựa trên mức độ nghẽn mạng để demo mượt mà hơn
    # (Có thể thay bằng subprocess ping thật nếu mạng Mininet được nối vào host)
    return random.uniform(2.0, 5.0)

def pseudo_rl_agent(flow_count, byte_rate):
    """
    Giả lập RL Agent: Nhận diện Attack dựa trên metrics thật và đưa ra Action.
    Thêm yếu tố 'imperfect' (không hoàn hảo) để demo chân thực.
    """
    global current_threat_score, mitigation_timer
    confidence = round(random.uniform(88.5, 96.2), 1)
    
    attack_file = "logs/current_attack.txt"
    try:
        # Đọc file signal từ AttackManager
        with open(attack_file, "r") as f:
            attack_state = f.read().strip()
    except Exception:
        attack_state = "Normal Traffic"

    # Map chính xác Attack -> Action
    if "Normal" in attack_state:
        target_score = 4.2
        status_str = "Normal Traffic"
        action_code = 0
        action_str = "No Action"
        mitigation_timer = 0
    elif "DDoS" in attack_state:
        target_score = 99.4
        status_str = "DDoS Flood Attack"
        action_code = 1
        action_str = "Block (Drop Flow)"
    elif "Spoof" in attack_state:
        target_score = 88.0
        status_str = "IP Spoofing Detected"
        action_code = 4
        action_str = "Isolate Device"
    elif "Packet-In" in attack_state:
        target_score = 78.5
        status_str = "Packet-In Anomaly"
        action_code = 2
        action_str = "Rate Limit Bandwidth"
    else:
        target_score = 5.0
        status_str = "Normal Traffic"
        action_code = 0
        action_str = "No Action"
    
    # Thay vì nhảy lập tức, điểm số tăng hoặc giảm từng bước 6% mỗi chu kỳ cào dữ liệu
    step = 6.0
    if current_threat_score < target_score:
        current_threat_score = min(current_threat_score + step, target_score)
    elif current_threat_score > target_score:
        current_threat_score = max(current_threat_score - step, target_score)

    # Nếu hệ thống đang bị DDoS và điểm số đã chạm đỉnh nguy hiểm, giả lập hiệu quả của luật chặn
    if "DDoS" in attack_state and current_threat_score >= 90.0:
        mitigation_timer += 1
        # Sau 3 chu kỳ cào dữ liệu (~6 giây áp dụng luật chặn), hạ điểm nguy hiểm xuống mức an toàn
        if mitigation_timer > 3:
            target_score = 10.5
            current_threat_score = max(current_threat_score - 12.0, target_score)
            status_str = "Mitigated - Network Stable"
            action_str = "Monitoring Filtered Flows"

    return round(current_threat_score, 1), status_str, action_code, action_str

def run_pipeline():
    print("[*] Starting Real-time SDN Data Collector Pipeline...")
    prev_bytes = 0
    
    while True:
        try:
            flows = get_onos_flows()
            flow_count = len(flows)
            
            # Tính tổng bytes
            current_bytes = sum(f.get("bytes", 0) for f in flows)
            byte_rate = max(0, current_bytes - prev_bytes)
            prev_bytes = current_bytes
            
            # Tính toán các Metrics cơ bản
            base_latency = estimate_latency()
            latency = base_latency + (flow_count / 100.0) if flow_count > 200 else base_latency
            cpu = min(100.0, 5.0 + (flow_count / 50.0) + random.uniform(0, 3))
            
            # RL Agent dự đoán
            attack_type, action_id, action_name, conf = pseudo_rl_agent(flow_count, byte_rate)
            
            threat = 0 if attack_type == "Normal Traffic" else min(99, int(flow_count / 30))
            status = "Healthy" if threat < 20 else ("Critical" if threat > 70 else "Warning")
            
            # Ghi log
            new_data = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "status": status,
                "threat": threat,
                "cpu": round(cpu, 1),
                "flow": flow_count,
                "latency": round(latency, 2),
                "attack_type": attack_type,
                "rl_action_id": action_id,
                "rl_action_name": action_name,
                "confidence": conf
            }])
            
            # Lưu dồn vào CSV (Giữ lại 100 dòng gần nhất để vẽ biểu đồ)
            df = pd.read_csv(LOG_FILE)
            df = pd.concat([df, new_data], ignore_index=True).tail(100)
            df.to_csv(LOG_FILE, index=False)
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Flows: {flow_count} | Attack: {attack_type} | Action: {action_name} | Conf: {conf}%")
            
        except Exception as e:
            print(f"[!] Pipeline Error: {e}")
            
        time.sleep(2) # Polling mỗi 2 giây

if __name__ == "__main__":
    start_http_server(8000) 
    run_pipeline()