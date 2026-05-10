import time
import pandas as pd
import os
import random
from datetime import datetime
from prometheus_client import start_http_server, Gauge, Counter
import requests 

LOG_FILE = "logs/live_metrics.csv"
SIGNAL_FILE = "logs/current_attack.txt"

ONOS_URL = "http://onos-controller:8181/onos/v1/statistics/delta/ports" # Hoặc đường dẫn API bạn đang dùng
AUTH = ('onos', 'rocks')



PROM_PACKET_RATE = Gauge('sdn_packet_rate', 'Real-time Packet Rate')
PROM_FLOW_COUNT = Gauge('sdn_flow_count', 'Real-time Flow Count')
PROM_THREAT_LEVEL = Gauge('sdn_threat_level', 'Threat Level (0=Normal, 1=Medium, 2=High)')
PROM_ACTION = Gauge('sdn_rl_action', 'Action taken by RL Agent')

INFERENCE_TIME = Gauge('rl_inference_time_ms', 'Inference latency in ms', ['agent'])
ACTION_COUNTER = Counter('rl_action_total', 'Total actions executed', ['agent', 'action_name'])

# Đảm bảo thư mục tồn tại
os.makedirs("logs", exist_ok=True)
    

# ---------------------------------------------------------
# 1. FIX LỖI: RESET TRẠNG THÁI NGAY KHI KHỞI ĐỘNG
# ---------------------------------------------------------
# Xóa bỏ dư âm của các lần test trước, ép về Normal Traffic
with open(SIGNAL_FILE, "w") as f:
    f.write("Normal Traffic")

if not os.path.exists(LOG_FILE):
    df_init = pd.DataFrame(columns=[
        "timestamp", "packet_rate", "flow_count", "level", "action_id", "action_name"
    ])
    df_init.to_csv(LOG_FILE, index=False)

def get_current_attack_state():
    if os.path.exists(SIGNAL_FILE):
        try:
            with open(SIGNAL_FILE, "r") as f:
                state = f.read().strip()
                return state if state else "Normal Traffic"
        except:
            pass
    return "Normal Traffic"

def main():
    print("[*] Starting Smooth Demo Agent...")
    
    # Khởi tạo giá trị hiện tại (Dùng để tính toán hiệu ứng tăng/giảm dần)
    current_packet_rate = 20.0
    current_flow_count = 50.0
    
    while True:
        try:
            current_attack = get_current_attack_state()
            
            # ---------------------------------------------------------
            # 2. XÁC ĐỊNH MỤC TIÊU VÀ HÀNH ĐỘNG THEO TỪNG LOẠI TẤN CÔNG
            # ---------------------------------------------------------
            # threat_num = 0
            if current_attack == "Normal Traffic":
                level, action_id, action_name = "NORMAL", 0, "No Action"
                threat_num = 0
                target_pr = random.randint(10, 40)
                target_fc = random.randint(40, 80)
                latency = random.uniform(1.2, 4.5)
                
            elif current_attack == "DDoS Flood":
                level, action_id, action_name = "HIGH_ATTACK (DDoS Flood)", 1, "Block (Drop Suspicious Flow)"
                target_pr = random.randint(4000, 5500)
                target_fc = random.randint(30, 50)
                latency = random.uniform(18.5, 42.1)

            elif current_attack == "Flow Table Overflow":
                level, action_id, action_name = "HIGH_ATTACK (Flow Table Overflow)", 2, "Rate Limit Bandwidth"
                target_pr = random.randint(2000, 2800)
                target_fc = random.randint(120, 160)
                latency = random.uniform(18.5, 42.1)
                
            elif current_attack == "Packet-In Flood":
                level, action_id, action_name = "MEDIUM_ATTACK (Packet-In Flood)", 1, "Block (Drop Suspicious Flow)"
                target_pr = random.randint(150, 300)
                target_fc = random.randint(80, 110)
                latency = random.uniform(12.3, 25.8)
                
            elif current_attack == "IP Spoofing":
                level, action_id, action_name = "HIGH_ATTACK (IP Spoofing)", 1, "Block (Drop Suspicious Flow)"
                target_pr = random.randint(1800, 2500)
                target_fc = random.randint(40, 60)
                latency = random.uniform(18.5, 42.1)
                
            elif current_attack == "Port Scanning":
                level, action_id, action_name = "HIGH_ATTACK (Port Scanning)", 3, "Redirect to Honeypot"
                target_pr = random.randint(3500, 4500)
                target_fc = random.randint(40, 60)
                latency = random.uniform(18.5, 42.1)

            else: # Fallback an toàn
                level, action_id, action_name = "NORMAL", 0, "No Action"
                target_pr = random.randint(10, 40)
                target_fc = random.randint(40, 80)
                latency = random.uniform(1.2, 4.5)

            # ---------------------------------------------------------
            # 3. THUẬT TOÁN LÀM MƯỢT (SMOOTHING) - TĂNG DẦN / GIẢM DẦN
            # ---------------------------------------------------------
            # Mỗi 2 giây, đường line trên biểu đồ chỉ tiến thêm 35% về phía target.
            # Giúp tạo cảm giác mạng đang "nóng lên" hoặc "hạ nhiệt" dần dần một cách rất thật.
            current_packet_rate += (target_pr - current_packet_rate) * 0.35
            current_flow_count += (target_fc - current_flow_count) * 0.35

            PROM_PACKET_RATE.set(int(current_packet_rate))
            PROM_FLOW_COUNT.set(int(current_flow_count))
            PROM_THREAT_LEVEL.set(threat_num)
            PROM_ACTION.set(action_id)
            INFERENCE_TIME.labels(agent='PPO').set(latency)
            ACTION_COUNTER.labels(agent='PPO', action_name=action_name).inc()
            
            # Ghi dữ liệu ra file CSV
            new_data = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "packet_rate": int(current_packet_rate),
                "flow_count": int(current_flow_count),
                "level": level,
                "action_id": action_id,
                "action_name": action_name
            }])
            
            df = pd.read_csv(LOG_FILE)
            df = pd.concat([df, new_data], ignore_index=True).tail(50)
            df.to_csv(LOG_FILE, index=False)
            
        except Exception as e:
            print(f"[!] Pipeline Error: {e}")
            
        time.sleep(2)

response = requests.get(ONOS_URL, auth=AUTH)

if __name__ == "__main__":
    # Start the Prometheus HTTP server
    start_http_server(8000)
    main()