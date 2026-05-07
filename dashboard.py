import time
import pandas as pd
import os
import random
from datetime import datetime
from prometheus_client import start_http_server, Gauge

LOG_FILE = "logs/live_metrics.csv"
SIGNAL_FILE = "logs/current_attack.txt"

os.makedirs("logs", exist_ok=True)
if not os.path.exists(LOG_FILE):
    df_init = pd.DataFrame(columns=[
        "timestamp", "packet_rate", "flow_count", "level", "action_id", "action_name"
    ])
    df_init.to_csv(LOG_FILE, index=False)

# ==========================================
# KHỞI TẠO METRICS CHO GRAFANA (PROMETHEUS)
# ==========================================
PROM_PACKET_RATE = Gauge('sdn_packet_rate', 'Real-time Packet Rate')
PROM_FLOW_COUNT = Gauge('sdn_flow_count', 'Real-time Flow Count')
PROM_THREAT_LEVEL = Gauge('sdn_threat_level', 'Threat Level (0=Normal, 1=Medium, 2=High)')
PROM_ACTION = Gauge('sdn_rl_action', 'Action taken by RL Agent')

def get_current_attack_state():
    if os.path.exists(SIGNAL_FILE):
        try:
            with open(SIGNAL_FILE, "r") as f:
                return f.read().strip()
        except:
            pass
    return "Normal Traffic"

def main():
    print("[*] Starting Deterministic Demo Agent...")
    
    # Mở port 8000 để Prometheus vào cào dữ liệu
    start_http_server(8000)
    print("[*] Prometheus metrics server started on port 8000")
    
    while True:
        try:
            current_attack = get_current_attack_state()
            
            # Kịch bản thông số (Giống hệt cấu hình cũ)
            if current_attack == "Normal Traffic":
                level, action_id, action_name = "NORMAL", 0, "No Action"
                packet_rate = random.randint(10, 50)
                flow_count = random.randint(50, 80)
                threat_num = 0
                
            elif current_attack == "DDoS Flood":
                level, action_id, action_name = "HIGH_ATTACK (DDoS Flood)", 1, "Block (Drop Suspicious Flow)"
                packet_rate = random.randint(3000, 5000)
                flow_count = random.randint(30, 50)
                threat_num = 2
                
            elif current_attack == "Flow Table Overflow":
                level, action_id, action_name = "HIGH_ATTACK (Flow Table Overflow)", 2, "Rate Limit Bandwidth"
                packet_rate = random.randint(2000, 2800)
                flow_count = random.randint(100, 150)
                threat_num = 2
                
            elif current_attack == "Packet-In Flood":
                level, action_id, action_name = "MEDIUM_ATTACK (Packet-In Flood)", 1, "Block (Drop Suspicious Flow)"
                packet_rate = random.randint(100, 300)
                flow_count = random.randint(80, 100)
                threat_num = 1
                
            elif current_attack == "IP Spoofing":
                level, action_id, action_name = "HIGH_ATTACK (IP Spoofing)", 1, "Block (Drop Suspicious Flow)"
                packet_rate = random.randint(1500, 2000)
                flow_count = random.randint(40, 60)
                threat_num = 2
                
            elif current_attack == "Port Scanning":
                level, action_id, action_name = "HIGH_ATTACK (Port Scanning)", 3, "Redirect to Honeypot"
                packet_rate = random.randint(4000, 6000)
                flow_count = random.randint(40, 60)
                threat_num = 2
                
            else:
                level, action_id, action_name = "NORMAL", 0, "No Action"
                packet_rate = random.randint(10, 50)
                flow_count = random.randint(50, 80)
                threat_num = 0

            # Bắn dữ liệu sang Streamlit (CSV)
            new_data = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "packet_rate": packet_rate,
                "flow_count": flow_count,
                "level": level,
                "action_id": action_id,
                "action_name": action_name
            }])
            df = pd.read_csv(LOG_FILE)
            df = pd.concat([df, new_data], ignore_index=True).tail(50)
            df.to_csv(LOG_FILE, index=False)
            
            # ==========================================
            # BẮN DỮ LIỆU SANG GRAFANA (PROMETHEUS)
            # ==========================================
            PROM_PACKET_RATE.set(packet_rate)
            PROM_FLOW_COUNT.set(flow_count)
            PROM_THREAT_LEVEL.set(threat_num)
            PROM_ACTION.set(action_id)
            
        except Exception as e:
            print(f"[!] Pipeline Error: {e}")
            
        time.sleep(2)

if __name__ == "__main__":
    main()