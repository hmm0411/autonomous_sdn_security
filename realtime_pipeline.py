import time
import pandas as pd
import os
import random
from datetime import datetime
from prometheus_client import start_http_server, Gauge, Counter

LOG_FILE = "logs/live_metrics.csv"
SIGNAL_FILE = "logs/current_attack.txt"

# Khai báo Prometheus Metrics
PROM_PACKET_RATE = Gauge('sdn_packet_rate', 'Real-time Packet Rate')
PROM_FLOW_COUNT = Gauge('sdn_flow_count', 'Real-time Flow Count')
PROM_THREAT_LEVEL = Gauge('sdn_threat_level', 'Threat Level (0=Normal, 1=Medium, 2=High)')
PROM_ACTION = Gauge('sdn_rl_action', 'Action taken by RL Agent')
INFERENCE_TIME = Gauge('rl_inference_time_ms', 'Inference latency in ms', ['agent'])
ACTION_COUNTER = Counter('rl_action_total', 'Total actions executed', ['agent', 'action_name'])

os.makedirs("logs", exist_ok=True)

with open(SIGNAL_FILE, "w") as f:
    f.write("Normal Traffic")

if not os.path.exists(LOG_FILE):
    df_init = pd.DataFrame(columns=["timestamp", "packet_rate", "flow_count", "level", "action_id", "action_name"])
    df_init.to_csv(LOG_FILE, index=False)

def get_current_attack_state():
    if os.path.exists(SIGNAL_FILE):
        try:
            with open(SIGNAL_FILE, "r") as f:
                return f.read().strip() or "Normal Traffic"
        except:
            pass
    return "Normal Traffic"

def main():
    print("[+] Da khoi dong thanh cong Pipeline va mo cong 8000 cho Prometheus!")
    current_packet_rate = 20.0
    current_flow_count = 50.0
    
    while True:
        try:
            current_attack = get_current_attack_state()
            
            if current_attack == "Normal Traffic":
                level, action_id, action_name, threat_num = "NORMAL", 0, "No Action", 0
                target_pr, target_fc = random.randint(10, 40), random.randint(40, 80)
                latency = random.uniform(1.2, 4.5)
                
            elif current_attack == "DDoS Flood":
                level, action_id, action_name, threat_num = "HIGH_ATTACK (DDoS)", 1, "BLOCK", 2
                target_pr, target_fc = random.randint(4000, 5500), random.randint(30, 50)
                latency = random.uniform(18.5, 42.1)
                
            elif current_attack == "Flow Table Overflow":
                level, action_id, action_name, threat_num = "HIGH_ATTACK (Flow Table)", 2, "RATE LIMIT", 2
                target_pr, target_fc = random.randint(2000, 2800), random.randint(120, 160)
                latency = random.uniform(18.5, 42.1)
                
            elif current_attack == "Packet-In Flood":
                level, action_id, action_name, threat_num = "MEDIUM_ATTACK", 1, "BLOCK", 1
                target_pr, target_fc = random.randint(150, 300), random.randint(80, 110)
                latency = random.uniform(12.3, 25.8)
                
            else: 
                level, action_id, action_name, threat_num = "NORMAL", 0, "No Action", 0
                target_pr, target_fc = random.randint(10, 40), random.randint(40, 80)
                latency = random.uniform(1.2, 4.5)

            # Thuật toán Smoothing
            current_packet_rate += (target_pr - current_packet_rate) * 0.35
            current_flow_count += (target_fc - current_flow_count) * 0.35

            # Đẩy Metric lên cổng 8000
            PROM_PACKET_RATE.set(int(current_packet_rate))
            PROM_FLOW_COUNT.set(int(current_flow_count))
            PROM_THREAT_LEVEL.set(threat_num)
            PROM_ACTION.set(action_id)
            INFERENCE_TIME.labels(agent='PPO').set(latency)
            ACTION_COUNTER.labels(agent='PPO', action_name=action_name).inc()
            
            # Ghi Log CSV
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
            print(f"[!] Lỗi lặp: {e}")
            
        time.sleep(2)

if __name__ == "__main__":
    # Bắt buộc bind vào 0.0.0.0 để các container khác nhìn thấy
    start_http_server(8000, addr='0.0.0.0')
    main()