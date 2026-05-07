import time
import pandas as pd
import os
from datetime import datetime
import requests
from requests.auth import HTTPBasicAuth

# Dùng internal DNS của Docker để gọi sang ONOS
ONOS_URL = "http://onos-controller:8181/onos/v1" 
AUTH = HTTPBasicAuth("onos", "rocks")
LOG_FILE = "logs/live_metrics.csv"

# Khởi tạo thư mục và file log
os.makedirs("logs", exist_ok=True)
if not os.path.exists(LOG_FILE):
    df_init = pd.DataFrame(columns=[
        "timestamp", "packet_rate", "flow_count", "level", "action_id", "action_name"
    ])
    df_init.to_csv(LOG_FILE, index=False)

def get_flows():
    try:
        r = requests.get(f"{ONOS_URL}/flows", auth=AUTH, timeout=2)
        if r.status_code == 200:
            return r.json().get("flows", [])
    except:
        pass
    return []

def main():
    print("[*] Starting Volume-based Defense Agent Service...")
    prev_packets = 0
    
    while True:
        try:
            flows = get_flows()
            flow_count = len(flows)
            
            # Tính tổng packet để ra packet_rate
            current_packets = sum(f.get("packets", 0) for f in flows)
            packet_rate = max(0, current_packets - prev_packets)
            prev_packets = current_packets
            
            # ===== VOLUME-BASED DETECTION =====
            if packet_rate < 50:
                level = "NORMAL"
                action_id = 0
                action_name = "No Action"
            elif packet_rate < 1500:
                level = "MEDIUM_ATTACK"
                action_id = 1
                action_name = "Block (Drop Flow)"
            else:
                level = "HIGH_ATTACK"
                action_id = 2
                action_name = "Rate Limit Bandwidth"

            # Log xuống CSV cho Dashboard đọc (Chỉ giữ 50 dòng mới nhất cho nhẹ)
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
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] PktRate: {packet_rate} | Level: {level} | Action: {action_name}")
            
        except Exception as e:
            print(f"[!] Pipeline Error: {e}")
            
        time.sleep(2) # Polling mỗi 2 giây

if __name__ == "__main__":
    main()