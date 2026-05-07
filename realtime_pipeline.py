import time
import pandas as pd
import os
import random
from datetime import datetime

LOG_FILE = "logs/live_metrics.csv"
SIGNAL_FILE = "logs/current_attack.txt"

os.makedirs("logs", exist_ok=True)
if not os.path.exists(LOG_FILE):
    df_init = pd.DataFrame(columns=[
        "timestamp", "packet_rate", "flow_count", "level", "action_id", "action_name"
    ])
    df_init.to_csv(LOG_FILE, index=False)

def get_current_attack_state():
    """Đọc cờ trạng thái từ Mininet truyền sang"""
    if os.path.exists(SIGNAL_FILE):
        try:
            with open(SIGNAL_FILE, "r") as f:
                return f.read().strip()
        except:
            pass
    return "Normal Traffic"

def main():
    print("[*] Starting Deterministic Demo Agent (Wizard of Oz Mode)...")
    
    while True:
        try:
            # 1. Đọc chính xác kịch bản đang chạy
            current_attack = get_current_attack_state()
            
            # 2. Hardcode mapping chính xác 100% cho Demo
            if current_attack == "Normal Traffic":
                level = "NORMAL"
                action_id = 0
                action_name = "No Action"
                packet_rate = random.randint(10, 50)
                flow_count = random.randint(50, 80)
                
            elif current_attack == "DDoS Flood":
                level = "HIGH_ATTACK (DDoS Flood)"
                action_id = 1
                action_name = "Block (Drop Suspicious Flow)"
                packet_rate = random.randint(3000, 5000)
                flow_count = random.randint(30, 50)
                
            elif current_attack == "Flow Table Overflow":
                level = "HIGH_ATTACK (Flow Table Overflow)"
                action_id = 2
                action_name = "Rate Limit Bandwidth"
                packet_rate = random.randint(2000, 2800)
                flow_count = random.randint(100, 150)
                
            elif current_attack == "Packet-In Flood":
                level = "MEDIUM_ATTACK (Packet-In Flood)"
                action_id = 1
                action_name = "Block (Drop Suspicious Flow)"
                packet_rate = random.randint(100, 300)
                flow_count = random.randint(80, 100)
                
            elif current_attack == "IP Spoofing":
                level = "HIGH_ATTACK (IP Spoofing)"
                action_id = 1
                action_name = "Block (Drop Suspicious Flow)"
                packet_rate = random.randint(1500, 2000)
                flow_count = random.randint(40, 60)
                
            elif current_attack == "Port Scanning":
                level = "HIGH_ATTACK (Port Scanning)"
                action_id = 3
                action_name = "Redirect to Honeypot"
                packet_rate = random.randint(4000, 6000)
                flow_count = random.randint(40, 60)
                
            else:
                level = "NORMAL"
                action_id = 0
                action_name = "No Action"
                packet_rate = random.randint(10, 50)
                flow_count = random.randint(50, 80)

            # 3. Ghi dữ liệu ra CSV cho Dashboard render
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
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] State: {current_attack} | PktRate: {packet_rate} | Action: {action_name}")
            
        except Exception as e:
            print(f"[!] Pipeline Error: {e}")
            
        time.sleep(2)

if __name__ == "__main__":
    main()