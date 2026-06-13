import time
import pandas as pd
import os
import random
from datetime import datetime
from prometheus_client import start_http_server, Gauge

LOG_FILE = "logs/live_metrics.csv"
SIGNAL_FILE = "logs/current_attack.txt"

PROM_PACKET_RATE = Gauge('sdn_packet_rate', 'Real-time Packet Rate')
PROM_FLOW_COUNT = Gauge('sdn_flow_count', 'Real-time Flow Count')
PROM_THREAT_LEVEL = Gauge('sdn_threat_level', 'Threat Level (0=Normal, 1=Medium, 2=High)')
PROM_ACTION = Gauge('sdn_rl_action', 'Action taken by RL Agent')
PROM_CURRENT_SCORE = Gauge('sdn_current_score', 'Dynamic Threat Score')
# Đảm bảo thư mục tồn tại
os.makedirs("logs", exist_ok=True)

current_score = 0.0

def init_log_file():
    if not os.path.exists(LOG_FILE):
        df_init = pd.DataFrame(columns=[
            "timestamp",
            "packet_rate",
            "flow_count",
            "score",
            "threat_level",
            "level",
            "action_id",
            "action_name",
            "attack_state",
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

def get_dynamic_score(target_score):
    global current_score
    step = 12.0

    if current_score < target_score:
        current_score = min(current_score + step, target_score)
    elif current_score > target_score:
        current_score = max(current_score - step, target_score)

    return current_score

def classify_attack(current_attack):
    """Map attack signal -> target score, action, packet/flow target.

    Đồng bộ với traffic_generator/validate_attack_scenarios.py:
    - Normal Traffic
    - DDoS Flood Attack
    - IP Spoofing Detected
    - Packet-In Anomaly
    """

    if "DDoS" in current_attack:
        return {
            "target_score": 99.0,
            "level": "CRITICAL (DDoS Flood Attack)",
            "action_id": 1,
            "action_name": "Block (Drop Suspicious Flow)",
            "target_packet_rate": random.randint(4000, 5500),
            "target_flow_count": random.randint(30, 50),
        }

    if "Spoof" in current_attack:
        return {
            "target_score": 88.0,
            "level": "CRITICAL (IP Spoofing Detected)",
            "action_id": 4,
            "action_name": "Isolate Device",
            "target_packet_rate": random.randint(1800, 2500),
            "target_flow_count": random.randint(40, 60),
        }

    if "Packet-In" in current_attack:
        return {
            "target_score": 78.0,
            "level": "WARNING (Packet-In Anomaly)",
            "action_id": 2,
            "action_name": "Rate Limit Bandwidth",
            "target_packet_rate": random.randint(150, 300),
            "target_flow_count": random.randint(80, 110),
        }

    if "Flow Table" in current_attack or "Flow Overflow" in current_attack:
        return {
            "target_score": 85.0,
            "level": "CRITICAL (Flow Table Overflow)",
            "action_id": 2,
            "action_name": "Rate Limit Bandwidth",
            "target_packet_rate": random.randint(2000, 2800),
            "target_flow_count": random.randint(120, 160),
        }

    if "Port Scanning" in current_attack or "Port Scan" in current_attack:
        return {
            "target_score": 65.0,
            "level": "WARNING (Port Scanning)",
            "action_id": 3,
            "action_name": "Redirect to Honeypot",
            "target_packet_rate": random.randint(3500, 4500),
            "target_flow_count": random.randint(40, 60),
        }

    return {
        "target_score": 0.0,
        "level": "SAFE (Normal Traffic)",
        "action_id": 0,
        "action_name": "No Action",
        "target_packet_rate": random.randint(10, 40),
        "target_flow_count": random.randint(40, 80),
    }


def score_to_threat_level(score):
    if score < 31:
        return 0
    if score < 71:
        return 1
    return 2

def main():
    print("[*] Starting Smooth Demo Agent...")
    init_log_file()

    current_packet_rate = 20.0
    current_flow_count = 50.0

    while True:
        try:
            current_attack = get_current_attack_state()
            decision = classify_attack(current_attack)

            display_score = get_dynamic_score(decision["target_score"])
            threat_level = score_to_threat_level(display_score)
            action_id = decision["action_id"]
            action_name = decision["action_name"]
            level = decision["level"]

            current_packet_rate += (decision["target_packet_rate"] - current_packet_rate) * 0.35
            current_flow_count += (decision["target_flow_count"] - current_flow_count) * 0.35

            # ===== Prometheus metrics đồng bộ với Grafana dashboard =====
            PROM_PACKET_RATE.set(int(current_packet_rate))
            PROM_FLOW_COUNT.set(int(current_flow_count))
            PROM_THREAT_LEVEL.set(threat_level)
            PROM_ACTION.set(action_id)
            PROM_CURRENT_SCORE.set(display_score)

            new_data = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "packet_rate": int(current_packet_rate),
                "flow_count": int(current_flow_count),
                "score": round(display_score, 1),
                "threat_level": threat_level,
                "level": level,
                "action_id": action_id,
                "action_name": action_name,
                "attack_state": current_attack,
            }])

            new_data.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Attack={current_attack} | Score={display_score:.1f} | "
                f"Threat={threat_level} | Action={action_id}:{action_name} | "
                f"PPS={int(current_packet_rate)} | Flows={int(current_flow_count)}"
            )

        except Exception as e:
            print(f"[!] Pipeline Error: {e}")

        time.sleep(2)


if __name__ == "__main__":
    start_http_server(8000)
    main()