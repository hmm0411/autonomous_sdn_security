import requests
import time
import numpy as np

ONOS_URL = "http://controller:8181/onos/v1"
AUTH = ("onos", "rocks")

prev_packets = 0
prev_bytes = 0
prev_flow = 0
prev_time = time.time()

def get_state():
    global prev_packets, prev_bytes, prev_flow, prev_time

    try:
        response = requests.get(f"{ONOS_URL}/flows", auth=AUTH)
        response.raise_for_status()

        flows = response.json().get('flows', [])

        flow_count = len(flows)
        total_packets = sum(f.get("packets", 0) for f in flows)
        total_bytes = sum(f.get("bytes", 0) for f in flows)

        now = time.time()
        dt = max(now - prev_time, 1e-6)

        # ===== RATE =====
        packet_rate = (total_packets - prev_packets) / dt
        byte_rate = (total_bytes - prev_bytes) / dt

        # ===== FLOW GROWTH =====
        flow_growth_rate = (flow_count - prev_flow) / dt

        # ===== SIMPLE ESTIMATIONS =====
        latency = 5 + flow_count * 0.1
        packet_loss = min(0.1, flow_count / 200)

        # entropy (simple)
        n = max(flow_count, 1)
        entropy = np.log2(n)

        controller_cpu = 10 + flow_count * 0.2

        # update prev
        prev_packets = total_packets
        prev_bytes = total_bytes
        prev_flow = flow_count
        prev_time = now

        return {
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "flow_count": flow_count,
            "flow_growth_rate": flow_growth_rate,
            "src_ip_entropy": entropy,
            "latency": latency,
            "packet_loss": packet_loss,
            "controller_cpu": controller_cpu,
        }

    except Exception as e:
        print("State error:", e)
        return {
            "packet_rate": 0,
            "byte_rate": 0,
            "flow_count": 0,
            "flow_growth_rate": 0,
            "src_ip_entropy": 0,
            "latency": 0,
            "packet_loss": 0,
            "controller_cpu": 0,
        }