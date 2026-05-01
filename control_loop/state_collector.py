import requests
import time

ONOS_URL = "http://controller:8181/onos/v1"
AUTH = ("onos", "rocks")

prev_packets = 0
prev_bytes = 0
prev_time = time.time()

def get_state():
    global prev_packets, prev_bytes, prev_time

    try:
        # ---- PORT STATS ----
        port_res = requests.get(f"{ONOS_URL}/statistics/ports", auth=AUTH)
        port_res.raise_for_status()
        port_data = port_res.json()

        total_packets = 0
        total_bytes = 0
        queue = 0

        for device in port_data.get("statistics", []):
            for port in device.get("ports", []):
                total_packets += port.get("packetsReceived", 0)
                total_bytes += port.get("bytesReceived", 0)
                queue += port.get("packetsDropped", 0)

        now = time.time()
        dt = max(now - prev_time, 1e-6)

        packet_rate = (total_packets - prev_packets) / dt
        byte_rate = (total_bytes - prev_bytes) / dt

        prev_packets = total_packets
        prev_bytes = total_bytes
        prev_time = now

        # ---- FLOW COUNT ----
        flow_res = requests.get(f"{ONOS_URL}/flows", auth=AUTH)
        flow_res.raise_for_status()
        flows = flow_res.json().get("flows", [])
        flow_count = len(flows)

        # ---- QoS (placeholder) ----
        latency = 0.1
        packet_loss = 0.0
        controller_cpu = 0.1

        return {
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "flow_count": flow_count,
            "src_ip_entropy": 0.0,
            "latency": latency,
            "packet_loss": packet_loss,
            "queue_length": queue,
            "controller_cpu": controller_cpu
        }

    except Exception as e:
        print("State error:", e)
        return None