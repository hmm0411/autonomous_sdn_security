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
        # --- FLOWS ---
        flow_res = requests.get(f"{ONOS_URL}/flows", auth=AUTH)
        flow_res.raise_for_status()
        flows = flow_res.json().get("flows", [])

        total_packets = sum(f.get("packets", 0) for f in flows)
        total_bytes = sum(f.get("bytes", 0) for f in flows)

        now = time.time()
        dt = max(now - prev_time, 1e-6)

        packet_delta = max(total_packets - prev_packets, 0)
        byte_delta = max(total_bytes - prev_bytes, 0)

        packet_rate = packet_delta / dt
        byte_rate = byte_delta / dt

        prev_packets = total_packets
        prev_bytes = total_bytes
        prev_time = now

        # --- PORT STATS (real queue approximation) ---
        port_res = requests.get(f"{ONOS_URL}/statistics/ports", auth=AUTH)
        port_res.raise_for_status()
        port_data = port_res.json()

        dropped = 0
        for device in port_data.get("statistics", []):
            for port in device.get("ports", []):
                dropped += port.get("packetsRxDropped", 0)

        # --- DEBUG PRINT ---
        print("PACKET RATE:", packet_rate)
        print("BYTE RATE:", byte_rate)
        print("FLOW COUNT:", len(flows))
        print("DROPPED:", dropped)

        return {
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "flow_count": len(flows),
            "src_ip_entropy": 0.0,   # bỏ tạm entropy
            "latency": packet_rate * 0.00001,  # proxy latency
            "packet_loss": dropped,
            "queue_length": dropped,
            "controller_cpu": min(packet_rate / 10000, 1.0)
        }

    except Exception as e:
        print("State error:", e)
        return None