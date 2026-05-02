import requests
import time
import math

ONOS_URL = "http://controller:8181/onos/v1"
AUTH = ("onos", "rocks")

prev_packets = 0
prev_bytes = 0
prev_time = time.time()

def calculate_entropy(ip_list):
    if not ip_list:
        return 0.0
    freq = {}
    for ip in ip_list:
        freq[ip] = freq.get(ip, 0) + 1
    total = len(ip_list)
    entropy = 0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy / math.log2(len(freq)) if len(freq) > 1 else 0.0


def get_state():
    global prev_packets, prev_bytes, prev_time

    try:
        # --- FLOWS ---
        flow_res = requests.get(f"{ONOS_URL}/flows", auth=AUTH)
        flow_res.raise_for_status()
        flows = flow_res.json().get("flows", [])

        total_packets = sum(f.get("packets", 0) for f in flows)
        total_bytes = sum(f.get("bytes", 0) for f in flows)

        src_ips = []
        for f in flows:
            crit = f.get("selector", {}).get("criteria", [])
            for c in crit:
                if c.get("type") == "IPV4_SRC":
                    src_ips.append(c.get("ip"))

        entropy = calculate_entropy(src_ips)

        now = time.time()
        dt = max(now - prev_time, 1e-6)

        packet_delta = max(total_packets - prev_packets, 0)
        byte_delta = max(total_bytes - prev_bytes, 0)

        packet_rate = packet_delta / dt
        byte_rate = byte_delta / dt

        prev_packets = total_packets
        prev_bytes = total_bytes
        prev_time = now

        flow_count = len(flows)

        packet_rate_scaled = packet_rate * 1000

        if len(src_ips) > 1:
            entropy = calculate_entropy(src_ips)
        else:
            entropy = min(flow_count / 50.0, 1.0)

        controller_cpu = min(packet_rate / 1000.0, 1.0)

        queue_length = min(len(flows) / 50.0, 1.0)

        latency = packet_rate_scaled * 0.00005

        print("PACKET RATE:", packet_rate)
        print("FLOW COUNT:", len(flows))
        print("ENTROPY:", entropy)

        return {
            "packet_rate": packet_rate_scaled,
            "byte_rate": byte_rate,
            "flow_count": len(flows),
            "src_ip_entropy": entropy,
            "latency": latency,
            "packet_loss": 0.0,
            "queue_length": queue_length,
            "controller_cpu": controller_cpu
        }

    except Exception as e:
        print("State error:", e)
        return None