import requests
import time
import math

ONOS_URL = "http://controller:8181/onos/v1"
AUTH = ("onos", "rocks")

prev_packets = 0
prev_bytes = 0
prev_time = time.time()

# EMA smoothing state
prev_smoothed_packet_rate = 0.0
alpha = 0.3


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
    global prev_smoothed_packet_rate

    try:
        # ---------------- FLOWS ----------------
        flow_res = requests.get(f"{ONOS_URL}/flows", auth=AUTH)
        flow_res.raise_for_status()
        flows = flow_res.json().get("flows", [])

        total_packets = sum(f.get("packets", 0) for f in flows)
        total_bytes = sum(f.get("bytes", 0) for f in flows)

        now = time.time()
        dt = max(now - prev_time, 1e-6)

        packet_delta = total_packets - prev_packets
        if packet_delta < 0:
            packet_delta = total_packets  # counter reset safety

        byte_delta = total_bytes - prev_bytes
        if byte_delta < 0:
            byte_delta = total_bytes

        # -------- RAW RATE --------
        packet_rate_raw = packet_delta / dt
        byte_rate = byte_delta / dt

        # -------- EMA SMOOTHING --------
        packet_rate = (
            alpha * packet_rate_raw +
            (1 - alpha) * prev_smoothed_packet_rate
        )

        prev_smoothed_packet_rate = packet_rate

        prev_packets = total_packets
        prev_bytes = total_bytes
        prev_time = now

        # -------- FLOW NORMALIZATION --------
        flow_ratio = min(len(flows) / 120.0, 1.0)

        # -------- ENTROPY --------
        src_ips = []
        for f in flows:
            for c in f.get("selector", {}).get("criteria", []):
                if c.get("type") == "IPV4_SRC":
                    src_ips.append(c.get("ip"))

        if len(src_ips) > 1:
            entropy = calculate_entropy(src_ips)
        else:
            entropy = min(flow_ratio * 2.0, 1.0)

        # -------- CPU & QUEUE --------
        controller_cpu = min(max(packet_rate / 2000.0, 0.0), 1.0)
        queue_length = min(len(flows) / 50.0, 1.0)

        # -------- LATENCY PROXY --------
        latency = min(packet_rate * 0.00005, 1.0)

        print("PACKET_RATE:", packet_rate)
        print("FLOW_COUNT:", len(flows))
        print("ENTROPY:", entropy)

        return {
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "flow_count": flow_ratio,
            "src_ip_entropy": entropy,
            "latency": latency,
            "packet_loss": 0.0,
            "queue_length": queue_length,
            "controller_cpu": controller_cpu
        }

    except Exception as e:
        print("State error:", e)
        return None