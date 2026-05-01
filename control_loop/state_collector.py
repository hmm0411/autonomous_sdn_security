import requests
import time

ONOS_URL = "http://controller:8181/onos/v1"
AUTH = ("onos", "rocks")

prev_flow_count = 0
prev_time = time.time()

def get_state():
    global prev_flow_count, prev_time

    try:
        flow_res = requests.get(f"{ONOS_URL}/flows", auth=AUTH)
        flow_res.raise_for_status()
        flows = flow_res.json().get("flows", [])

        flow_count = len(flows)

        now = time.time()
        dt = max(now - prev_time, 1e-6)

        flow_rate = (flow_count - prev_flow_count) / dt

        prev_flow_count = flow_count
        prev_time = now

        # Entropy (IP diversity)
        src_ips = []
        for f in flows:
            crit = f.get("selector", {}).get("criteria", [])
            for c in crit:
                if c.get("type") == "IPV4_SRC":
                    src_ips.append(c.get("ip"))

        entropy = len(set(src_ips))

        return {
            "packet_rate": flow_rate,  # đổi thành flow_rate
            "byte_rate": 0.0,
            "flow_count": flow_count,
            "src_ip_entropy": entropy,
            "latency": 0.1,
            "packet_loss": 0.0,
            "queue_length": 0.0,
            "controller_cpu": 0.1
        }

    except Exception as e:
        print("State error:", e)
        return None