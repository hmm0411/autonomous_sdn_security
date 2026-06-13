import math
import os
import time

import requests


ONOS_URL = os.getenv("ONOS_URL", "http://controller:8181/onos/v1")
AUTH = (
    os.getenv("ONOS_USER", "onos"),
    os.getenv("ONOS_PASS", "rocks"),
)
TIMEOUT = float(os.getenv("ONOS_TIMEOUT", "3"))


prev_packets = 0.0
prev_bytes = 0.0
prev_flow_count = 0.0
prev_time = time.time()


def _entropy(values):
    if not values:
        return 0.0

    counts = {}

    for value in values:
        counts[value] = counts.get(value, 0) + 1

    total = float(len(values))

    entropy = 0.0

    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def _extract_src_ips_from_flows(flows):
    src_ips = []

    for flow in flows:
        selector = flow.get("selector", {})
        criteria = selector.get("criteria", [])

        for criterion in criteria:
            if criterion.get("type") == "IPV4_SRC":
                ip_value = criterion.get("ip")

                if ip_value:
                    src_ips.append(str(ip_value))

    return src_ips


def get_state():
    """
    Return raw dict for state 8:

    0 packet_rate
    1 byte_rate
    2 flow_count
    3 flow_growth_rate
    4 src_ip_entropy
    5 latency
    6 packet_loss
    7 controller_cpu
    """
    global prev_packets
    global prev_bytes
    global prev_flow_count
    global prev_time

    try:
        response = requests.get(
            f"{ONOS_URL}/flows",
            auth=AUTH,
            timeout=TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()
        flows = data.get("flows", [])

        total_packets = sum(
            float(flow.get("packets", 0) or 0)
            for flow in flows
        )

        total_bytes = sum(
            float(flow.get("bytes", 0) or 0)
            for flow in flows
        )

        flow_count = float(len(flows))

        now = time.time()
        dt = max(now - prev_time, 1e-6)

        packet_rate = max(total_packets - prev_packets, 0.0) / dt
        byte_rate = max(total_bytes - prev_bytes, 0.0) / dt
        flow_growth_rate = (flow_count - prev_flow_count) / dt

        src_ips = _extract_src_ips_from_flows(flows)
        src_ip_entropy = _entropy(src_ips)

        # ONOS REST không cung cấp latency/loss trực tiếp.
        # Ở runtime demo, ta dùng proxy từ counters để hệ thống có metric liên tục.
        latency = min(
            300.0,
            5.0
            + packet_rate * 0.02
            + max(flow_growth_rate, 0.0) * 0.5,
        )

        packet_loss = min(
            1.0,
            max(
                0.0,
                packet_rate / 100000.0
                + max(flow_growth_rate, 0.0) / 10000.0,
            ),
        )

        controller_cpu = min(
            100.0,
            10.0
            + packet_rate / 100.0
            + flow_count * 0.05,
        )

        prev_packets = total_packets
        prev_bytes = total_bytes
        prev_flow_count = flow_count
        prev_time = now

        raw = {
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "flow_count": flow_count,
            "flow_growth_rate": flow_growth_rate,
            "src_ip_entropy": src_ip_entropy,
            "latency": latency,
            "packet_loss": packet_loss,
            "controller_cpu": controller_cpu,
        }

        print(
            "[STATE]",
            {k: round(float(v), 4) for k, v in raw.items()},
            flush=True,
        )

        return raw

    except Exception as e:
        print(f"[STATE_ERROR] {e}", flush=True)
        return None