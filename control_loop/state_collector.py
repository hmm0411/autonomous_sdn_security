import requests
import numpy as np

ONOS_URL = "http://controller:8181/onos/v1"   
AUTH = ("onos", "rocks")

def get_state():
    try:
        response = requests.get(f"{ONOS_URL}/flows", auth=AUTH, timeout=5)
        response.raise_for_status()
        flows = response.json().get('flows', [])
        flow_count = len(flows)

        # Trả về DICT để StateBuilder.build() gọi .get() được
        return {
            "packet_rate":    float(flow_count * 10),
            "byte_rate":      0.0,
            "flow_count":     float(flow_count),
            "src_ip_entropy": 0.0,
            "latency":        0.0,
            "packet_loss":    0.0,
            "queue_length":   min(1.0, flow_count / 100),
            "controller_cpu": min(1.0, flow_count / 100),
            "attack_type":    None,
        }
    except Exception as e:
        print(f"Lỗi khi lấy state từ ONOS: {e}")
        # Trả về DICT rỗng thay vì list
        return {
            "packet_rate": 0.0, "byte_rate": 0.0, "flow_count": 0.0,
            "src_ip_entropy": 0.0, "latency": 0.0, "packet_loss": 0.0,
            "queue_length": 0.0, "controller_cpu": 0.0, "attack_type": None,
        }