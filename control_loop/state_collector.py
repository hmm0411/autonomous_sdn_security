import requests

ONOS_URL = "http://onos:8181/onos/v1"
AUTH = ("onos", "rocks")

def get_state():
    try:
        response = requests.get(f"{ONOS_URL}/flows", auth=AUTH)
        response.raise_for_status()

        flows = response.json().get('flows', [])

        flow_count = len(flows)
        total_packets = sum(f.get("packets", 0) for f in flows)
        total_bytes = sum(f.get("bytes", 0) for f in flows)

        raw = {
            "total_packets": total_packets,
            "total_bytes": total_bytes,
            "latency": 0,
            "flow_count": flow_count,
            "packet_loss": 0,
            "src_ip_entropy": 0,
            "queue_length": 0,
            "controller_cpu": 0
        }

        return raw

    except Exception as e:
        print(f"Lỗi ONOS: {e}")

        return {
            "total_packets": 0,
            "total_bytes": 0,
            "latency": 0,
            "flow_count": 0,
            "packet_loss": 0,
            "src_ip_entropy": 0,
            "queue_length": 0,
            "controller_cpu": 0
        }