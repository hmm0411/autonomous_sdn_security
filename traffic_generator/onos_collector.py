import requests
import time
import numpy as np
import collections
import re
import subprocess
from requests.auth import HTTPBasicAuth

class ONOSCollector:
    def __init__(self, onos_ip="127.0.0.1", victim_ip="10.0.0.8"):
        self.victim_ip = victim_ip
        self.url_flows = f"http://{onos_ip}:8181/onos/v1/flows"
        self.url_ports = f"http://{onos_ip}:8181/onos/v1/statistics/ports"
        self.auth = HTTPBasicAuth('onos', 'rocks')

        self.prev_packets = 0
        self.prev_bytes = 0
        self.prev_time = time.time()

    def _get_port_stats(self):
        try:
            r = requests.get(self.url_ports, auth=self.auth, timeout=5)
            data = r.json()

            total_packets = 0
            total_bytes = 0
            total_drop = 0

            for dev in data.get("statistics", []):
                for port in dev.get("ports", []):
                    total_packets += port.get("packetsReceived", 0)
                    total_bytes += port.get("bytesReceived", 0)
                    total_drop += port.get("packetsDropped", 0)

            return total_packets, total_bytes, total_drop

        except:
            return 0, 0, 0

    def _get_flow_count(self):
        try:
            r = requests.get(self.url_flows, auth=self.auth, timeout=5)
            return len(r.json().get("flows", []))
        except:
            return 0

    def _measure_latency(self):
        try:
            result = subprocess.run(
                ["ping", "-c", "1", self.victim_ip],
                capture_output=True,
                text=True,
                timeout=2
            )

            match = re.findall(r"time=([\d.]+)", result.stdout)
            if match:
                return float(match[0])
        except:
            pass

        return 500.0  # Mặc định nếu không đo được

    def get_raw_state(self, attack_indicator=0):
        packets, bytes_, drops = self._get_port_stats()

        now = time.time()
        dt = max(now - self.prev_time, 1e-6)

        packet_rate = (packets - self.prev_packets) / dt
        byte_rate = (bytes_ - self.prev_bytes) / dt
        drop_rate = drops / dt

        self.prev_packets = packets
        self.prev_bytes = bytes_
        self.prev_time = now

        flow_count = self._get_flow_count()
        latency = self._measure_latency()

        return {
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "flow_count": flow_count,
            "latency": latency,
            "packet_loss": min(1.0, drop_rate / 1000.0),
        }

    def save_to_csv(self, state, filename):
        with open(filename, "a") as f:
            f.write(",".join(map(str, state.values())) + "\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--label", type=int, default=0)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--output", type=str, default="dataset.csv")
    args = p.parse_args()

    collector = ONOSCollector()
    print(f"[*] Thu thập nhãn {args.label}...")
    
    try:
        with open(args.output, "x") as f:
            f.write("packet_rate,byte_rate,flow_count,latency,packet_loss,label\n")
    except FileExistsError: pass

    for i in range(1, args.samples + 1):
        state = collector.get_raw_state(attack_indicator=args.label)
        collector.save_to_csv(state, args.output)
        print(f"[{i}/{args.samples}] {state}")
        time.sleep(args.interval)
