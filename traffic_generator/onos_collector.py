import requests
import time
import numpy as np
import collections
import re
import psutil
import subprocess
from requests.auth import HTTPBasicAuth


class ONOSCollector:

    def __init__(self, onos_ip="127.0.0.1", victim_ip="10.0.0.8"):

        self.victim_ip = victim_ip
        self.url_flows = f"http://{onos_ip}:8181/onos/v1/flows"
        self.url_ports = f"http://{onos_ip}:8181/onos/v1/statistics/ports"
        self.auth = HTTPBasicAuth('onos', 'rocks')

        self.prev_packet, self.prev_byte, self.prev_drop = self._get_port_stats()
        self.prev_time = time.time()

        self.ip_history = collections.deque(maxlen=2000)

    def _get_port_stats(self):
        res = requests.get(self.url_ports, auth=self.auth)
        data = res.json()

        total_packets = 0
        total_bytes = 0
        total_drops = 0

        for device in data.get("statistics", []):
            for port in device.get("ports", []):
                total_packets += port.get("packetsReceived", 0)
                total_bytes += port.get("bytesReceived", 0)
                total_drops += port.get("packetsDropped", 0)

        return total_packets, total_bytes, total_drops

    def _get_flow_count(self):
        res = requests.get(self.url_flows, auth=self.auth)
        flows = res.json().get("flows", [])
        return len(flows)

    def _measure_latency(self):
        output = subprocess.getoutput(f"ping -c 1 -W 1 {self.victim_ip}")
        match = re.findall(r"time=([\d.]+)", output)
        return np.mean([float(x) for x in match]) if match else 0

    def get_state(self, attack_indicator=0):

        packet, byte, drop = self._get_port_stats()
        now = time.time()
        dt = max(now - self.prev_time, 1e-6)

        packet_rate = (packet - self.prev_packet) / dt
        byte_rate = (byte - self.prev_byte) / dt
        drop_rate = (drop - self.prev_drop) / dt

        self.prev_packet = packet
        self.prev_byte = byte
        self.prev_drop = drop
        self.prev_time = now

        flow_count = self._get_flow_count()
        latency = self._measure_latency()

        return [
            packet_rate,
            byte_rate,
            flow_count,
            latency,
            drop_rate,
            attack_indicator
        ]

    def save_to_csv(self, state, filename):
        with open(filename, "a") as f:
            f.write(",".join(map(str, state)) + "\n")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, default=0,
                        help="0 = normal, 1 = attack")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of samples to collect")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sampling interval (seconds)")
    parser.add_argument("--output", type=str, default="dataset.csv",
                        help="Output CSV file")

    args = parser.parse_args()

    collector = ONOSCollector()

    print("[*] Starting data collection...")
    print(f"Label: {args.label}")
    print(f"Samples: {args.samples}")
    print(f"Interval: {args.interval}s")
    print(f"Output: {args.output}")
    print("="*40)

    # Write header if file not exists
    try:
        with open(args.output, "x") as f:
            f.write("packet_rate,byte_rate,flow_count,latency,drop_rate,label\n")
    except FileExistsError:
        pass

    count = 0

    while count < args.samples:
        state = collector.get_state(attack_indicator=args.label)

        collector.save_to_csv(state, args.output)

        count += 1

        print(f"[{count}/{args.samples}] {state}")

        time.sleep(args.interval)

    print("Data collection completed.")
