import requests
import time
import numpy as np
import collections
import re
import psutil
import subprocess
from requests.auth import HTTPBasicAuth

class ONOSCollector:
    def __init__(self, onos_ip="127.0.0.1", victim_ip="10.0.0.7"):
        self.victim_ip = victim_ip
        self.url_flows = f"http://{onos_ip}:8181/onos/v1/flows"
        self.url_ports = f"http://{onos_ip}:8181/onos/v1/statistics/ports"
        self.auth = HTTPBasicAuth('onos', 'rocks')

        self.prev_packet = 0
        self.prev_byte = 0
        self.prev_drop = 0
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
        output = subprocess.getoutput(f"ping -c 3 {self.victim_ip}")
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