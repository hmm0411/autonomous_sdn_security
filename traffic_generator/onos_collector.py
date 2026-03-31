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
        self.prev_packet, self.prev_byte, self.prev_drop = self._get_port_stats()
        self.prev_time = time.time()

    def _get_port_stats(self):
        try:
            res = requests.get(self.url_ports, auth=self.auth)
            data = res.json()
            tp, tb, td = 0, 0, 0
            for device in data.get("statistics", []):
                for port in device.get("ports", []):
                    tp += port.get("packetsReceived", 0)
                    tb += port.get("bytesReceived", 0)
                    td += port.get("packetsDropped", 0)
            return tp, tb, td
        except:
            return 0, 0, 0

    def _get_flow_count(self):
        try:
            res = requests.get(self.url_flows, auth=self.auth)
            return len(res.json().get("flows", []))
        except:
            return 0

    def _measure_latency(self):
        try:
            # Tìm PID của h7 và ping sang h8 (Bỏ sudo bên trong vì script chạy bằng sudo rồi)
            pid = subprocess.getoutput("pgrep -f 'mininet:h7'").strip().split('\n')[0]
            if not pid: return 500.0
            
            cmd = f"mnexec -a {pid} ping -c 1 -W 1 {self.victim_ip}"
            output = subprocess.getoutput(cmd)
            match = re.findall(r"time=([\d.]+)", output)
            return float(match[0]) if match else 500.0
        except:
            return 500.0

    def get_state(self, attack_indicator=0):
        packet, byte, drop = self._get_port_stats()
        now = time.time()
        dt = max(now - self.prev_time, 1e-6)
        
        pr = (packet - self.prev_packet) / dt
        br = (byte - self.prev_byte) / dt
        dr = (drop - self.prev_drop) / dt

        self.prev_packet, self.prev_byte, self.prev_drop, self.prev_time = packet, byte, drop, now
        
        return [pr, br, self._get_flow_count(), self._measure_latency(), dr, attack_indicator]

    def save_to_csv(self, state, filename):
        with open(filename, "a") as f:
            f.write(",".join(map(str, state)) + "\n")

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
            f.write("packet_rate,byte_rate,flow_count,latency,drop_rate,label\n")
    except FileExistsError: pass

    for i in range(1, args.samples + 1):
        state = collector.get_state(attack_indicator=args.label)
        collector.save_to_csv(state, args.output)
        print(f"[{i}/{args.samples}] {state}")
        time.sleep(args.interval)