import os
import csv
import re
import time
import math
import collections
import subprocess
import requests
from requests.auth import HTTPBasicAuth


class ONOSCollector:
    def __init__(
        self,
        onos_ip="127.0.0.1",
        victim_ip="10.0.0.8",
        latency_host="h1",
        entropy_host="h8",
        entropy_iface="h8-eth0",
        onos_container="onos-controller",
    ):
        self.victim_ip = victim_ip
        self.latency_host = latency_host
        self.entropy_host = entropy_host
        self.entropy_iface = entropy_iface
        self.onos_container = onos_container

        self.url_flows = f"http://{onos_ip}:8181/onos/v1/flows"
        self.url_ports = f"http://{onos_ip}:8181/onos/v1/statistics/ports"
        self.auth = HTTPBasicAuth("onos", "rocks")

        self.prev_packets = 0
        self.prev_bytes = 0
        self.prev_drops = 0
        self.prev_flow_count = 0
        self.prev_time = time.time()

    # =========================
    # MININET HOST PID
    # =========================
    def _find_mininet_host_pid(self, host_name):
        try:
            result = subprocess.run(
                f"pgrep -f 'mininet:{host_name}' | head -n 1",
                shell=True,
                capture_output=True,
                text=True,
                timeout=2
            )
            pid = result.stdout.strip()
            return pid if pid else None
        except Exception as e:
            print(f"[WARN] Cannot find PID for {host_name}: {e}")
            return None

    # =========================
    # ONOS PORT STATS
    # =========================
    def _get_port_stats(self):
        try:
            r = requests.get(self.url_ports, auth=self.auth, timeout=5)
            r.raise_for_status()
            data = r.json()

            total_packets = 0
            total_bytes = 0
            total_drops = 0

            for dev in data.get("statistics", []):
                for port in dev.get("ports", []):
                    total_packets += port.get("packetsReceived", 0)
                    total_packets += port.get("packetsSent", 0)

                    total_bytes += port.get("bytesReceived", 0)
                    total_bytes += port.get("bytesSent", 0)

                    total_drops += port.get("packetsDropped", 0)

            return total_packets, total_bytes, total_drops

        except Exception as e:
            print(f"[WARN] Cannot get port stats: {e}")
            return 0, 0, 0

    # =========================
    # ONOS FLOWS
    # =========================
    def _get_flows(self):
        try:
            r = requests.get(self.url_flows, auth=self.auth, timeout=5)
            r.raise_for_status()
            return r.json().get("flows", [])
        except Exception as e:
            print(f"[WARN] Cannot get flows: {e}")
            return []

    # =========================
    # ENTROPY
    # =========================
    def _entropy(self, values):
        if not values:
            return 0.0

        counter = collections.Counter(values)
        total = sum(counter.values())

        result = 0.0
        for count in counter.values():
            p = count / total
            result -= p * math.log2(p)

        return result

    def _get_src_ip_entropy_from_tcpdump(self, duration=0.4):
        """
        Tính src_ip_entropy bằng cách bắt packet tại victim h8.
        Cách này tốt hơn đọc IPV4_SRC từ ONOS flow vì nhiều flow không chứa IPV4_SRC.
        """
        try:
            pid = self._find_mininet_host_pid(self.entropy_host)
            if not pid:
                return 0.0

            cmd = (
                f"sudo mnexec -a {pid} timeout {duration} "
                f"tcpdump -i {self.entropy_iface} -n -tt ip 2>/dev/null"
            )

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=duration + 2
            )

            src_ips = []

            for line in result.stdout.splitlines():
                # Ví dụ:
                # IP 10.0.0.5.12345 > 10.0.0.8.80: Flags [S]
                m = re.search(r"IP\s+(\d+\.\d+\.\d+\.\d+)(?:\.\d+)?\s+>", line)
                if m:
                    src_ips.append(m.group(1))

            return self._entropy(src_ips)

        except Exception as e:
            print(f"[WARN] tcpdump entropy error: {e}")
            return 0.0

    # =========================
    # LATENCY / LOSS FROM MININET HOST
    # =========================
    def _measure_latency(self):
        try:
            pid = self._find_mininet_host_pid(self.latency_host)
            if not pid:
                print(f"[WARN] Cannot find Mininet host {self.latency_host}")
                return 500.0

            cmd = f"sudo mnexec -a {pid} ping -c 1 -W 1 {self.victim_ip}"

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3
            )

            match = re.findall(r"time=([\d.]+)", result.stdout)
            if match:
                return float(match[0])

            return 500.0

        except Exception as e:
            print(f"[WARN] latency error: {e}")
            return 500.0

    def _measure_ping_loss(self):
        try:
            pid = self._find_mininet_host_pid(self.latency_host)
            if not pid:
                return 1.0

            cmd = f"sudo mnexec -a {pid} ping -c 3 -W 1 {self.victim_ip}"

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=6
            )

            match = re.search(r"(\d+(?:\.\d+)?)% packet loss", result.stdout)
            if match:
                return float(match.group(1)) / 100.0

            return 1.0

        except Exception as e:
            print(f"[WARN] ping loss error: {e}")
            return 1.0

    # =========================
    # CONTROLLER CPU
    # =========================
    def _get_controller_cpu(self):
        try:
            result = subprocess.run(
                f"docker stats {self.onos_container} --no-stream --format '{{{{.CPUPerc}}}}'",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3
            )

            value = result.stdout.strip().replace("%", "")
            if value:
                return float(value)

        except Exception as e:
            print("[WARN] docker cpu error:", e)

        return 0.0

    # =========================
    # RAW STATE
    # =========================
    def get_raw_state(self):
        packets, bytes_, drops = self._get_port_stats()
        flows = self._get_flows()
        flow_count = len(flows)

        now = time.time()
        dt = max(now - self.prev_time, 1e-6)

        delta_packets = max(0, packets - self.prev_packets)
        delta_bytes = max(0, bytes_ - self.prev_bytes)
        delta_drops = max(0, drops - self.prev_drops)

        packet_rate = delta_packets / dt
        byte_rate = delta_bytes / dt

        flow_growth_rate = max(0, flow_count - self.prev_flow_count) / dt

        # Ưu tiên packet_loss đo bằng ping từ Mininet host
        packet_loss = self._measure_ping_loss()

        self.prev_packets = packets
        self.prev_bytes = bytes_
        self.prev_drops = drops
        self.prev_flow_count = flow_count
        self.prev_time = now

        return {
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "flow_count": flow_count,
            "flow_growth_rate": flow_growth_rate,
            "src_ip_entropy": self._get_src_ip_entropy_from_tcpdump(),
            "latency": self._measure_latency(),
            "packet_loss": packet_loss,
            "controller_cpu": self._get_controller_cpu()
        }

    def save_to_csv(self, state, filename, label):
        fieldnames = [
            "packet_rate",
            "byte_rate",
            "flow_count",
            "flow_growth_rate",
            "src_ip_entropy",
            "latency",
            "packet_loss",
            "controller_cpu",
            "label"
        ]

        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        file_exists = os.path.exists(filename)

        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            row = state.copy()
            row["label"] = label
            writer.writerow(row)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--onos-ip", type=str, default="127.0.0.1")
    p.add_argument("--victim-ip", type=str, default="10.0.0.8")
    p.add_argument("--latency-host", type=str, default="h1")
    p.add_argument("--entropy-host", type=str, default="h8")
    p.add_argument("--entropy-iface", type=str, default="h8-eth0")
    p.add_argument("--onos-container", type=str, default="onos-controller")
    p.add_argument("--label", type=int, required=True)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--output", type=str, required=True)

    args = p.parse_args()

    collector = ONOSCollector(
        onos_ip=args.onos_ip,
        victim_ip=args.victim_ip,
        latency_host=args.latency_host,
        entropy_host=args.entropy_host,
        entropy_iface=args.entropy_iface,
        onos_container=args.onos_container
    )

    print(f"[*] Collecting label={args.label}")
    print(f"[*] Output={args.output}")

    for i in range(1, args.samples + 1):
        state = collector.get_raw_state()
        collector.save_to_csv(state, args.output, args.label)

        print(f"[{i}/{args.samples}] {state}")
        time.sleep(args.interval)