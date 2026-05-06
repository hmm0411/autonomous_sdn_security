from urllib import response
import json
import requests
import numpy as np
import time
import subprocess
from requests.auth import HTTPBasicAuth
from rl_engine.state_builder import StateBuilder

class OnlineSDNEnv:

    def __init__(
        self,
        controller_url="http://34.126.64.185:8181/onos/v1",
        username="onos",
        password="rocks",
        polling_interval=2
    ):
        self.controller_url = controller_url
        self.auth = HTTPBasicAuth(username, password)
        self.polling_interval = polling_interval
        self.previous_action = 0
        self.prev_packets = 0
        self.prev_bytes = 0
        self.prev_time = time.time()
        self.baseline = {
            "flow_count": None,
            "entropy": None,
            "packet_rate": None,
            "latency": None
        }
        self.builder = StateBuilder()

    # ==========================================
    # RESET
    # ==========================================
    def reset(self):
        self.previous_action = 0
        self.builder.prev_action = 0
        self.prev_packets = 0
        self.prev_bytes = 0
        self.prev_time = time.time()
        return self._get_state()

    # ==========================================
    # STEP
    # ==========================================
    def step(self, action):

        self._apply_action(action)

        time.sleep(1)

        state = self._get_state()
        reward = self._compute_reward(state, action)

        self.previous_action = action
        self.builder.prev_action = action

        return state, reward, False, {}
    # ==========================================
    # STATE COLLECTION
    # ==========================================
    def _get_state(self):

        flows = self._get_flows()

        current_packets = sum(f.get("packets", 0) for f in flows)
        current_bytes = sum(f.get("bytes", 0) for f in flows)
        flow_count = len(flows)

        now = time.time()
        dt = max(now - self.prev_time, 1e-6)

        packet_rate = (current_packets - self.prev_packets) / dt
        byte_rate = (current_bytes - self.prev_bytes) / dt

        self.prev_packets = current_packets
        self.prev_bytes = current_bytes
        self.prev_time = now

        entropy = self._compute_entropy(flows)
        latency = self._estimate_latency()

        raw = {
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "flow_count": flow_count,
            "latency": latency,
            "packet_loss": 0.0,
            "src_ip_entropy": entropy,
            "queue_length": flow_count,
            "controller_cpu": 0.0
        }

        state = self.builder.build(raw)

        print(
            f"Flow={flow_count} | "
            f"PacketRate={packet_rate:.2f} | "
            f"Entropy={entropy:.2f}"
        )

        return state

    # ==========================================
    # ONOS FLOWS
    # ==========================================
    def push_flow(self, device_id, flow):
        url = f"{self.controller_url}/flows/{device_id}"
        headers = {"Content-Type": "application/json"}

        r = requests.post(
            url,
            json=flow,
            headers=headers,
            auth=self.auth,
            timeout=5
        )

        if r.status_code not in [200, 201, 204]:
            print("Failed to push flow:", r.status_code, r.text)
        else:
            print("Flow pushed successfully:", r.status_code)

        return r.status_code
    
    def _get_flows(self):
        try:
            r = requests.get(
                f"{self.controller_url}/flows",
                auth=self.auth,
                timeout=5
            )
            if r.status_code == 200:
                return r.json().get("flows", [])
            return []
        except:
            return []
        
    def _detect_top_src_ip(self, flows):
        ip_count: dict[str, int] = {}

        for f in flows:
            for c in f.get("selector", {}).get("criteria", []):
                if c.get("type") == "IPV4_SRC":
                    ip = c.get("ip")
                    ip_count[ip] = ip_count.get(ip, 0) + 1

        if not ip_count:
            return None

        return max(ip_count.keys(), key=lambda k: ip_count[k])

    # ==========================================
    # ENTROPY
    # ==========================================
    def _compute_entropy(self, flows):
        src_ips = []

        for f in flows:
            for c in f.get("selector", {}).get("criteria", []):
                if c.get("type") == "IPV4_SRC":
                    src_ips.append(c.get("ip", ""))

        if not src_ips:
            return 0.0

        unique, counts = np.unique(src_ips, return_counts=True)
        probs = counts / counts.sum()

        return float(-np.sum(probs * np.log2(probs + 1e-8)))

    # ==========================================
    # LATENCY REAL PING
    # ==========================================
    def _estimate_latency(self):
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "10.0.0.1"],
                capture_output=True,
                text=True,
                timeout=2
            )

            for line in result.stdout.split("\n"):
                if "time=" in line:
                    return float(line.split("time=")[1].split(" ")[0])
        except:
            pass

        return 0.0
    
    # ==========================================
    # DETECT ATTACK
    # ==========================================
    
    def _detect_attack(self, state):

        packet_rate = state[0]
        flow_count = state[2]
        entropy = state[3]
        latency = state[4]

        if flow_count > 0.15:
            return 1

        if entropy > 2.0:
            return 1

        if packet_rate > 0.01:
            return 1

        return 0

    # ==========================================
    # ACTION MAPPING
    # ==========================================
    def _apply_action(self, action):

        if action == 0:
            return

        if action == 1:
            self._block_suspicious_flow()

        elif action == 2:
            self._limit_bandwidth()

        elif action == 3:
            self._redirect_traffic()

        elif action == 4:
            self._isolate_device()

    # ==========================================
    # BLOCK SUSPICIOUS FLOW (DROP RULE)
    # ==========================================
    def _block_suspicious_flow(self):

        flows = self._get_flows()
        attacker_ip = self._detect_top_src_ip(flows)

        if attacker_ip is None:
            return

        rule = {
            "priority": 50000,
            "timeout": 30,
            "isPermanent": False,
            "deviceId": "of:0000000000000001",
            "treatment": {
                "instructions": [
                    {"type": "METER", "meterId": "1"},
                    {"type": "OUTPUT", "port": "1"}
                ]
            },
            "selector": {
                "criteria": [
                    {"type": "ETH_TYPE", "ethType": "0x0800"},
                    {"type": "IPV4_SRC", "ip": attacker_ip}
                ]
            }
        }

        self.push_flow("of:0000000000000001", rule)

        print("BLOCK:", attacker_ip)


    # ==========================================
    # LIMIT BANDWIDTH (METER RULE)
    # ==========================================
    def _limit_bandwidth(self):

        meter = {
            "deviceId": "of:0000000000000001",
            "unit": "KB_PER_SEC",
            "bands": [
                {
                    "type": "DROP",
                    "rate": 200
                }
            ]
        }

        requests.post(
            f"{self.controller_url}/meters/of:0000000000000001",
            json=meter,
            auth=self.auth,
            timeout=5
        )

        print("Meter rule installed")


    # ==========================================
    # REDIRECT TO HONEYPOT
    # ==========================================
    def _redirect_traffic(self):
        flows = self._get_flows()
        attacker_ip = self._detect_top_src_ip(flows)

        if attacker_ip is None:
            return
        
        rule = {
            "priority": 45000,
            "timeout": 30,
            "isPermanent": False,
            "deviceId": "of:0000000000000001",
            "treatment": {
                "instructions": [
                    {
                        "type": "OUTPUT",
                        "port": "2"  # port nối sang s3
                    }
                ]
            },
            "selector": {
                "criteria": [
                    {"type": "ETH_TYPE", "ethType": "0x0800"},
                    {"type": "IPV4_SRC", "ip": attacker_ip}
                ]
            }
        }

        requests.post(
            f"{self.controller_url}/flows/of:0000000000000001",
            json=rule,
            auth=self.auth,
            timeout=5
        )

        print("Traffic redirected to Honeypot")


    # ==========================================
    # ISOLATE DEVICE (DROP ALL FROM HOST)
    # ==========================================
    def _isolate_device(self):

        flows = self._get_flows()
        attacker_ip = self._detect_top_src_ip(flows)

        if attacker_ip is None:
            return

        rule = {
            "priority": 60000,
            "isPermanent": False,
            "timeout": 60,
            "deviceId": "of:0000000000000001",
            "treatment": {
                "instructions": [
                    {"type": "METER", "meterId": "1"},
                    {"type": "OUTPUT", "port": "1"}
                ]
            },
            "selector": {
                "criteria": [
                    {"type": "IPV4_SRC", "ip": attacker_ip}
                ]
            }
        }

        requests.post(
            f"{self.controller_url}/flows/of:0000000000000001",
            json=rule,
            auth=self.auth,
            timeout=5
        )

        print("Device isolated:", attacker_ip)

    # ==========================================
    # REWARD ONLINE
    # ==========================================
    def _compute_reward(self, state, action):

        flow_count = state[2]
        latency = state[4]

        qos_penalty = 2.0 * flow_count + 1.0 * latency
        switching_penalty = 0.3 if action != self.previous_action else 0.0

        return 1.0 - qos_penalty - switching_penalty