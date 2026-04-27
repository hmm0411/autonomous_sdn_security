import requests
import numpy as np
import time
import subprocess


class OnlineSDNEnv:

    def __init__(
        self,
        controller_url="http://34.126.64.185:8181/onos/v1",
        username="onos",
        password="rocks",
        polling_interval=2
    ):
        self.controller_url = controller_url
        self.auth = (username, password)
        self.polling_interval = polling_interval
        self.previous_action = 0

    # ==========================================
    # RESET
    # ==========================================
    def reset(self):
        self.previous_action = 0
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

        done = False

        return state, reward, done, {}

    # ==========================================
    # STATE COLLECTION
    # ==========================================
    def _get_state(self):

        flows = self._get_flows()

        packet_rate = sum(f.get("packets", 0) for f in flows)
        byte_rate = sum(f.get("bytes", 0) for f in flows)
        flow_count = len(flows)

        src_ip_entropy = self._compute_entropy(flows)
        latency = self._estimate_latency()

        packet_loss = 0.0
        queue_length = flow_count / 100.0
        controller_cpu = 0.0

        # IMPORTANT: normalize giống offline
        packet_rate /= 1e6
        byte_rate /= 1e9
        flow_count /= 1000.0

        state = np.array([
            packet_rate,
            byte_rate,
            flow_count,
            src_ip_entropy,
            latency,
            packet_loss,
            queue_length,
            controller_cpu,
            self.previous_action
        ], dtype=np.float32)

        return state

    # ==========================================
    # ONOS FLOWS
    # ==========================================
    def _get_flows(self):
        try:
            r = requests.get(
                f"{self.controller_url}/flows",
                auth=self.auth,
                timeout=5
            )
            return r.json().get("flows", [])
        except:
            return []

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
                    return float(line.split("time=")[1].split(" ")[0]) / 1000.0

        except:
            pass

        return 0.0

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

        attacker_ip = "10.0.0.1/32"  # demo: có thể dynamic detect sau

        rule = {
            "priority": 50000,
            "isPermanent": True,
            "deviceId": "of:0000000000000001",
            "treatment": {},
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

        print("BLOCK rule installed for", attacker_ip)


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
                    "rate": 100
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

        rule = {
            "priority": 45000,
            "isPermanent": True,
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
                    {"type": "ETH_TYPE", "ethType": "0x0800"}
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

        attacker_ip = "10.0.0.1/32"

        rule = {
            "priority": 60000,
            "isPermanent": True,
            "deviceId": "of:0000000000000001",
            "treatment": {},
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

        qos_penalty = 0.5 * flow_count + 2.0 * latency

        switching_penalty = 0.2 if action != self.previous_action else 0.0

        return -qos_penalty - switching_penalty