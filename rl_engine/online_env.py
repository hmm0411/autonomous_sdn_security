from urllib import response
import json
import requests
import numpy as np
import time
import subprocess
from requests.auth import HTTPBasicAuth
from rl_engine.state_builder import StateBuilder
from traffic_generator.onos_collector import ONOSCollector

class OnlineSDNEnv:

    def __init__(
        self,
        controller_url="http://controller:8181/onos/v1",
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
        self.prev_packets = 0
        self.prev_bytes = 0
        self.prev_time = time.time()
        return self._get_state()

    # ==========================================
    # STEP
    # ==========================================
    def step(self, action):
        action = int(action)

        # Lấy state trước khi apply action
        state_before = self._get_state()

        # Apply action vào ONOS
        self._apply_action(action)

        # Chờ mạng phản ứng sau khi cài flow/meter
        time.sleep(self.polling_interval)

        # Lấy state sau khi apply action
        next_state = self._get_state()

        # Tính reward online dựa trên state trước/sau
        reward = self._compute_online_reward(
            state_before=state_before,
            next_state=next_state,
            action=action
        )

        self.previous_action = action
        # self.builder.prev_action = action

        done = False

        return next_state, reward, done, False, {}
    # ==========================================
    # STATE COLLECTION
    # ==========================================
    def _get_state(self):
        if not hasattr(self, "collector"):
            base_url = self.controller_url.replace("http://", "").replace("/onos/v1", "")
            onos_ip = base_url.split(":")[0]

            self.collector = ONOSCollector(
                onos_ip=onos_ip,
                victim_ip="10.0.0.8",
                latency_host="h1",
                entropy_host="h8",
                entropy_iface="h8-eth0",
                onos_container="onos-controller"
            )

        raw = self.collector.get_raw_state()

        if raw is None:
            print("Cảnh báo: Không kết nối được ONOS! Dùng state mặc định.")
            raw = {
                "packet_rate": 0.0, "byte_rate": 0.0, "flow_count": 0,
                "flow_growth_rate": 0.0, "src_ip_entropy": 0.0,
                "latency": 0.0, "packet_loss": 0.0, "controller_cpu": 0.0
            }
        state = self.builder.build(raw)

        print(
            f"PacketRate={raw['packet_rate']:.2f} | "
            f"Flow={raw['flow_count']} | "
            f"Growth={raw['flow_growth_rate']:.2f} | "
            f"Entropy={raw['src_ip_entropy']:.2f} | "
            f"Latency={raw['latency']:.2f} | "
            f"CPU={raw['controller_cpu']:.2f}"
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
        flow_growth_rate = state[3]
        entropy = state[4]
        latency = state[5]

        if flow_growth_rate > 0.2:
            return 1
        if entropy > 0.5:
            return 1
        if packet_rate > 0.2:
            return 1
        if latency > 0.5:
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
                "instructions": []
            },
            "selector": {
                "criteria": [
                    {"type": "ETH_TYPE", "ethType": "0x0800"},
                    {"type": "IPV4_SRC", "ip": attacker_ip},
                    {"type": "IPV4_DST", "ip": "10.0.0.8/32"}
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
                        "port": "9"  # port nối sang s3
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
                "instructions": []
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
    def _compute_online_reward(self, state_before, next_state, action):
        action = int(action)

        # State format:
        # 0 packet_rate
        # 1 byte_rate
        # 2 flow_count
        # 3 flow_growth_rate
        # 4 src_ip_entropy
        # 5 latency
        # 6 packet_loss
        # 7 controller_cpu
        # 8 previous_action

        before_packet_rate = float(state_before[0])
        before_flow_growth = float(state_before[3])
        before_entropy = float(state_before[4])
        before_latency = float(state_before[5])
        before_loss = float(state_before[6])
        before_cpu = float(state_before[7])

        after_packet_rate = float(next_state[0])
        after_flow_growth = float(next_state[3])
        after_entropy = float(next_state[4])
        after_latency = float(next_state[5])
        after_loss = float(next_state[6])
        after_cpu = float(next_state[7])

        # Risk trước và sau khi apply action
        risk_before = (
            0.25 * before_packet_rate +
            0.25 * before_flow_growth +
            0.20 * before_entropy +
            0.15 * before_latency +
            0.10 * before_loss +
            0.05 * before_cpu
        )

        risk_after = (
            0.25 * after_packet_rate +
            0.25 * after_flow_growth +
            0.20 * after_entropy +
            0.15 * after_latency +
            0.10 * after_loss +
            0.05 * after_cpu
        )

        # Nếu action làm risk giảm thì thưởng
        improvement = risk_before - risk_after

        # Chi phí hành động
        action_costs = {
            0: 0.00,
            1: 0.10,
            2: 0.08,
            3: 0.12,
            4: 0.50,
        }

        action_cost = action_costs.get(action, 0.20)

        # Phạt đổi action liên tục
        switching_penalty = 0.05 if action != self.previous_action else 0.0

        # Nếu risk thấp mà vẫn dùng action mạnh thì phạt false positive
        false_positive_penalty = 0.0
        if risk_before < 0.30 and action != 0:
            false_positive_penalty = 0.60

        # Nếu risk cao mà no_action thì phạt
        miss_attack_penalty = 0.0
        if risk_before > 0.55 and action == 0:
            miss_attack_penalty = 0.80

        reward = (
            1.5 * improvement
            - action_cost
            - switching_penalty
            - false_positive_penalty
            - miss_attack_penalty
        )

        return float(np.clip(reward, -2.0, 2.0))