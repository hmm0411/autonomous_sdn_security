import numpy as np

class Reward:

    def __init__(self):
        self.prev_action = 0

        # ===== weights (tune được) =====
        self.w_latency = 0.8
        self.w_loss = 1.5
        self.w_flow = 0.6
        self.w_switch = 0.2
        self.w_action = 0.5

        # ===== thresholds =====
        self.flow_threshold = 50
        self.latency_threshold = 20

    def calculate(self, raw, action):

        latency = raw.get("latency", 0)
        packet_loss = raw.get("packet_loss", 0)
        flow_count = raw.get("flow_count", 0)

        reward = 0

        # ===== 1. QoS penalty =====
        reward -= self.w_latency * (latency / 100)
        reward -= self.w_loss * packet_loss

        # ===== 2. Adaptive overload penalty =====
        overload = max(0, flow_count - self.flow_threshold)
        reward -= self.w_flow * (overload / self.flow_threshold)

        # ===== 3. Switching penalty =====
        if action != self.prev_action:
            reward -= self.w_switch

        # ===== 4. Action shaping =====
        if flow_count < self.flow_threshold:
            # mạng bình thường → ưu tiên no action
            if action == 0:
                reward += self.w_action
            else:
                reward -= self.w_action * 0.5
        else:
            # nghi ngờ overload → cần hành động
            if action != 0:
                reward += self.w_action
            else:
                reward -= self.w_action

        # ===== normalize (optional nhưng nên có) =====
        reward = np.clip(reward, -5, 5)

        self.prev_action = action

        return float(reward)