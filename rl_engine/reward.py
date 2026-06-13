class Reward:
    """
    Runtime/evaluation reward for 8-dim state.

    Action map:
    0 no_action
    1 block_suspicious_flow
    2 limit_bandwidth
    3 redirect_traffic
    4 isolate_device
    """

    def __init__(self):
        self.last_action = 0

    def calculate(self, raw, action):
        if raw is None:
            return -1.0

        action = int(action)

        packet_rate = float(raw.get("packet_rate", 0.0) or 0.0)
        flow_count = float(raw.get("flow_count", 0.0) or 0.0)
        flow_growth = abs(float(raw.get("flow_growth_rate", 0.0) or 0.0))
        latency = float(raw.get("latency", 0.0) or 0.0)
        packet_loss = float(raw.get("packet_loss", 0.0) or 0.0)
        controller_cpu = float(raw.get("controller_cpu", 0.0) or 0.0)

        threat = 0.0
        threat += min(packet_rate / 1000.0, 3.0)
        threat += min(flow_count / 200.0, 2.0)
        threat += min(flow_growth / 50.0, 2.0)
        threat += min(latency / 200.0, 1.5)
        threat += min(packet_loss * 5.0, 2.0)
        threat += min(controller_cpu / 100.0, 1.0)

        action_cost = {
            0: 0.00,
            1: 0.35,
            2: 0.18,
            3: 0.22,
            4: 0.45,
        }.get(action, 0.5)

        switch_penalty = 0.08 if action != self.last_action else 0.0

        if threat < 0.8:
            if action == 0:
                reward = 1.0
            else:
                reward = 0.3 - action_cost
        else:
            if action == 0:
                reward = -threat
            elif action == 2:
                reward = 1.2 - 0.18 * threat - action_cost
            elif action in (1, 3, 4):
                reward = 1.0 - 0.20 * threat - action_cost
            else:
                reward = -0.5

        reward -= switch_penalty
        self.last_action = action

        return float(max(-5.0, min(2.0, reward)))