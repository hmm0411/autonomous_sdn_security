class Reward:
    """
    Reward cho runtime benchmark.

    Nguyên tắc:
    - Không dùng flow_count đơn lẻ để kết luận attack vì ONOS có thể giữ flow cũ.
    - Normal + action 0: thưởng cao.
    - Normal + action phòng vệ: phạt overreaction.
    - Attack + action 0: phạt nặng.
    - Attack + action phòng vệ: thưởng cao hơn no_action.
    """

    def __init__(self):
        pass

    @staticmethod
    def _f(raw, key, default=0.0):
        try:
            return float(raw.get(key, default) or default)
        except Exception:
            return float(default)

    def threat_score(self, raw):
        if raw is None:
            return 0.0

        pps = self._f(raw, "packet_rate")
        bps = self._f(raw, "byte_rate")
        growth = abs(self._f(raw, "flow_growth_rate"))
        entropy = self._f(raw, "src_ip_entropy")
        latency = self._f(raw, "latency")
        loss = self._f(raw, "packet_loss")
        cpu = self._f(raw, "controller_cpu")

        score = 0.0

        # Không dùng flow_count ở đây để tránh false positive do flow cũ.
        score += min(pps / 500.0, 4.0)
        score += min(bps / 100000.0, 2.0)
        score += min(growth / 5.0, 2.0)
        score += min(entropy * 2.0, 2.0)
        score += min(max(latency - 5.0, 0.0) / 20.0, 4.0)
        score += min(loss / 0.01, 4.0)
        score += min(max(cpu - 10.0, 0.0) / 25.0, 3.0)

        return score

    def is_attack_like(self, raw):
        if raw is None:
            return False

        pps = self._f(raw, "packet_rate")
        bps = self._f(raw, "byte_rate")
        growth = abs(self._f(raw, "flow_growth_rate"))
        latency = self._f(raw, "latency")
        loss = self._f(raw, "packet_loss")
        cpu = self._f(raw, "controller_cpu")

        return (
            pps >= 100
            or bps >= 10000
            or growth >= 3
            or latency >= 8
            or loss >= 0.001
            or cpu >= 18
            or self.threat_score(raw) >= 3
        )

    def calculate(self, raw, action):
        if raw is None:
            return 0.0

        action = int(action)
        attack = self.is_attack_like(raw)
        score = self.threat_score(raw)

        latency = self._f(raw, "latency")
        loss = self._f(raw, "packet_loss")
        cpu = self._f(raw, "controller_cpu")

        qos_penalty = 0.0
        qos_penalty += min(max(latency - 5.0, 0.0) / 50.0, 3.0)
        qos_penalty += min(loss / 0.05, 3.0)
        qos_penalty += min(max(cpu - 10.0, 0.0) / 50.0, 2.0)

        # Normal traffic
        if not attack:
            if action == 0:
                return 1.0
            return -1.0

        # Attack traffic
        if action == 0:
            return max(-5.0, -1.0 - 0.5 * score - 0.3 * qos_penalty)

        action_cost = {
            1: 0.30,  # block
            2: 0.20,  # limit
            3: 0.25,  # redirect
            4: 0.40,  # isolate
        }.get(action, 0.30)

        reward = 0.8 + 0.18 * score - 0.20 * qos_penalty - action_cost

        return max(-5.0, min(3.0, reward))
