import numpy as np


class OfflineSDNEnv:
    def __init__(self, dataframe, max_steps_per_episode=1000):
        if dataframe is None or dataframe.empty:
            raise ValueError("OfflineSDNEnv received an empty dataframe.")

        self.df = dataframe.reset_index(drop=True)
        self.max_steps_per_episode = max_steps_per_episode
        self.idx = 0
        self.previous_action = 0
        self.current_step = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.previous_action = 0

        max_start = max(0, len(self.df) - self.max_steps_per_episode - 1)

        if max_start > 0:
            self.idx = np.random.randint(0, max_start)
        else:
            self.idx = 0

        return self._get_state(), {}

    def step(self, action):
        row = self.df.iloc[self.idx]

        reward = self._compute_reward(action, row)

        self.previous_action = int(action)

        self.idx += 1
        self.current_step += 1

        done = (
            self.idx >= len(self.df) - 1
            or self.current_step >= self.max_steps_per_episode
        )

        next_state = self._get_state()

        return next_state, reward, done, False, {}

    def _value(self, row, key, default=0.0):
        if key in row:
            value = row[key]
            if value is None:
                return float(default)

            try:
                if np.isnan(value):
                    return float(default)
            except TypeError:
                pass

            return float(value)

        return float(default)

    def _scale(self, value, raw_max):
        """
        Chấp nhận cả dataset đã normalize 0-1 và dataset raw.
        Nếu value nằm trong khoảng nhỏ, giữ nguyên.
        Nếu value lớn, scale theo raw_max.
        """
        value = float(value)

        if value < 0:
            value = abs(value)

        if value <= 1.5:
            return float(np.clip(value, 0.0, 1.0))

        return float(np.clip(value / raw_max, 0.0, 1.0))

    def _get_attack_label(self, row):
        """
        Nhãn chỉ dùng cho offline reward.
        Không đưa label/attack_indicator vào state runtime.
        Label convention:
        0 normal
        1 ddos
        2 flow_overflow
        3 ip_spoofing
        4 packet_in_flood
        5 port_scanning
        """
        if "label" in row:
            try:
                return int(row["label"])
            except Exception:
                return 0

        if "attack_indicator" in row:
            try:
                return int(round(float(row["attack_indicator"]) * 5))
            except Exception:
                return 0

        return 0

    def _get_state(self):
        """
        CHUẨN 8 CHIỀU:
        [
            packet_rate,
            byte_rate,
            flow_count,
            flow_growth_rate,
            src_ip_entropy,
            latency,
            packet_loss,
            controller_cpu,
        ]
        """
        row = self.df.iloc[self.idx]

        return np.array(
            [
                self._value(row, "packet_rate"),
                self._value(row, "byte_rate"),
                self._value(row, "flow_count"),
                self._value(row, "flow_growth_rate", 0.0),
                self._value(row, "src_ip_entropy"),
                self._value(row, "latency"),
                self._value(row, "packet_loss"),
                self._value(row, "controller_cpu"),
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, action, row):
        action = int(action)

        packet_rate = self._value(row, "packet_rate")
        byte_rate = self._value(row, "byte_rate")
        flow_count = self._value(row, "flow_count")
        flow_growth_rate = self._value(row, "flow_growth_rate", 0.0)
        src_ip_entropy = self._value(row, "src_ip_entropy")
        latency = self._value(row, "latency")
        packet_loss = self._value(row, "packet_loss")
        controller_cpu = self._value(row, "controller_cpu")

        attack_label = self._get_attack_label(row)

        # Normalize robustly for reward calculation.
        packet_norm = self._scale(packet_rate, 5000.0)
        byte_norm = self._scale(byte_rate, 5_000_000.0)
        flow_norm = self._scale(flow_count, 200.0)
        growth_norm = self._scale(flow_growth_rate, 100.0)
        entropy_norm = self._scale(src_ip_entropy, 8.0)
        latency_norm = self._scale(latency, 300.0)

        if packet_loss <= 1.0:
            loss_norm = float(np.clip(packet_loss, 0.0, 1.0))
        else:
            loss_norm = float(np.clip(packet_loss / 100.0, 0.0, 1.0))

        cpu_norm = self._scale(controller_cpu, 100.0)

        qos_penalty = (
            0.10 * latency_norm
            + 0.20 * loss_norm
            + 0.10 * cpu_norm
            + 0.10 * growth_norm
        )

        action_costs = {
            0: 0.00,  # no_action
            1: 0.12,  # block
            2: 0.08,  # limit bandwidth
            3: 0.10,  # redirect honeypot
            4: 0.75,  # isolate device
        }

        action_cost = action_costs.get(action, 0.20)
        switching_penalty = 0.03 if action != self.previous_action else 0.0

        risk_score = (
            0.20 * packet_norm
            + 0.15 * byte_norm
            + 0.15 * flow_norm
            + 0.20 * growth_norm
            + 0.15 * entropy_norm
            + 0.15 * cpu_norm
        )

        # =========================
        # Security reward
        # =========================
        # 0 = normal
        if attack_label == 0:
            if action == 0:
                security_reward = 2.50
            elif action == 2:
                security_reward = -2.00
            elif action == 3:
                security_reward = -2.50
            elif action == 1:
                security_reward = -3.50
            elif action == 4:
                security_reward = -4.50
            else:
                security_reward = -3.00

        # 1 = ddos
        elif attack_label == 1:
            if action == 2:
                security_reward = 2.50
            elif action == 1:
                security_reward = 0.80
            elif action == 3:
                security_reward = 0.40
            elif action == 4:
                security_reward = -0.80
            elif action == 0:
                security_reward = -3.00
            else:
                security_reward = -1.50

        # 2 = flow_overflow
        elif attack_label == 2:
            if action == 1:
                security_reward = 2.50
            elif action == 2:
                security_reward = 2.00
            elif action == 4:
                security_reward = 0.50
            elif action == 3:
                security_reward = -0.50
            elif action == 0:
                security_reward = -3.00
            else:
                security_reward = -1.50

        # 3 = ip_spoofing
        elif attack_label == 3:
            if action == 4 and risk_score >= 0.55:
                security_reward = 2.80
            elif action == 1:
                security_reward = 2.20
            elif action == 4:
                security_reward = 1.20
            elif action == 3:
                security_reward = 0.80
            elif action == 2:
                security_reward = -0.30
            elif action == 0:
                security_reward = -3.20
            else:
                security_reward = -1.50

        # 4 = packet_in_flood
        elif attack_label == 4:
            if action == 2:
                security_reward = 2.60
            elif action == 1:
                security_reward = 1.70
            elif action == 4:
                security_reward = 0.30
            elif action == 3:
                security_reward = -0.30
            elif action == 0:
                security_reward = -3.20
            else:
                security_reward = -1.50

        # 5 = port_scanning
        elif attack_label == 5:
            if action == 3:
                security_reward = 2.80
            elif action == 1:
                security_reward = 0.80
            elif action == 2:
                security_reward = 0.30
            elif action == 4:
                security_reward = -0.50
            elif action == 0:
                security_reward = -2.80
            else:
                security_reward = -1.50

        else:
            security_reward = -1.00

        severity_bonus = 0.0
        if attack_label != 0 and action != 0:
            severity_bonus = 0.40 * risk_score

        false_positive_penalty = 0.0
        if attack_label == 0 and action != 0:
            false_positive_penalty = 1.20

        reward = (
            security_reward
            + severity_bonus
            - qos_penalty
            - action_cost
            - switching_penalty
            - false_positive_penalty
        )

        return float(np.clip(reward, -5.0, 3.0))