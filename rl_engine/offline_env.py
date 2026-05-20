import numpy as np


class OfflineSDNEnv:
    def __init__(self, dataframe, max_steps_per_episode=1000):
        self.df = dataframe.reset_index(drop=True)
        self.max_steps_per_episode = max_steps_per_episode
        self.idx = 0
        self.previous_action = 0
        self.current_step = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.idx = 0
        self.previous_action = 0
        self.current_step = 0

        return self._get_state(), {}

    def step(self, action):
        row = self.df.iloc[self.idx]

        reward = self._compute_reward(action, row)

        self.previous_action = int(action)

        self.idx += 1
        self.current_step += 1

        done = (
            self.idx >= len(self.df) - 1 or
            self.current_step >= self.max_steps_per_episode
        )

        next_state = self._get_state()

        return next_state, reward, done, False, {}

    def _get_state(self):
        row = self.df.iloc[self.idx]

        previous_action_norm = self.previous_action / 4.0

        return np.array([
            row["packet_rate"],
            row["byte_rate"],
            row["flow_count"],
            row["flow_growth_rate"],
            row["src_ip_entropy"],
            row["latency"],
            row["packet_loss"],
            row["controller_cpu"],
            previous_action_norm
        ], dtype=np.float32)

    def _compute_reward(self, action, row):
        action = int(action)

        # Các feature đã được scale về khoảng 0-1 sau QuantileTransformer
        packet_rate = float(row["packet_rate"])
        byte_rate = float(row["byte_rate"])
        flow_count = float(row["flow_count"])
        flow_growth_rate = float(row["flow_growth_rate"])
        src_ip_entropy = float(row["src_ip_entropy"])
        latency = float(row["latency"])
        packet_loss = float(row["packet_loss"])
        controller_cpu = float(row["controller_cpu"])
        attack_indicator = float(row["attack_indicator"])

        # attack_indicator: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        attack_label = int(round(attack_indicator * 5))

        # =========================
        # 1. QoS / system penalty
        # =========================
        # Penalty này phải nhỏ, nếu không reward sẽ âm nặng.
        qos_penalty = (
            0.20 * latency +
            0.35 * packet_loss +
            0.15 * controller_cpu +
            0.15 * flow_growth_rate
        )

        # =========================
        # 2. Action cost
        # =========================
        # Hành động càng mạnh thì càng tốn chi phí.
        action_costs = {
            0: 0.00,  # no_action
            1: 0.10,  # block
            2: 0.08,  # limit bandwidth
            3: 0.12,  # redirect honeypot
            4: 0.25,  # isolate device
        }

        action_cost = action_costs.get(action, 0.20)

        # =========================
        # 3. Switching penalty
        # =========================
        # Phạt nhỏ nếu đổi action liên tục.
        switching_penalty = 0.05 if action != self.previous_action else 0.0

        # =========================
        # 4. Security reward
        # =========================

        # Normal traffic
        if attack_label == 0:
            if action == 0:
                security_reward = 1.20
            elif action == 2:
                # limit bandwidth trên normal vẫn sai nhưng nhẹ hơn block/isolate
                security_reward = -0.60
            elif action == 3:
                security_reward = -0.80
            elif action == 1:
                security_reward = -1.00
            elif action == 4:
                security_reward = -1.40
            else:
                security_reward = -1.00

        # DDoS
        elif attack_label == 1:
            if action == 2:
                security_reward = 1.60
            elif action in [1, 3]:
                security_reward = 1.10
            elif action == 4:
                security_reward = 0.60
            else:
                security_reward = -1.50

        # Flow overflow
        elif attack_label == 2:
            if action in [1, 2]:
                security_reward = 1.50
            elif action == 4:
                security_reward = 0.90
            elif action == 3:
                security_reward = 0.40
            else:
                security_reward = -1.50

        # IP spoofing
        elif attack_label == 3:
            if action in [1, 4]:
                security_reward = 1.60
            elif action == 3:
                security_reward = 1.00
            elif action == 2:
                security_reward = 0.40
            else:
                security_reward = -1.60

        # Packet-in flood
        elif attack_label == 4:
            if action in [1, 2]:
                security_reward = 1.50
            elif action == 4:
                security_reward = 0.90
            elif action == 3:
                security_reward = 0.50
            else:
                security_reward = -1.60

        # Port scanning
        elif attack_label == 5:
            if action in [1, 3]:
                security_reward = 1.50
            elif action == 2:
                security_reward = 0.90
            elif action == 4:
                security_reward = 0.60
            else:
                security_reward = -1.40

        else:
            security_reward = -1.0

        # =========================
        # 5. Severity bonus
        # =========================
        # Nếu attack mạnh và agent chọn hành động phòng thủ, thưởng thêm nhẹ.
        risk_score = (
            0.25 * packet_rate +
            0.20 * byte_rate +
            0.20 * flow_growth_rate +
            0.20 * src_ip_entropy +
            0.15 * controller_cpu
        )

        if attack_label != 0 and action != 0:
            severity_bonus = 0.30 * risk_score
        else:
            severity_bonus = 0.0

        # Nếu normal mà dùng action mạnh khi risk thấp thì phạt thêm.
        false_positive_penalty = 0.0
        if attack_label == 0 and action != 0 and risk_score < 0.40:
            false_positive_penalty = 0.30

        reward = (
            security_reward
            + severity_bonus
            - qos_penalty
            - action_cost
            - switching_penalty
            - false_positive_penalty
        )

        # Giữ reward trong khoảng vừa phải để train ổn định.
        return float(np.clip(reward, -2.0, 2.0))